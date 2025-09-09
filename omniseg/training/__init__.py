import os
import math
import random
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader, Dataset

# -----------------------------
# Config and utils
# -----------------------------
@dataclass
class MuZeroConfig:
    # Data / DB
    use_cache: bool = True
    cache_dir: str = "./cache"
    symbols_to_keep: tuple = None
    per_symbol: bool = True  # streaming by symbol
    use_synthetic_data: bool = True  # If True, generate synthetic local data for testing

    # Environment
    lookback_window: int = 10
    action_space_size: int = 3  # 0: Hold, 1: Buy, 2: Sell

    # MCTS (Search)
    num_simulations: int = 25
    discount: float = 0.997
    pb_c_base: float = 19652
    pb_c_init: float = 1.25
    root_dirichlet_alpha: float = 0.25
    root_exploration_fraction: float = 0.25

    # Training
    batch_size: int = 16
    replay_buffer_size: int = 2000
    num_unroll_steps: int = 5
    td_steps: int = 10
    lr_init: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 5
    steps_per_epoch: int = 200  # Batches per epoch

    # System
    parallel_workers: int = 2
    min_initial_games: int = 25

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# (Optional) DB Engine / Synthetic data
# -----------------------------
def get_engine():
    db_user = os.environ.get("DB_USER", "postgres")
    db_password = os.environ.get("DB_PASSWORD", "changeme")
    db_host = os.environ.get("DB_HOST", "127.0.0.1")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "stocks_db")
    return create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

price_cols = ['datetime', 'symbol', 'open', 'low', 'high', 'close', 'volume', 'sent_positive', 'sent_negative', 'sent_neutral']

def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype('category')
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    for c in df.select_dtypes(include=['int64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='integer')
    return df

def ensure_cache_dir(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)

def generate_synthetic_symbol(symbol: str, length: int = 500) -> pd.DataFrame:
    # Random walk close prices with volume & fake sentiment
    rng = np.random.RandomState(hash(symbol) % (2**32))
    
    steps = rng.normal(loc=0.0, scale=1.0, size=length)
    price = 100 + np.cumsum(steps)
    volume = np.maximum(1, rng.lognormal(mean=5.0, sigma=1.0, size=length))
    
    df = pd.DataFrame({
        'datetime': pd.date_range('2020-01-01', periods=length, freq='H'),
        'symbol': symbol,
        'open': price + rng.normal(0, 0.1, size=length),
        'low': price - np.abs(rng.normal(0, 0.5, size=length)),
        'high': price + np.abs(rng.normal(0, 0.5, size=length)),
        'close': price,
        'volume': volume,
        'sent_positive': rng.rand(length),
        'sent_negative': rng.rand(length),
        'sent_neutral': rng.rand(length),
    })
    
    return downcast_df(df)

def load_data_from_db_or_synthetic(config: MuZeroConfig):
    # If synthetic mode, yield synthetic frames per symbol
    if config.use_synthetic_data:
        symbols = config.symbols_to_keep or ('AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN')
        for s in symbols:
            df = generate_synthetic_symbol(s, length=800)
            # simple train/val/test splits
            n = len(df)
            train_df = df.iloc[: int(0.7*n)].copy()
            val_df = df.iloc[int(0.7*n): int(0.85*n)].copy()
            test_df = df.iloc[int(0.85*n):].copy()
            yield s, {'train': train_df, 'val': val_df, 'test': test_df}
        return

    # Otherwise attempt DB streaming (fallback to synthetic if DB not configured)
    try:
        ensure_cache_dir(config.cache_dir)
        engine = get_engine()
        if config.symbols_to_keep is None:
            symbols_df = pd.read_sql("SELECT DISTINCT symbol FROM prod.symbols", engine)
            symbols = tuple(symbols_df['symbol'].unique())
        else:
            symbols = tuple(config.symbols_to_keep)

        symbols_sql = ",".join(f"'{s}'" for s in symbols)
        main_split_queries = {
            'train': f"SELECT {', '.join(price_cols)} FROM prod.historic_data_training WHERE symbol IN ({symbols_sql})",
            'val':   f"SELECT {', '.join(price_cols)} FROM prod.historic_data_val WHERE symbol IN ({symbols_sql})",
            'test':  f"SELECT {', '.join(price_cols)} FROM prod.historic_data_test WHERE symbol IN ({symbols_sql})"
        }

        for sym in symbols:
            sym_safe = sym.replace('/', '_')
            train_q = main_split_queries['train'].replace(f"({symbols_sql})", f"('{sym}')")
            val_q = main_split_queries['val'].replace(f"({symbols_sql})", f"('{sym}')")
            test_q = main_split_queries['test'].replace(f"({symbols_sql})", f"('{sym}')")
            train_df = pd.read_sql_query(train_q, engine, parse_dates=['datetime'])
            val_df = pd.read_sql_query(val_q, engine, parse_dates=['datetime'])
            test_df = pd.read_sql_query(test_q, engine, parse_dates=['datetime'])
            for df in [train_df, val_df, test_df]:
                if not df.empty: downcast_df(df)
            yield sym, {'train': train_df, 'val': val_df, 'test': test_df}
    except Exception as e:
        # If DB fails, fallback to synthetic
        print("DB load failed or not configured, using synthetic data. Error:", e)
        config.use_synthetic_data = True
        yield from load_data_from_db_or_synthetic(config)

# -----------------------------
# Environment: Simulates the stock trading game
# -----------------------------
class StockEnv:
    def __init__(self, df: pd.DataFrame, config: MuZeroConfig):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.current_step = 0
        self.position = 0  # -1 = short, 0 = flat, 1 = long
        self.initial_cash = 10000.0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.portfolio_value = self.cash

    def reset(self):
        # start after lookback window so we have context
        self.current_step = self.config.lookback_window
        self.position = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.portfolio_value = self.cash
        return self._get_obs()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        # ✅ FIX 1: Store portfolio value before the step to calculate per-step reward
        last_portfolio_value = self.portfolio_value
        
        # price at current step
        price = float(self.df.loc[self.current_step, 'close'])

        # Execute a simple market order with all-in/all-out for simplicity
        if action == 1 and self.position <= 0:  # buy
            self.shares = self.cash / (price + 1e-8)
            self.cash = 0.0
            self.position = 1
        elif action == 2 and self.position >= 0:  # sell/short (interpreted as going flat)
            self.cash = self.shares * price
            self.shares = 0.0
            self.position = 0

        # advance
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # portfolio valuation (use last price observed)
        price_next = float(self.df.loc[min(self.current_step, len(self.df)-1), 'close'])
        self.portfolio_value = self.cash + self.shares * price_next

        # ✅ FIX 1: Reward is the immediate change in portfolio value (per-step P&L)
        reward = self.portfolio_value - last_portfolio_value
        return self._get_obs(), float(reward), done

    def _get_obs(self) -> torch.Tensor:
        # return position + lookback of close and volume (flattened)
        start = max(0, self.current_step - self.config.lookback_window)
        window = self.df.iloc[start:self.current_step]
        # If window smaller than lookback, pad with zeros
        closes = window['close'].values
        volumes = window['volume'].values
        if len(closes) < self.config.lookback_window:
            pad_len = self.config.lookback_window - len(closes)
            closes = np.concatenate([np.zeros(pad_len), closes])
            volumes = np.concatenate([np.zeros(pad_len), volumes])
        obs = np.concatenate(([self.position], closes, volumes)).astype(np.float32)
        return torch.tensor(obs)

# -----------------------------
# MuZero Network
# -----------------------------
class MuZeroNet(nn.Module):
    def __init__(self, obs_dim: int, action_space_size: int, hidden_size: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.action_space_size = action_space_size

        # representation: obs -> hidden
        self.representation = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # dynamics: (hidden, action) -> next_hidden + reward
        self.dyn_fc = nn.Linear(hidden_size + action_space_size, hidden_size)
        self.dyn_gru = nn.GRUCell(hidden_size, hidden_size)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

        # prediction heads: hidden -> value + policy
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_space_size)
        )

    def initial_inference(self, obs: torch.Tensor):
        # obs: [obs_dim] or [batch, obs_dim]
        hidden = self.representation(obs)
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return hidden, policy_logits, value

    def recurrent_inference(self, hidden: torch.Tensor, action_onehot: torch.Tensor):
        # hidden: [hidden_size], action_onehot: [action_space_size] or batch
        x = torch.cat([hidden, action_onehot], dim=-1)
        x = F.relu(self.dyn_fc(x))
        next_hidden = self.dyn_gru(x, hidden)
        reward = self.reward_head(next_hidden)
        policy_logits = self.policy_head(next_hidden)
        value = self.value_head(next_hidden)
        return next_hidden, reward, policy_logits, value

# -----------------------------
# Monte Carlo Tree Search (MCTS)
# -----------------------------
class Node:
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'Node'] = {}
        self.prior = float(prior)
        self.reward = 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

def ucb_score(parent: Node, child: Node, config: MuZeroConfig):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    return child.value() + pb_c * child.prior

def select_child(node: Node, config: MuZeroConfig) -> Tuple[int, Node]:
    action, child = max(node.children.items(), key=lambda it: ucb_score(node, it[1], config))
    return action, child

def add_dirichlet_noise(root: Node, alpha: float, fraction: float):
    priors = np.array([child.prior for child in root.children.values()], dtype=np.float32)
    if len(priors) == 0:
        return
    noise = np.random.dirichlet([alpha] * len(priors))
    for (a, child), n in zip(root.children.items(), noise):
        child.prior = child.prior * (1 - fraction) + n * fraction

def run_mcts(obs: torch.Tensor, net: MuZeroNet, config: MuZeroConfig):
    # Build root from initial inference
    root = Node()
    hidden, policy_logits, value = net.initial_inference(obs.unsqueeze(0))
    policy = F.softmax(policy_logits, dim=-1).squeeze(0).detach().cpu().numpy()
    for a, p in enumerate(policy):
        root.children[a] = Node(prior=p)

    add_dirichlet_noise(root, config.root_dirichlet_alpha, config.root_exploration_fraction)

    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        actions_taken = []
        # descend
        while node.expanded():
            action, node = select_child(node, config)
            search_path.append(node)
            actions_taken.append(action)
        
        # leaf; run recurrent inference for the last action in path (or initial if none)
        simulated_hidden = hidden.squeeze(0)
        # apply recurrent inferences along path to get leaf hidden
        for a in actions_taken:
            action_onehot = F.one_hot(torch.tensor([a]), config.action_space_size).float().to(hidden.device)
            simulated_hidden, reward, policy_logits, value = net.recurrent_inference(simulated_hidden.unsqueeze(0), action_onehot)
            simulated_hidden = simulated_hidden.squeeze(0)
            
        # Expand leaf with policy from last value (or initial if no actions)
        policy = F.softmax(policy_logits, dim=-1).detach().cpu().numpy().reshape(-1)
        for a, p in enumerate(policy):
            if a not in node.children:
                node.children[a] = Node(prior=float(p))
                
        # backup: use value.item()
        leaf_value = float(value.detach().cpu().item())
        for n in reversed(search_path):
            n.value_sum += leaf_value
            n.visit_count += 1
    return root

# -----------------------------
# Replay Buffer and Game History
# -----------------------------
class Game:
    """
    Stores sequences of observations, actions, rewards.
    Observations are torch tensors (obs vectors).
    """
    def __init__(self):
        self.observations: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []

    def store(self, obs: torch.Tensor, action: int, reward: float):
        # store a copy to avoid referencing same tensor
        self.observations.append(obs.detach().cpu().clone())
        self.actions.append(int(action))
        self.rewards.append(float(reward))

    def __len__(self):
        return len(self.actions)

class ReplayBuffer:
    def __init__(self, config: MuZeroConfig):
        self.buffer = deque(maxlen=config.replay_buffer_size)
        self.config = config

    def __len__(self):
        return len(self.buffer)

    def add_game(self, game: Game):
        if len(game) > 0:
            self.buffer.append(game)

    def sample_game(self) -> Game:
        return random.choice(list(self.buffer))

    def sample_batch(self, batch_size: int):
        games = [self.sample_game() for _ in range(batch_size)]
        return games

# -----------------------------
# DataLoader Implementation
# -----------------------------
class ReplayBufferIterableDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer, steps_per_epoch: int, batch_size: int):
        self.replay_buffer = replay_buffer
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

    def __iter__(self):
        # Each yield is a batch of games
        for _ in range(self.steps_per_epoch):
            if len(self.replay_buffer) == 0:
                continue
            yield self.replay_buffer.sample_batch(self.batch_size)

def muzero_collate(batch):
    # batch is a list of Game objects (size = batch_size)
    # We'll return lists and handle padding in training_step
    return batch

class EvaluationDataset(Dataset):
    def __init__(self, data_dict):
        self.symbols = list(data_dict.keys())
        self.data = data_dict
    def __len__(self):
        return len(self.symbols)
    def __getitem__(self, idx):
        symbol = self.symbols[idx]
        return symbol, self.data[symbol]

def eval_collate_fn(batch):
    return batch[0]

# -----------------------------
# Main Lightning Module
# -----------------------------
class MuZeroLightning(pl.LightningModule):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.replay_buffer = ReplayBuffer(config)

        obs_dim = 1 + (config.lookback_window) * 2  # position + closes + volumes
        self.net = MuZeroNet(obs_dim, self.config.action_space_size)

        self.train_data, self.val_data, self.test_data = {}, {}, {}
        self.aux_data = {}  # reserved

    def configure_optimizers(self):
        return optim.AdamW(self.net.parameters(), lr=self.config.lr_init, weight_decay=self.config.weight_decay)

    def setup(self, stage=None):
        if self.train_data: return  # avoid reloading
        print(f"Setting up data for stage: {stage}...")
        cfg = self.config
        data_src = load_data_from_db_or_synthetic(cfg)

        print("Loading data...")
        for sym, splits in tqdm(data_src, desc="Loading symbol data"):
            if not splits['train'].empty:
                self.train_data[sym] = splits['train']
            if not splits['val'].empty:
                self.val_data[sym] = splits['val']
            if not splits['test'].empty:
                self.test_data[sym] = splits['test']
        print(f"Data loaded. Train symbols: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")

    def on_train_start(self):
        if len(self.replay_buffer) < self.config.min_initial_games:
            print("Populating replay buffer with initial self-play games...")
            self.populate_replay_buffer(self.config.min_initial_games)
            print(f"Replay buffer size: {len(self.replay_buffer)}")

    def train_dataloader(self):
        dataset = ReplayBufferIterableDataset(self.replay_buffer, self.config.steps_per_epoch, self.config.batch_size)
        return DataLoader(dataset, batch_size=None, collate_fn=muzero_collate, num_workers=0)

    def val_dataloader(self):
        if not self.val_data: return None
        dataset = EvaluationDataset(self.val_data)
        return DataLoader(dataset, batch_size=1, collate_fn=eval_collate_fn, num_workers=0)

    def test_dataloader(self):
        if not self.test_data: return None
        dataset = EvaluationDataset(self.test_data)
        return DataLoader(dataset, batch_size=1, collate_fn=eval_collate_fn, num_workers=0)

    # Utilities for n-step bootstrapped returns
    def compute_n_step_return(self, rewards: List[float], start_index: int, n: int, discount: float, bootstrap_value: float = 0.0):
        ret = 0.0
        discount_pow = 1.0
        for i in range(n):
            idx = start_index + i
            if idx >= len(rewards):
                break
            ret += rewards[idx] * discount_pow
            discount_pow *= discount
        ret += bootstrap_value * discount_pow
        return ret

    def training_step(self, batch, batch_idx):
        # ✅ FIX 2: This training step has been completely rewritten to correctly
        #           compute and aggregate losses for the entire batch.
        games: List[Game] = batch
        all_game_losses = []
        
        # Scalar losses for logging
        total_value_loss, total_reward_loss, total_policy_loss = 0.0, 0.0, 0.0
        total_count = 0

        for game in games:
            if len(game) < 2:
                continue
                
            # Pick a random start index leaving room for unroll
            max_start = max(1, len(game) - self.config.num_unroll_steps)
            start_idx = random.randint(0, max_start - 1)

            # --- Initial Inference ---
            obs = game.observations[start_idx].to(self.device)
            root = run_mcts(obs, self.net, self.config)

            # Policy Target: normalized visit counts from MCTS
            visit_counts = np.array([root.children.get(a, Node()).visit_count for a in range(self.config.action_space_size)], dtype=np.float32)
            if visit_counts.sum() == 0:
                policy_target = torch.ones_like(torch.from_numpy(visit_counts)) / len(visit_counts)
            else:
                policy_target = torch.from_numpy(visit_counts / visit_counts.sum()).to(self.device)

            # Value Target: n-step bootstrapped return
            bootstrap_idx = start_idx + self.config.td_steps
            bootstrap_value = 0.0
            if bootstrap_idx < len(game.observations):
                with torch.no_grad():
                    obs_boot = game.observations[bootstrap_idx].to(self.device)
                    _, _, v_boot = self.net.initial_inference(obs_boot.unsqueeze(0))
                    bootstrap_value = v_boot.item()
            
            value_target = self.compute_n_step_return(game.rewards, start_idx, self.config.td_steps, self.config.discount, bootstrap_value)
            value_target = torch.tensor(value_target, dtype=torch.float32, device=self.device)

            # Get predictions from the network
            hidden_state, policy_logits, value_pred = self.net.initial_inference(obs.unsqueeze(0))
            
            # Calculate initial losses
            policy_loss = F.cross_entropy(policy_logits, policy_target.unsqueeze(0))
            value_loss = F.mse_loss(value_pred.squeeze(), value_target)
            
            game_loss_components = [policy_loss, value_loss]
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_count += 1

            # --- Unroll Steps ---
            current_hidden = hidden_state
            for k in range(self.config.num_unroll_steps):
                step_idx = start_idx + k
                if step_idx >= len(game.actions):
                    break
                
                action = game.actions[step_idx]
                action_onehot = F.one_hot(torch.tensor([action]), self.config.action_space_size).float().to(self.device)
                
                # Get predictions
                next_hidden, reward_pred, policy_logits_k, value_pred_k = self.net.recurrent_inference(current_hidden, action_onehot)
                
                # Targets for unroll step
                reward_target = torch.tensor(game.rewards[step_idx], dtype=torch.float32, device=self.device)
                
                bootstrap_idx_k = step_idx + 1 + self.config.td_steps
                bootstrap_value_k = 0.0
                if bootstrap_idx_k < len(game.observations):
                     with torch.no_grad():
                        obs_boot_k = game.observations[bootstrap_idx_k].to(self.device)
                        _, _, v_boot_k = self.net.initial_inference(obs_boot_k.unsqueeze(0))
                        bootstrap_value_k = v_boot_k.item()
                
                value_target_k = self.compute_n_step_return(game.rewards, step_idx + 1, self.config.td_steps, self.config.discount, bootstrap_value_k)
                value_target_k = torch.tensor(value_target_k, dtype=torch.float32, device=self.device)
                
                # Losses for unroll step
                reward_loss_k = F.mse_loss(reward_pred.squeeze(), reward_target)
                value_loss_k = F.mse_loss(value_pred_k.squeeze(), value_target_k)
                
                game_loss_components.extend([reward_loss_k, value_loss_k])
                total_reward_loss += reward_loss_k.item()
                total_value_loss += value_loss_k.item()
                total_count += 1
                
                current_hidden = next_hidden

            # Sum up all loss components for this game
            total_game_loss = sum(game_loss_components)
            all_game_losses.append(total_game_loss)

        if not all_game_losses:
            return None
        
        # Final loss is the mean over the batch
        total_loss = torch.stack(all_game_losses).mean()

        # Log average scalar losses
        if total_count > 0:
            self.log("train/value_loss", total_value_loss / total_count, prog_bar=False, on_step=True)
            self.log("train/policy_loss", total_policy_loss / total_count, prog_bar=False, on_step=True)
            self.log("train/reward_loss", total_reward_loss / total_count, prog_bar=False, on_step=True)
            self.log("train/total_loss", total_loss.item(), prog_bar=True, on_step=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        symbol, df = batch
        env = StockEnv(df, self.config)
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            root = run_mcts(obs.to(self.device), self.net, self.config)
            action = max(root.children.items(), key=lambda it: it[1].visit_count)[0]
            obs, reward, done = env.step(action)
            total_reward += reward
            
        final_return = env.portfolio_value / env.initial_cash
        self.log(f"val_return/{symbol}", final_return, batch_size=1)
        self.log("val/total_reward_sum", total_reward, batch_size=1, reduce_fx="sum")
        return total_reward

    def test_step(self, batch, batch_idx):
        symbol, symbol_df = batch
        env = StockEnv(symbol_df, self.config)
        obs, done = env.reset(), False
        while not done:
            root = run_mcts(obs.to(self.device), self.net, self.config)
            action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
            obs, reward, done = env.step(action)
            
        normalized_return = env.portfolio_value / env.initial_cash
        self.log(f'test_return/{symbol}', normalized_return, batch_size=1)
        return normalized_return

    def populate_replay_buffer(self, num_games: int):
        # Self-play on training symbols using current policy (MCTS)
        rng_syms = list(self.train_data.keys())
        if not rng_syms:
            for s in (self.config.symbols_to_keep or ('AAPL','GOOG','MSFT')):
                self.train_data[s] = generate_synthetic_symbol(s, length=600)
            rng_syms = list(self.train_data.keys())
            
        for i in tqdm(range(num_games), desc="Self-Play"):
            sym = random.choice(rng_syms)
            df = self.train_data[sym]
            game = self.run_episode(df)
            self.replay_buffer.add_game(game)

    def run_episode(self, df: pd.DataFrame) -> Game:
        env = StockEnv(df, self.config)
        obs, done = env.reset(), False
        game = Game()
        # Run until end or a maximum horizon to limit game length
        steps = 0
        max_steps = min(250, len(df) - self.config.lookback_window - 1)
        
        while not done and steps < max_steps:
            root = run_mcts(obs.to(self.device), self.net, self.config)
            if root.children:
                action = max(root.children.items(), key=lambda it: it[1].visit_count)[0]
            else:
                action = random.randint(0, self.config.action_space_size - 1)
                
            next_obs, reward, done = env.step(action)
            game.store(obs, action, reward)
            obs = next_obs
            steps += 1
        return game

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == '__main__':
    set_seed(123)
    cfg = MuZeroConfig(
        num_epochs=3,
        steps_per_epoch=100,
        min_initial_games=40,
        batch_size=32,
        symbols_to_keep=('AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN'),
        use_synthetic_data=True
    )

    model = MuZeroLightning(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        val_check_interval=1.0,
        log_every_n_steps=10,
        enable_checkpointing=False,
        # ✅ FIX 3: Add gradient clipping to prevent exploding gradients
        gradient_clip_val=1.0
    )

    print("Starting training (toy MuZero-style)...")
    trainer.fit(model)

    print("\nStarting testing...")
    trainer.test(model)