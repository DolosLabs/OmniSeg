# OmniSeg

A versatile, semi-supervised instance segmentation framework built on PyTorch Lightning. OmniSeg enables training with various state-of-the-art backbones and segmentation heads, including a novel contour-based approach, on the COCO dataset. It's designed for flexibility, allowing researchers and developers to easily experiment with different model architectures and training strategies.

## ✨ Features

  * **Semi-Supervised Learning (SSL)**: Leverages a teacher-student model architecture to use both labeled and unlabeled data, a crucial technique for training with limited annotations.
  * **Flexible Backbones**: Supports a wide range of popular vision models as backbones, including:
      * **ViT**: DINO (Work in Progress), SAM (Work in Progress)
      * **CNN**: ResNet (Work in Progress), ConvNeXt (Work in Progress), RepVGG (Work in Progress)
      * **Hybrid**: Swin Transformer (Work in Progress)
      * **Test**: Simple CNN backbone for testing and development (Working)
  * **Modular Segmentation Heads**: Experiment with different instance segmentation approaches:
      * **Mask R-CNN**: The classic, widely-used two-stage detector (Work in Progress)
      * **ContourFormer**: A custom head that predicts object contours directly (Work in Progress)
      * **Deformable DETR**: Advanced DETR-based segmentation head (Working)
  * **PyTorch Lightning**: Training and validation are managed by PyTorch Lightning, which simplifies boilerplate code, ensures reproducibility, and supports distributed training.
  * **Automated Data Handling**: The project automatically downloads and prepares the COCO 2017 dataset, or use the tiny synthetic dataset for quick testing.

## 🚀 Getting Started

Follow these steps to set up and run OmniSeg.

### Prerequisites

  * Python 3.8+
  * NVIDIA GPU with CUDA (recommended)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/DolosLabs/omniseg.git
    cd omniseg
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Note: You will need to create a `requirements.txt` file from the imports in your code, or manually install them: `pip install transformers torch torchvision tqdm pycocotools pytorch-lightning timm scikit-image scipy`)
3.  **Prepare the tiny dataset (recommended for testing)**:
    ```bash
    python generate_tiny_data.py
    ```
    This creates a small synthetic dataset with geometric shapes for quick training and testing.

4.  **Or prepare the COCO 2017 dataset**:
    The script automatically handles the download and extraction of the COCO 2017 dataset when you first run it. No manual setup is required.

## 💻 Usage

To train a model, use the `train.py` script with command-line arguments to configure your desired backbone and segmentation head.

### Basic Training

Train a model with a specified backbone and head. The script will automatically download the dataset if it doesn't exist.

```bash
# Example 1: Quick test with working combination
python train.py --backbone simple --head deformable_detr --use_tiny_data --fast_dev_run

# Example 2: Train with COCO dataset (requires network access)
python train.py --backbone resnet --head maskrcnn

# Example 3: Train with tiny dataset for development
python train.py --backbone simple --head deformable_detr --use_tiny_data --max_steps 100
```

-----

### Advanced Configuration

You can customize training with various parameters.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--backbone` | `str` | `dino` | Backbone model (`dino`, `sam`, `swin`, `convnext`, `repvgg`, `resnet`). |
| `--head` | `str` | `mask2former` | Segmentation head (`mask2former`, `maskrcnn`, `contourformer`). |
| `--learning_rate` | `float` | `5e-5` | Learning rate. |
| `--image_size` | `int` | `None` | Custom image size for resizing. |
| `--batch_size` | `int` | `None` | Batch size per GPU. |
| `--num_labeled_images` | `int` | `-1` | Number of labeled images to use for training (`-1` for all). |
| `--num_unlabeled_images` | `int` | `-1` | Number of unlabeled images to use for training (`-1` for all). |
| `--fast_dev_run` | `flag` | `False` | Run a single batch for testing the code. |
| `--warmup_steps` | `int` | `500` | Steps to train on labeled data only before adding unsupervised loss. |
| `--unsup_rampup_steps`| `int` | `5000`| Steps to ramp up the unsupervised loss weight. |
| `--use_tiny_data` | `flag` | `False` | Use the tiny synthetic dataset for quick training and testing. |

**Example with advanced settings**:

```bash
python train.py --backbone simple --head deformable_detr --use_tiny_data --fast_dev_run
```

### Quick Testing

For rapid development and testing, use the tiny synthetic dataset:

```bash
# Generate tiny dataset (100 train, 20 val, 20 test images)
python generate_tiny_data.py

# Quick training test with tiny data
python train.py --backbone simple --head deformable_detr --use_tiny_data --fast_dev_run
```

-----

## 🧪 Testing Status

This table shows the current status of all backbone-head combinations:

| Backbone | Head | Status | Notes |
|----------|------|--------|-------|
| simple | maskrcnn | 🔄 Work in Progress | Head expects training targets |
| simple | contourformer | 🔄 Work in Progress | Shape mismatch in feature dimensions |
| simple | deformable_detr | ✅ Working | Ready for training and evaluation |
| dino | maskrcnn | 🔄 Work in Progress | Requires external model download |
| dino | contourformer | 🔄 Work in Progress | Requires external model download |
| dino | deformable_detr | 🔄 Work in Progress | Requires external model download |
| sam | maskrcnn | 🔄 Work in Progress | Requires external model download |
| sam | contourformer | 🔄 Work in Progress | Requires external model download |
| sam | deformable_detr | 🔄 Work in Progress | Requires external model download |
| swin | maskrcnn | 🔄 Work in Progress | Requires external model download |
| swin | contourformer | 🔄 Work in Progress | Requires external model download |
| swin | deformable_detr | 🔄 Work in Progress | Requires external model download |
| convnext | maskrcnn | 🔄 Work in Progress | Requires external model download |
| convnext | contourformer | 🔄 Work in Progress | Requires external model download |
| convnext | deformable_detr | 🔄 Work in Progress | Requires external model download |
| repvgg | maskrcnn | 🔄 Work in Progress | Requires external model download |
| repvgg | contourformer | 🔄 Work in Progress | Requires external model download |
| repvgg | deformable_detr | 🔄 Work in Progress | Requires external model download |
| resnet | maskrcnn | 🔄 Work in Progress | Requires external model download |
| resnet | contourformer | 🔄 Work in Progress | Requires external model download |
| resnet | deformable_detr | 🔄 Work in Progress | Requires external model download |

### Test Details

- **Status**: Whether the combination can be instantiated and run without errors
- **Working**: Ready for production training and evaluation
- **Work in Progress**: Known issues that need to be resolved

### Running Tests

To run the comprehensive test suite:

```bash
# Generate tiny dataset first
python generate_tiny_data.py

# Quick model compatibility test (fast) - now located in tests/ folder
./run_tests.sh
# OR
python -m tests.quick_model_test

# Full training convergence test (slower)
python test_model_combinations.py --max_steps 10
```

**HuggingFace Authentication Setup:**

Most backbone models require downloading from HuggingFace Hub. To enable this:

1. Get a HuggingFace token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Set it as an environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   # OR
   export HUGGINGFACE_TOKEN=your_token_here
   ```
3. Run tests with authentication:
   ```bash
   HF_TOKEN=your_token ./run_tests.sh
   ```

For GitHub Actions, set the token as a repository secret named `HF_TOKEN`:

1. Go to your repository settings
2. Navigate to "Secrets and variables" → "Actions" 
3. Click "New repository secret"
4. Name: `HF_TOKEN`
5. Value: `your_huggingface_token_here`

The automated tests will run on every push and pull request.

Test results are saved to JSON files for detailed analysis:
- `tests/quick_test_results.json` - Model instantiation and forward pass results
- `test_results.json` - Full training test results (if run)

### Current Limitations

Most backbone models require downloading pretrained weights from HuggingFace/timm, which may fail in restricted network environments. The `simple` backbone works without external dependencies for testing purposes.

-----

## 🖼️ Visualization

To visualize the predictions of a trained model, use the `visualize_model.py` script. This script loads a saved checkpoint and displays predictions on random validation images.

```bash
python visualize_model.py path/to/your/checkpoint.ckpt --project_dir ./SSL_Instance_Segmentation
```

A file named `model_predictions_viz.png` will be saved in the same directory as your checkpoint.

-----

## 🔧 Project Structure

This project has been refactored into a modular structure for better maintainability:

- **`omniseg/`** - Main package with organized modules
  - **`config/`** - Configuration settings and model defaults
  - **`data/`** - Dataset utilities and data loading
  - **`models/`** - Backbone and head model implementations
  - **`training/`** - PyTorch Lightning training logic
- **`tests/`** - Test modules and test results
  - **`quick_model_test.py`** - Fast model compatibility testing
  - **`quick_test_results.json`** - Test results (generated after running tests)
- **`docs/`** - Additional documentation including refactoring details
- **`generate_tiny_data.py`** - Creates synthetic dataset for testing
- **`test_model_combinations.py`** - Comprehensive training convergence testing
- **`run_tests.sh`** - Convenient test runner script

For detailed information about the modular structure, see [`docs/REFACTORED_README.md`](docs/REFACTORED_README.md).

-----

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements, new backbones, or segmentation heads, please open an issue or submit a pull request.

-----

## 📝 Attributions

This project makes extensive use of and builds upon the work of others in the open-source community. We would like to acknowledge the following key dependencies and their creators:

  * **PyTorch**: An open-source machine learning framework.
  * **PyTorch Lightning**: A lightweight PyTorch wrapper for high-performance AI research.
  * **Transformers (Hugging Face)**: Provides pre-trained models for DINO, SAM, and Swin Transformer.
  * **timm (PyTorch Image Models)**: A collection of models for computer vision, including ConvNeXt, RepVGG, and ResNet.
  * **COCO Dataset**: The Common Objects in Context dataset, providing a benchmark for object detection, segmentation, and captioning.
  * **Mask2Former**: A universal segmentation framework.
  * **Mask R-CNN**: A widely adopted instance segmentation model.

The semi-supervised learning framework in this project is inspired by the **Mean Teacher** methodology, which is an effective approach for training with limited labeled data.

-----
**Contact**: Ben Harris - https://doloslabs.com
