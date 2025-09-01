#!/usr/bin/env python3
"""
Quick model combination test - just tests instantiation and forward pass.
This is much faster than full training and can verify basic compatibility.
"""

import os
import sys
import json
import torch
import traceback
from typing import Dict, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omniseg.config import get_available_backbones, get_available_heads
from omniseg.models.backbones import get_backbone
from omniseg.models.heads import get_head


class QuickModelTester:
    """Quick test runner for model instantiation and forward pass."""
    
    def __init__(self):
        self.results = {}
        
    def test_backbone_instantiation(self, backbone_type: str) -> Dict:
        """Test if a backbone can be instantiated."""
        result = {
            'backbone': backbone_type,
            'instantiation': False,
            'forward_pass': False,
            'error': None
        }
        
        try:
            backbone = get_backbone(backbone_type)
            result['instantiation'] = True
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                features = backbone(dummy_input)
                if isinstance(features, dict) and len(features) > 0:
                    result['forward_pass'] = True
                    
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Backbone {backbone_type}: {e}")
            
        return result
    
    def test_head_instantiation(self, head_type: str, backbone_type: str = 'simple') -> Dict:
        """Test if a head can be instantiated with given backbone."""
        result = {
            'head': head_type,
            'backbone': backbone_type,
            'instantiation': False,
            'forward_pass': False,
            'error': None
        }
        
        try:
            head = get_head(head_type, num_classes=3, backbone_type=backbone_type, image_size=64)
            result['instantiation'] = True
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                outputs = head(dummy_input)
                if outputs is not None:
                    result['forward_pass'] = True
                    
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Head {head_type} with {backbone_type}: {e}")
            
        return result
    
    def test_full_model_combination(self, backbone_type: str, head_type: str) -> Dict:
        """Test a full backbone-head combination."""
        result = {
            'backbone': backbone_type,
            'head': head_type,
            'status': 'failed',
            'instantiation': False,
            'forward_pass': False,
            'error': None
        }
        
        try:
            # Test backbone first
            backbone_result = self.test_backbone_instantiation(backbone_type)
            if not backbone_result['forward_pass']:
                result['error'] = f"Backbone failed: {backbone_result['error']}"
                return result
            
            # Test head with backbone
            head_result = self.test_head_instantiation(head_type, backbone_type)
            if not head_result['forward_pass']:
                result['error'] = f"Head failed: {head_result['error']}"
                return result
                
            result['instantiation'] = True
            result['forward_pass'] = True
            result['status'] = 'working'
            print(f"✅ {backbone_type} + {head_type}: OK")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ {backbone_type} + {head_type}: {e}")
            
        return result
    
    def run_all_tests(self):
        """Run quick tests for all combinations."""
        backbones = get_available_backbones()
        heads = get_available_heads()
        
        print(f"🚀 Running quick model compatibility tests...")
        print(f"Backbones: {backbones}")
        print(f"Heads: {heads}")
        
        # Test each combination
        for backbone in backbones:
            for head in heads:
                print(f"\nTesting {backbone} + {head}...")
                result = self.test_full_model_combination(backbone, head)
                self.results[f"{backbone}_{head}"] = result
        
        self._print_summary()
        self._save_results()
    
    def _save_results(self):
        """Save test results to JSON file."""
        with open('quick_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to quick_test_results.json")
    
    def _print_summary(self):
        """Print test summary."""
        total = len(self.results)
        working = sum(1 for r in self.results.values() if r['status'] == 'working')
        failed = total - working
        
        print(f"\n{'='*60}")
        print(f"QUICK TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total combinations tested: {total}")
        print(f"Working combinations: {working} ({working/total*100:.1f}%)")
        print(f"Failed combinations: {failed} ({failed/total*100:.1f}%)")
        
        working_combinations = []
        failed_combinations = []
        
        for key, result in self.results.items():
            combo = f"{result['backbone']}+{result['head']}"
            if result['status'] == 'working':
                working_combinations.append(combo)
            else:
                failed_combinations.append(combo)
        
        if working_combinations:
            print(f"\n✅ WORKING COMBINATIONS:")
            for combo in working_combinations:
                print(f"  - {combo}")
        
        if failed_combinations:
            print(f"\n❌ FAILED COMBINATIONS (Work in Progress):")
            for combo in failed_combinations:
                backbone, head = combo.split('+')
                result = self.results[f"{backbone}_{head}"]
                error_msg = result['error'][:100] if result['error'] else 'Unknown error'
                print(f"  - {combo}: {error_msg}...")


def main():
    """Main test runner."""
    tester = QuickModelTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()