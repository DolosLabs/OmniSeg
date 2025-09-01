# OmniSeg Testing Guide

## Quick Start

```bash
# Run tests with the convenient script
./run_tests.sh

# Or run directly with Python
python -m tests.quick_model_test

# Run tests with HuggingFace authentication
HF_TOKEN=your_token ./run_tests.sh
```

## Setting up HuggingFace Authentication

### For Local Development

1. Get a token from [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Copy `.env.example` to `.env` and add your token
3. Or export directly: `export HF_TOKEN=your_token_here`

### For GitHub Actions

1. Go to repository Settings → Secrets and variables → Actions
2. Add new repository secret:
   - Name: `HF_TOKEN`
   - Value: `your_huggingface_token_here`

## Test Structure

- `tests/quick_model_test.py` - Main test file
- `tests/quick_test_results.json` - Generated results (created after running tests)
- `run_tests.sh` - Convenient runner script
- `.github/workflows/test-models.yml` - Automated CI testing

## Test Results

The tests check all backbone + head combinations:

- **simple**: Works without external downloads (for testing purposes)
- **dino**, **sam**, **swin**, **convnext**, **repvgg**, **resnet**: Require HuggingFace Hub access

Results are saved as JSON with detailed error information for debugging.