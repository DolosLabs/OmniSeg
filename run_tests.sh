#!/bin/bash
#
# Run OmniSeg quick model tests
#
# Usage: ./run_tests.sh

echo "🚀 Running OmniSeg Quick Model Tests..."
echo "   Tests located in: tests/quick_model_test.py"
echo "   Results will be saved to: tests/quick_test_results.json"
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "⚠️  Warning: No HuggingFace token found in environment"
    echo "   Set HF_TOKEN or HUGGINGFACE_TOKEN to enable model downloads"
    echo "   Only models that don't require authentication will work"
    echo ""
fi

# Run the tests
python -m tests.quick_model_test