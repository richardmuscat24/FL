#!/bin/bash
# Clean environment before running FL training

cd "$(dirname "$0")"

echo "🧹 Cleaning environment..."
rm -rf __pycache__
rm -f final_model.json personalized_model_*.json
echo "✅ Clean! Ready to run training."
echo ""
echo "Now run: flwr run ."
