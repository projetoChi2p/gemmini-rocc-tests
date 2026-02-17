#!/bin/bash
# Example script to test multiple model/dataset combinations

echo "======================================"
echo "ViT Model Testing Script"
echo "======================================"

# Define configurations to test
CONFIGS=(
    "minivit sat6"
    "minivit mnist"
    "deitvit mnist"
)

# Run each configuration
for config in "${CONFIGS[@]}"; do
    read -r model dataset <<< "$config"
    
    echo ""
    echo "--------------------------------------"
    echo "Testing: $model on $dataset"
    echo "--------------------------------------"
    
    make -f RunViT.mk run MODEL=$model DATASET=$dataset
    
    if [ $? -eq 0 ]; then
        echo "✓ Success: $model on $dataset"
    else
        echo "✗ Failed: $model on $dataset"
    fi
done

echo ""
echo "======================================"
echo "All tests complete!"
echo "Results saved in outputs/ directory"
echo "======================================"
ls -lh outputs/
