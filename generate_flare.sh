#!/bin/bash
# generate_flare.sh

model_name=${1}

if [ -z "$model_name" ]; then
    echo "Usage: ./generate_flare.sh <model_name>"
    echo "Example: ./generate_flare.sh beat_block_hammer"
    exit 1
fi

echo "Generating FLARE configuration for task: $model_name"

python ./model_config/_generate_flare_config.py $model_name

echo "FLARE configuration generated successfully!"
echo "You can now run: ./finetune_flare.sh ${model_name}_flare"