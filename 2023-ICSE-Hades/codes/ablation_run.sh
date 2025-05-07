#!/bin/bash

for model in "hades" "hades-wom" "hades-wol" "hades-wof" "hades-woa" "hades-woc" "hades-anno"; do
    for run in {1..3}; do
        python run.py \
            --data ../data/chunk_10 \
            --main_model "$model"
        sleep 1
    done
done