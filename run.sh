#!/bin/bash

MODEL_LIST=("42dot/42dot_LLM-SFT-1.3B")
ITERATION_LIST=("1" "2" "3")
BATCH_SIZE=4

for MODEL in "${MODEL_LIST[@]}"; do
    echo "Running evaluation for $MODEL."

    for IT in "${ITERATION_LIST[@]}"; do
        echo "Iteration $IT"

        python generate_cot.py --model_path "$MODEL" --dataset "HAERAE-HUB/QARV-binary-set" --options A B C --last_option "None of the Above." --language "english" --iteration "$IT" --bs $BATCH_SIZE --device "cuda"
        python generate_result_cot.py --model_path "$MODEL" --options A B C --language "english" --iteration "$IT" --bs $BATCH_SIZE --device "cuda"
    done

    python generate_mc.py --model_path "$MODEL" --dataset "HAERAE-HUB/QARV-binary-set" --options A B C --last_option "None of the Above." --language "english" --bs $BATCH_SIZE --device "cuda"
done
