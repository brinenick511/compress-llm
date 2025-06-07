model_path=${HOME}/models/meta-llama/Llama-3.2-3B-Instruct/

python SVDLLM.py \
    --model ${model_path} \
    --step 1 \
    --ratio 0.2 \
    --whitening_nsamples 32 \
    --dataset wikitext2 \
    --seed 3 \
    --model_seq_len 2048 \
    --save_path .
