model_path=${HOME}/models/meta-llama/Llama-3.2-3B-Instruct/
model_path=${HOME}/models/meta-llama/Llama-3.1-8B-Instruct/

nsamples=32

ratio=20

python SVDLLM.py \
    --model ${model_path} \
    --step 1 \
    --ratio "0."${ratio} \
    --whitening_nsamples ${nsamples} \
    --dataset wikitext2 \
    --seed 3 \
    --model_seq_len 2048 \
    --DEV cuda:0 \
    --save_path ${HOME}/compress-llm/cache/test_${ratio}/
    # --save_path ${HOME}/compress-llm/cache/${ratio}/

# python SVDLLM.py \
#     --model_path /data/yanghq/compress-llm/cache/30/_data_yanghq_models_meta_llama_Llama_3.2_3B_Instruct__whitening_only_0.7.pt \
#     --step 6 \
