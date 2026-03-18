#!/usr/bin/env bash

export GLOO_SOCKET_IFNAME="lo"
export CUDA_VISIBLE_DEVICES="4"

# model="path/to/Rethinker"
model="/dataset/chenyuting/verl/DQwen-no_suf-thinkless-dapo-8k/global_step_480/hf"
served_model_name="Rethinker"

# api
port=8002

vllm serve ${model} --served-model-name ${served_model_name} \
                        --max-num-seqs 128  \
                        --tensor-parallel-size 1 \
                        --enforce-eager \
                        --gpu_memory_utilization 0.95 \
                        --enable-auto-tool-choice \
                        --tool-call-parser hermes \
                        --port ${port} \
                        --host 0.0.0.0 \
                        --dtype bfloat16 \
                        --max-model-len 32768 \