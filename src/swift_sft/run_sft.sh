export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NPROC_PER_NODE=4
export MASTER_PORT=29501

swift sft \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --train_type full \
  --dataset data/sft_7k.jsonl \
  --num_train_epochs 3 \
  --split_dataset_ratio 0.000 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 16 \
  --save_steps 50 \
  --logging_steps 5 \
  --max_length 4096 \
  --deepspeed zero2 \
  --warmup_ratio 0.05 \
  --attn_impl flash_attn \
  --response_prefix '' \
  --output_dir path/to/save/sft/model