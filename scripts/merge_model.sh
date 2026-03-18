export CUDA_VISIBLE_DEVICES="6"

python model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/save/model/global_step_xxx/actor \
    --target_dir /path/to/save/model/global_step_xx/hf
