## Table of Contents
- [Creating Environment](#adapt_think)
- [Training](#training)
- [Evaluation](#evaluation)

## Creating Environment

- You can refer to verl documentation `https://verl.readthedocs.io/en/v0.4.1/`

1. For vLLM with Megatron or FSDP, please use the stable version of image whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3.

2. Launch the desired Docker image and attach into it:
```bash
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag>
docker start verl
docker exec -it verl bash
```

3. Inside the container, install latest verl:
```bash
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
# pick your choice of inference engine: vllm
# pip3 install -e .[vllm]
# or install from pypi instead of git via:
# pip3 install verl[vllm]
```

4. Install other packages:
```bash
pip install -r requirements.txt
```

## Training

1. You need to confirm the parameter values in the training script, and then run it:
```bash
bash run src/dapo/run_grpo_deepseek_qwen_1.5B.sh
```

2. After the training is completed, you can run the following script to obtain the full model:
```bash
# confirm the parameter
bash run scripts/merge_model.sh
```

3. The trained model is available on HuggingFace: [🤗 HF Repo](https://huggingface.co/YTChan/Rethinker-1.5B)

## Evaluation

1. First, deploy the model using vllm:
```bash
# confirm the parameter
bash tests/run_vllm.sh
```

2. Run the test script:
```bash
# confirm the parameter
python tests/math_eval.py
```