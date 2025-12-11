export HF_ENDPOINT=https://hf-mirror.com


export HF_HOME=huggingface

export HF_TOKEN='hf_token'

mkdir -p $HF_HOME

pip install huggingface_hub -U
pip install -U git+https://github.com/huggingface/diffusers


hf auth login --token $HF_TOKEN

# Tongyi-MAI/Z-Image-Turbo 
python scripts/download.py --model Tongyi-MAI/Z-Image-Turbo 

# stabilityai/stable-diffusion-3-medium-diffusers
python scripts/download.py --model stabilityai/stable-diffusion-3-medium-diffusers 

# Qwen/Qwen3-VL-8B-Instruct 
python scripts/download.py --model Qwen/Qwen3-VL-8B-Instruct 

# Qwen/Qwen2.5-VL-7B-Instruct
python scripts/download.py --model Qwen/Qwen2.5-VL-7B-Instruct 



