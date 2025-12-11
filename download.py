import os 
import torch 
import argparse

try:
    from diffusers import ZImagePipeline
except ImportError:
    print(f"\033[91mdiffusers has not installed or diffusers.ZImagePipeline is not available, please install it first\033[0m")
    print(f"\033[93mpip install -U git+https://github.com/huggingface/diffusers\033[0m")
    import sys
    sys.exit(1)

from huggingface_hub import snapshot_download


def download_model(repo_id: str, local_dir: str, max_retries: int = 3) -> str:
    """下载模型到本地目录，支持断点续传和重试"""
    for attempt in range(max_retries):
        try:
            return snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False, 
                resume_download=True,  
            )
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\033[91m下载失败 (尝试 {attempt + 1}/{max_retries}): {e}\033[0m")
                print("\033[93m正在重试...\033[0m")
            else:
                raise Exception(f"\033[91m下载失败，已重试 {max_retries} 次\033[0m") from e


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="下载 Hugging Face 模型")
    parser.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo", 
                        help="模型仓库ID")
    parser.add_argument("--local_dir", type=str, default=None,
                        help="本地保存目录")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="最大重试次数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    repo_id = args.model
    local_dir = args.local_dir or os.path.join(
        os.getenv("HF_HOME", "./hf_models"), 
        repo_id.replace("/", "--")
    )
    
    print(f"\033[94m开始下载模型: {repo_id}\033[0m")
    print(f"\033[94m保存路径: {local_dir}\033[0m")
    
    model_path = download_model(repo_id, local_dir, max_retries=args.max_retries)
    print(f"\033[92m模型已下载到: {model_path}\033[0m")