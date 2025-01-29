"""
Usage: Download models from Hugging Face

From command line:
python hf_downloader.py mlx-community/OpenELM-3B --save-dir ./my_model


From Python:

from hf_downloader import ModelDownloader

downloader = ModelDownloader()
downloader.download_model("mlx-community/OpenELM-3B", "./my_model")


Returns:
        _type_: _description_
"""

import os
import argparse
from pathlib import Path
import transformers
from transformers import logging
from huggingface_hub import snapshot_download
from tqdm import tqdm

class ModelDownloader:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'huggingface'
        logging.set_verbosity_error()

    def download_model(self, model_name: str, save_dir: str, download_tokenizer=True):
        """
        Download a model from Hugging Face.
        
        Args:
            model_name (str): Model name or URL from Hugging Face
            save_dir (str): Directory to save the model
            download_tokenizer (bool): Whether to download tokenizer
        """
        try:
            # Create save directory
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            print(f"Downloading model: {model_name}")
            
            # Download model
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            model.save_pretrained(save_dir)

            # Download tokenizer
            if download_tokenizer:
                print("Downloading tokenizer...")
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
                tokenizer.save_pretrained(save_dir)

            print(f"✅ Successfully downloaded to: {save_dir}")
            return True

        except Exception as e:
            print(f"❌ Error downloading model: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument("model_name", help="Model name or URL from Hugging Face")
    parser.add_argument("--save-dir", default="./downloaded_model",
                      help="Directory to save the model")
    parser.add_argument("--no-tokenizer", action="store_true",
                      help="Skip tokenizer download")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    
    downloader.download_model(
        args.model_name,
        args.save_dir,
        not args.no_tokenizer
    )

def test():
    # test
    model_name = "mlx-community/SmolVLM-500M-Instruct-bf16"
    save_dir = "./base_model/vlm/"
    
    downloader = ModelDownloader()
    
    downloader.download_model(
        model_name,
        save_dir
    )
    
if __name__ == "__main__":
    #main()
    test()