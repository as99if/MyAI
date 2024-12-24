import os
import transformers

def download_llm(model_name, save_dir):
    """
    Download a language model from Hugging Face and save it to a specified directory.

    Args:
        model_name (str): The name of the language model to download (e.g. "bert-base-uncased").
        save_dir (str): The directory to save the downloaded model to.
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Download the model using the Transformers library
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    # Save the model to the specified directory
    model.save_pretrained(save_dir)

    # Save the tokenizer to the same directory
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)

    print(f"Model {model_name} downloaded and saved to {save_dir}")


def download_yolo():
    from huggingface_hub import hf_hub_download 
    hf_hub_download("merve/yolov9", filename="yolov9-c.pt", local_dir="./")

# Example usage:
#model_name = "mlx-community/OpenELM-3B"
save_dir = "/Users/asifahmed/Development/projectKIT/llm/base_model"

download_llm(model_name, save_dir)