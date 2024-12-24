from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import os
import sys

def main():
    try:
        # Add project root to Python path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(project_root)

        # Load config
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_name = config['model']['name']
        print(f"\nDownloading model: {model_name}")
        print("This may take a few minutes...")
        
        # Download model and tokenizer
        print("\nDownloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("\nDownloading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print("\nModel and tokenizer downloaded successfully!")
        print(f"Model name: {model_name}")
        
        # Print cache location
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"\nFiles cached in: {cache_dir}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()