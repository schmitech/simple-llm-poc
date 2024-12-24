# First import system modules and set path
import sys
import os

# Add project root to Python path BEFORE other imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Now import the project modules
from src.config import ModelConfig
from src.model import LLMInference

def main():
    try:
        # Load config
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        config = ModelConfig.from_yaml(config_path)
        
        # Initialize model
        llm = LLMInference(config)
        
        # Test prompts
        prompts = [
            "Explain what LLMs are in one sentence:",
            "Write a haiku about programming:",
            "What is the capital of France?"
        ]

        print("\nRunning inference tests...")
        print("-" * 50)
        
        for prompt in prompts:
            try:
                print(f"\nPrompt: {prompt}")
                response = llm.generate(prompt)
                print(f"Response: {response}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing prompt: {e}")
        
        # Clean up
        llm.clear_memory()
        print("\nTests completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()