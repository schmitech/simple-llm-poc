from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import warnings
from .config import ModelConfig

class LLMInference:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self):
        """Load model with appropriate optimizations"""
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pytorch_utils")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            use_cache=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            padding=True,
            truncation=True
        )
        
        self.pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=self.config.max_length
        )
        
    def generate(self, prompt: str, max_length: int = None) -> str:
        """Generate text from prompt"""
        try:
            if self.pipeline is None:
                self.load_model()
                
            max_length = max_length or self.config.max_length
            response = self.pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                truncation=True
            )
            
            # Clean up the response to remove the prompt from the output
            generated_text = response[0]['generated_text']
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text

        except Exception as e:
            self.clear_memory()
            raise e

    def clear_memory(self):
        """Clear model from memory when not in use"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    def __del__(self):
        """Cleanup when object is deleted"""
        self.clear_memory()