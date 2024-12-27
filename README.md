# Simple LLM PoC

A lightweight proof of concept for working with open-source Large Language Models (LLMs) locally. This project demonstrates how to set up, load, and interact with LLMs on your local machine, with a focus on minimal dependencies and clear implementation.

## Overview

This project provides a simple framework for:
- Loading and running open-source LLMs locally
- Basic text generation and inference
- Memory-efficient model handling
- Easy configuration management

## Requirements

- Python 3.10 or higher
- macOS (Intel or Apple Silicon) or Linux
- Minimum 16GB RAM recommended
- At least 20GB free disk space
- Conda or Miniconda

## Project Structure

```bash
simple-llm-poc/
├── src/                   
│   ├── config.py          # Configuration management
│   └── model.py           # Model implementation
├── scripts/               
│   ├── download_model.py  # Model download script
│   └── test_inference.py  # Testing script
└── configs/               
    └── config.yaml        # Model configuration
```

## Quick Start

1. Clone the repository
```bash
git clone https://github.com/schmitech/simple-llm-poc.git
cd simple-llm-poc
```

2. Set up the environment
```bash
# Install Miniconda if you haven't already
brew install --cask miniconda

# Initialize conda (if you haven't done this before)
conda init zsh  # or conda init bash if you're using bash
# Restart your terminal after this step

# Create and activate the environment
conda create -n simple-llm-poc python=3.10
conda activate simple-llm-poc
```

3. Install dependencies
```bash
# Install PyTorch first
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other requirements
pip install transformers datasets accelerate bitsandbytes pyyaml
```

4. Download the model
```bash
# Approximately 1.1GB space for the TinyLlama model
python scripts/download_model.py
# To see how much space it's using:
du -sh ~/.cache/huggingface/hub/
```

5. Run the test script
```bash
python scripts/test_inference.py
```

## Configuration

The project uses a YAML configuration file (`configs/config.yaml`) for model settings, using the 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' model.

## Usage Examples

### Basic Text Generation
```python
from src.config import ModelConfig
from src.model import LLMInference

# Load configuration
config = ModelConfig.from_yaml('configs/config.yaml')

# Initialize model
llm = LLMInference(config)

# Generate text
prompt = "Explain what LLMs are in one sentence:"
response = llm.generate(prompt)
print(response)

# Clean up
llm.clear_memory()
```

## Memory Management

The project includes built-in memory management features:
- Automatic model cleanup after generation
- Manual memory clearing with `clear_memory()`
- Efficient resource handling for longer sessions
- To remove the model: ```rm -rf ~/.cache/huggingface/hub/*```

## Troubleshooting

### Common Issues

1. Import Errors
```bash
# If you see "No module named 'src'"
# Make sure you're running from the project root directory
cd simple-llm-poc
python scripts/test_inference.py
```

2. Memory Issues
```python
# If you encounter memory errors, try:
llm.clear_memory()  # Clear current model from memory
```

3. CUDA/MPS Errors
```yaml
# In config.yaml, ensure device is set correctly:
model:
  device: null  # Let PyTorch handle device management
```

## Advanced Usage

### Using Different Models

You can easily switch to different models by modifying the config:

```yaml
model:
  name: "microsoft/phi-2"  # Or any other Hugging Face model
```

Recommended models for local use:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~1.1B parameters)
- microsoft/phi-2 (~2.7B parameters)
- TheBloke/Mistral-7B-v0.1-GGUF (Quantized version)

### Memory Optimization

For larger models or limited memory:
```python
# In your code
model = LLMInference(config)
try:
    response = model.generate(prompt)
finally:
    model.clear_memory()  # Always clear memory after use
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2 License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Hugging Face team for their transformers library
- Thanks to the TinyLlama team for their efficient model implementation

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{simple_llm_poc,
  author = {Remsy Schmilinsky},
  title = {Simple LLM PoC},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/schmitech/simple-llm-poc}
}
```

Copyright 2024 Schmitech Inc.