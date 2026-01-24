# Libero Client

## Requirements
```bash
sudo apt-get install cmake
```

## Installation
Libero uses a separate virtual environment. To set it up, go into the `libero_client` directory and install the virtual environment:
```bash
cd clients/libero_client
uv sync # This installs a separate virtual environment!!
```

## Usage
Terminal window 1:
```bash
uv run python scripts/eval.py policy:libero
```

Terminal window 2:
```bash
cd clients/libero_client
uv run python main.py
```