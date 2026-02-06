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
uv run python scripts/eval.py \
  policy:libero \
  --policy.pretrained_repo_id_or_path output/checkpoint
```

Terminal window 2:
```bash
cd clients/libero_client
uv run python main.py
```

## Troubleshooting
- If you have installed LIBERO before, make sure to remove cache with `rm -rf ~/.libero`.