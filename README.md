# vLLA Policies

## Setup
```
git submodule update --init --recursive
uv sync
```

## Testing
```
uv run pytest --cov=vlla --cov-report html
```
```
cd htmlcov
uv run python -m http.server 8000
```
