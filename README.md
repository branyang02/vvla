# vVLA Policies

## Setup
```
git submodule update --init --recursive
uv sync
uv run pre-commit install
```

## Testing
```
uv run pytest --cov=vvla --cov-report html
```
```
cd htmlcov
uv run python -m http.server 8000
```
