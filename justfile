# run everything
dev:
    cargo test --lib && uv tool run maturin develop && uv run python scripts/benchmark.py

# rust tests
test:
    cargo test

# rust benchmarks
bench:
    cargo bench

# build python module
build:
    uv tool run maturin develop

# python tests
pytest:
    uv tool run maturin develop && uv run pytest tests/ -v

# python benchmarks
bench-py:
    uv tool run maturin develop && uv run python scripts/benchmark.py
