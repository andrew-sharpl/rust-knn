# run everything
dev:
    cargo test --lib && uv run maturin develop && uv run python scripts/benchmark.py

# just tests
test:
    cargo test --lib

# just benchmark
bench:
    uv run maturin develop && uv run python scripts/benchmark.py

# build without running
build:
    cargo build && uv run maturin develop
