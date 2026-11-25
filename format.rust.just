set shell := ["bash", "-euo", "pipefail", "-c"]

default: format

# Apply rustfmt to Rust backend crates. The workspace is outside GenADCS/tests.
format:
    command -v cargo >/dev/null
    cargo fmt --manifest-path GenADCS/backends/rust/Cargo.toml --all
