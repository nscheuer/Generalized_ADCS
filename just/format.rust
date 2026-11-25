set shell := ["bash", "-euo", "pipefail", "-c"]

default: format

# Apply rustfmt to every crate in the Rust workspace.
format:
    command -v cargo >/dev/null
    cargo fmt --manifest-path GenADCS/backends/rust/Cargo.toml --all
