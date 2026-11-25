set shell := ["bash", "-euo", "pipefail", "-c"]

default: all

rust_manifest := "GenADCS/backends/rust/Cargo.toml"

# Type-check Rust workspace crates without compiling test targets.
typecheck:
    command -v cargo >/dev/null
    cargo check --manifest-path {{rust_manifest}} --workspace

# Run Clippy on workspace library and binary targets, excluding test targets.
lint:
    command -v cargo >/dev/null
    cargo clippy --manifest-path {{rust_manifest}} --workspace -- -D warnings

all: typecheck lint
