#!/usr/bin/env bash
set -euo pipefail

rust_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:-release}"
if [[ "$profile" != debug && "$profile" != release ]]; then
    echo "Usage: $0 [debug|release]" >&2
    exit 2
fi

is_macos=false
if [[ "$(uname -s)" == Darwin ]]; then
    is_macos=true
fi

packages=(
    "bias:_bias"
    "state:_state"
    "helpers/math:_helpers"
    "helpers/random:_random"
    "helpers/math/frames:_frames"
    "hardware/satellite_geometry:_geometry"
    "hardware/ADCS_subsystem:_subsystem"
    "hardware/OBC_subsystem:_obc_subsystem"
    "hardware/satellite:_satellite"
    "environment/atmosphere:_atmosphere"
    "environment/ephemeris:_ephemeris"
    "environment/gravity:_gravity"
    "environment/magnetic_field_model:_magnetic_field_model"
    "environment/planetary_body:_planetary_body"
    "dynamics:_dynamics"
    "config:_config"
    "solver:_solver"
    "propagation/coupled_propagators:_coupled"
    ".:_simulate"
)

for entry in "${packages[@]}"; do
    directory="${entry%%:*}"
    module="${entry##*:}"
    package_dir="$rust_root/$directory"
    options=(build --manifest-path "$package_dir/Cargo.toml" --quiet)
    if [[ "$directory" == "." ]]; then
        # The root simulation crate is excluded from Cargo's default-members.
        options+=(--package adcs-omni-rust-simulate)
    fi
    case "$directory" in
        bias|state|helpers/math|helpers/math/frames|helpers/random|hardware/satellite_geometry|hardware/ADCS_subsystem|hardware/OBC_subsystem|hardware/satellite|environment/atmosphere|environment/ephemeris|environment/gravity|environment/magnetic_field_model|environment/planetary_body|dynamics|config|solver|propagation/coupled_propagators|.)
            options+=(--features python-extension)
            ;;
    esac
    if [[ "$profile" == release ]]; then
        options+=(--release)
    fi
    if [[ "$is_macos" == true ]]; then
        # Python provides these symbols when it loads an extension module.
        # Passing this through Cargo config avoids an inherited
        # CARGO_ENCODED_RUSTFLAGS value silently taking precedence over
        # RUSTFLAGS on hosted runners.
        options+=(--config 'target.aarch64-apple-darwin.rustflags=["-C", "link-arg=-Wl,-undefined,dynamic_lookup"]')
        options+=(--config 'target.x86_64-apple-darwin.rustflags=["-C", "link-arg=-Wl,-undefined,dynamic_lookup"]')
    fi
    cargo "${options[@]}"
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            built_module="$rust_root/target/$profile/${module}.dll"
            installed_module="$package_dir/${module}.pyd"
            ;;
        Darwin*)
            built_module="$rust_root/target/$profile/lib${module}.dylib"
            installed_module="$package_dir/${module}.so"
            ;;
        *)
            built_module="$rust_root/target/$profile/lib${module}.so"
            installed_module="$package_dir/${module}.so"
            ;;
    esac
    cp "$built_module" "$installed_module"
    echo "Built $directory/$module"
done
