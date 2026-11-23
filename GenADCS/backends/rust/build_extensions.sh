#!/usr/bin/env bash
set -euo pipefail

rust_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:-release}"
if [[ "$profile" != debug && "$profile" != release ]]; then
    echo "Usage: $0 [debug|release]" >&2
    exit 2
fi

packages=(
    "bias:_bias"
    "state:_state"
    "helpers/math:_helpers"
    "helpers/math/frames:_frames"
    "hardware/satellite_geometry:_geometry"
    "hardware/ADCS_subsystem:_subsystem"
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
    case "$directory" in
        state|environment/atmosphere|environment/ephemeris|environment/gravity|environment/magnetic_field_model|propagation/coupled_propagators)
            options+=(--features python-extension)
            ;;
    esac
    if [[ "$profile" == release ]]; then
        options+=(--release)
    fi
    cargo "${options[@]}"
    cp "$package_dir/target/$profile/lib${module}.so" "$package_dir/${module}.so"
    echo "Built $directory/$module"
done
