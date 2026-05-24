# Install SALTRO

SALTRO is an optional add-on module.

The C++ source lives in the separate repository
[`nscheuer/SALTRO`](https://github.com/nscheuer/SALTRO), which
Generalized_ADCS pulls in as a git submodule at `./SALTRO/`.

## Initialise the submodule

If you cloned Generalized_ADCS without `--recurse-submodules`:

```bash
git submodule update --init --recursive SALTRO
```

(Re-run this after pulling new commits if the submodule pointer changed.)

For installation instructions, use the official SALTRO installation page:

- https://nscheuer.github.io/SALTRO/datasheets/Installation.html
