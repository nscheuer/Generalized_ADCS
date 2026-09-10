# One Wheel Is Enough — IAC-26 paper artifact

Code and data for **"One Wheel Is Enough: Control Strategies for the 3+1
Actuator Architecture Across Small Satellite Mission Profiles"**
(IAC-26,B4,6A,2,x109468), presented at the 77th International Astronautical
Congress.

> ### ➡️  Everything the paper reports is in [`papers/IAC_1RW/`](papers/IAC_1RW/)
> Start with **[`papers/IAC_1RW/README.md`](papers/IAC_1RW/README.md)**. It maps
> every figure and every table in the manuscript to the script that produces it
> and the data it reads.

This branch is a frozen snapshot: the simulation framework as it stood for the
campaign, plus the paper's own harness, campaign data, and figure scripts. It is
not the framework's development branch. The tag `iac26-submission` marks the
state the manuscript was written from.

## Layout

| path | what it is |
|---|---|
| [`papers/IAC_1RW/`](papers/IAC_1RW/) | **the paper**: simulation harness, campaign generators, figure scripts, and `output_data/` with the per-trial results behind every reported number |
| `papers/Generalized_ACS/_paper1_sim.py` | the earlier generalized-ACS harness, kept because the paper's Campaign R reconciles against it |
| `ADCS/` | the attitude determination and control framework the campaign runs on |
| `testing/`, `examples/`, `docs/` | the framework's own tests, examples, and documentation |

## Running it

```bash
pip install -e .
python papers/IAC_1RW/fig_grid_dots.py     # prints the Table 3 statistics
```

Run scripts from the repository root. Each puts the repository root first on
`sys.path`, so the `ADCS` package in this checkout is the one that executes.
See `papers/IAC_1RW/README.md` for per-script detail and runtimes.

**Clone size.** `papers/IAC_1RW/output_data/` holds about 1.8 GB of per-trial
results, which is what makes the campaign statistics re-derivable without
re-running it. To skip the data and take only the code:

```bash
git clone --filter=blob:none --no-checkout <url> && cd Generalized_ADCS
git sparse-checkout set --no-cone '/*' '!papers/IAC_1RW/output_data'
git checkout paper/iac-1rw-clean
```

## Framework and license

The underlying framework is open source and developed separately; this branch
carries a snapshot of it, not its latest state. See `LICENSE`, `CONTRIBUTING.md`,
and `docs/` for the framework itself.
