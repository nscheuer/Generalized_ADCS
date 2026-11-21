# Small installed examples

This folder contains short examples shipped with the Python package. From an
environment where `generalized-adcs` is installed, run one with:

```bash
python -m ADCS.examples.basic_simulation
python -m ADCS.examples.orbit_elements
python -m ADCS.examples.environment_field
```

The larger example collection lives in the repository's top-level
[`examples/`](https://github.com/nscheuer/Generalized_ADCS/tree/main/examples)
directory. Clone the repository to get the complete collection:

```bash
git clone --depth 1 https://github.com/nscheuer/Generalized_ADCS.git
```

GitHub does not provide a direct `curl` download for one folder. To fetch only
`examples/`, the command below uses GitHub's tree API to find its files and
`curl` to download those files. It requires Python 3 and `curl`:

```bash
mkdir -p examples
curl -fsSL "https://api.github.com/repos/nscheuer/Generalized_ADCS/git/trees/main?recursive=1" |
python3 -c 'import json, pathlib, subprocess, sys; [ (pathlib.Path(f["path"]).parent.mkdir(parents=True, exist_ok=True), subprocess.run(["curl", "-fsSL", "https://raw.githubusercontent.com/nscheuer/Generalized_ADCS/main/" + f["path"], "-o", f["path"]], check=True)) for f in json.load(sys.stdin)["tree"] if f["type"] == "blob" and f["path"].startswith("examples/") ]'
```
