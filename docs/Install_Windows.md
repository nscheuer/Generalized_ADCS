# Prerequisites
- C++, preferably using [Visual Studio](https://learn.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022)
- Python 3.13, with Python executable on system [PATH](https://realpython.com/add-python-to-path/)

# What is trajectory_planner?
`trajectory_planner` is the optimization module that computes reference attitude/control trajectories for difficult pointing maneuvers.

For more algorithm background, see https://nscheuer.github.io/SALTRO.

# Instructions
## Create Virtual Environment
### Ensure correct Python version
Ensure that the version of Python being used in Powershell is the correct one (13.x+). Do this by running:
```powershell
python --version
```
In case Powershell does not display your installed version of Python (13.x+), but you are confident it is installed on your machine, run Python (13.x+) explicitly using:
```powershell
py -3.13 --version
```

### Virtual environment
Inside the `Generalized_ADCS/` folder run in Powershell:
```powershell
python -m venv venv
.\.venv\Scripts\activate
```

### Install Python dependencies using pip
In the same terminal, run:
```powershell
pip install -r requirements.txt
```

## Build trajectory_planner
`trajectory_planner` installation is documented in one canonical page:

- [Install_Trajectory_Planner.md](Install_Trajectory_Planner.md)

Use the Windows section there for CMake, vcpkg, and build commands.

## Optional: Build SALTRO (saltro_py)
SALTRO installation is documented in one canonical page:

- [Install_SALTRO.md](Install_SALTRO.md)

`OldPlanner/` and `SALTRO/` are both included as git submodules in this
repository. If you did not clone with `--recurse-submodules`, initialise them
before building either optional add-on:

```powershell
git submodule update --init --recursive OldPlanner SALTRO
```
