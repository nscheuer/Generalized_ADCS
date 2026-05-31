# Prerequisities
- Linux or WSL Machine

# What is trajectory_planner?
`trajectory_planner` is the optimization module that computes reference attitude/control trajectories for difficult pointing maneuvers.

For more algorithm background, see https://nscheuer.github.io/SALTRO.

# WSL Setup
To install WSL onto on a Windows machine, open Powershell in Administrator Mode.
```powershell
wsl --install -d Ubuntu
```
After installation, Windows will open a new terminal window. Here, enter a `username` and `password`. Remember the `password`, as it is required to run `sudo` commands in the WSL Shell. After this is complete, you should have a working Linux shell.
```bash
user@machine:~$
```

## Connect WSL to VS Code
Open VS Code in Windows and install the [Remote-WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) extension. Now, you can open WSL using VS Code by selecting the blue arrow button in the bottom left of the window and selecting 'Connect to WSL'.

### Note about filesystems
When working with WSL, it is recommended to have projects stored/cloned on the WSL partition of your PC, rather than in the Windows filesystem. When opening File Explorer, you may note a new folder at the very bottom of the left pane, called 'Linux'. A recommended location for projects may be 'Linux/Ubuntu/home/user/Documents'. 

Running WSL on a Windows filepath is very slow, since Windows uses a NTFS filesystem, while WSL works with ext4 images, with a virtualization layer between. This layer can make certain operations extremely slow, such as `pip install ...` and `cmake ...`.

# Instructions
## Create Virtual Environment
Ensure that Python 3.x is installed in your WSL distro. This can be checked by opening a WSL terminal and running 'python3 --version'.
Create a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
```
Install Python dependencies using pip:
```bash
pip install -r requirements.txt
pip install git+https://github.com/jcrudy/choldate.git --no-build-isolation
```
The reason choldate has to be installed with --no-build-isolation is that its compilation depends on pip having access to Cython.

## Build trajectory_planner
`trajectory_planner` installation is documented in one canonical page:

- [Install_Trajectory_Planner.md](Install_Trajectory_Planner.md)

Use the Linux/WSL section there for dependency and build commands.

## Optional: Build SALTRO (saltro_py)
SALTRO installation is documented in one canonical page:

- [Install_SALTRO.md](Install_SALTRO.md)

`OldPlanner/` and `SALTRO/` are both included as git submodules in this
repository. If you did not clone with `--recurse-submodules`, initialise them
before building either optional add-on:

```bash
git submodule update --init --recursive OldPlanner SALTRO
```

## Debugging tplaunch
Install GDB:
```bash
sudo apt update
sudo apt install gdb
which gdb
```
