# funrobo_kinematics

**funrobo_kinematics** is a Python-based teaching library for explring robot kinematics, visualization, path and trajectory planning, etc. It serves as a **visualization tool (viz tool)** for working on kinematics modeling and analysis. It accompanies the class activities in modules 2-4 focusing on the following:
1. **Forward position kinematics (FPK)**
2. **Forward velocity kinematics (FVK)**
3. **Inverse position kinematics (IPK)**
4. **Trajectory generation**


## Viz Tool

<img src = "media/FPK.png">

## Prerequisites

#### Recommended tools
- **Visual Studio Code (VS Code)** (I strongly recommend using this, if you don’t already do. It’s the best IDE in my humble opinion). Follow the instructions [here to install.](https://code.visualstudio.com/download)


## Project structure (overview)
```bash
funrobo_kinematics/
  funrobo_kinematics/        # Python package
    core/                   # Core library (arm models, utils, visualization)
    __init__.py
  scripts/                  # Runnable demo scripts (YOUR CODE SHOULD GO HERE!)
  tests/                    # Automated tests (USE THIS TO CHECK YOUR WORK)
  docs/
  assignments/              # Description for mini-projects
  environment.yml           # Conda environment definition
  pyproject.toml            # Package configuration
  README.md

```


## Step-by-step setup

- You can complete this assignment on any computer OS: Mac, Windows, Linux, etc. All you need is Python 3 interpreter (code was tested using Python 3.11).


### Step 1: Install Python 3 (if not already installed)
- First, check if you have Python3, to do that, open your terminal and type:
```bash
$ python3 --version     # <--- type this
Python 3.11.14          # <--- you should see something like this
```
- If you don’t have Python installed, follow this [tutorial here](https://realpython.com/installing-python/) to install it.


### Step 2: Install Miniconda
We use **conda** to handle our virtual environments in Python
1. Install Miniconda for your OS: https://www.anaconda.com/docs/getting-started/miniconda/main
2. After installation, open a new terminal
3. (Optional but recommended) Disable auto-activation of the base environment:
```base
$ conda config --set auto_activate_base false
```


### Step 2: Get this repository from Github
I recommend that one teammate forks the repository, and others clone from that fork.
- Fork this repository on Github
- Clone your fork:
```base
$ git clone <YOUR_FORK_URL>
$ cd funrobo_kinematics
```


### Step 3: Create a conda environment
- From the root of the repository (same folder as ``environment.yml``)
```bash
# cd to the project folder
$ cd funrobo_kinematics
$ conda env create -f environment.yml
```
- Activate the environment:
```bash
$ conda activate funrobo
```

### Step 4: Install the package (in edit mode)
- From the root of the repository
```bash
# cd to the project folder
$ cd funrobo_kinematics
$ pip install -e .
```


## How to run

### Run the main script
All example scripts live in the ``examples/`` folder
```bash
$ python examples/basic.py
```

### Running tests
Tests are written using **pytest**.
- From the repo root:
```bash
$ pytest
```

- Run a specific test file (e.g., the fk solver test):
```bash
$ pytest tests/test_fk.py
```

- There is a known error where the pytest fails because of some system-wide dependencies that are unmet. If you encounter that, use this command to run the test:
```bash
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_fk.py
```


## Updating out the course

Through the duration of the course, we will be pushing updates to the repository. Please run the following commands whenever there's an update

```bash
# pull the latest version from github
$ git pull

# activate the conda environment
$ conda activate funrobo

# install the package
$ pip install -e .
```


### Usage Guide

<img src = "media/arm-kinematics-viz-tool.png">


### Generative AI Use Disclosure
- Please make sure to briefly describe what and how generative AI tools were used in developing the contents of your work.
- Acceptable use:
    - To research a related topic to the subject at hand
    - As a substitute to "Stackoverflow" guides on quick programming how-tos, etc.
- Unacceptable use:
    - Directly copying large swaths of code from a ChatGPT response to a prompt in part or entirely related to the assignment's problem

For instance, I used ChatGPT in generating the docstrings for the code in this repository as well as drafting this README.

