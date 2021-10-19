## LAPS

This repository implements a library for running LAPS (language / abstraction and program search) experiments.

## Getting Started -- Dependencies and Requirements
The codebase is implemented in both Python and OCaml. The repository contains prebuilt OCaml binaries that have been tested on OS X. However, 

We also provide scripts for installing and running this repository on two MIT-specific computing clusters, which do not allow directly installing software and run on Linux architectures.
#### Setting up on OpenMind
These are instructions for setting up on the OpenMind computing cluster managed by the MIT BCS department.
1. Git clone this repository into a local directory on OpenMind. This is likely at `om2/user/<USERNAME>` after logging in via `ssh <USERNAME>@openmind7.mit.edu`
2. Run `setup_om.sh` from the top-level repository. In particular, this should copy over a Singularity container with the necessary dependencies and also rebuild a Linux-specific version of the dependencies locally.
3. You can now run experiments via standard SLURM commands, using the Singularity container to execute the Python/OCaml.
  - If you are running `evaluate_compression_model_scoring.py`, running your desired command with `--util_generate_cloud_command om` will automatically generate command-line parameters to run on OpenMind.
  - If you are running experiments from `evaluate_compression_model_scoring.py`, you can reference previous reported results [here](https://docs.google.com/spreadsheets/d/11-qKHK_pOyF4lfwhaonRQTZEqHTepPKI4MTdWYAU9hM/edit#gid=0).

#### Setting up on Supercloud.
These are instructions for setting up on the OpenMind computing cluster managed by Lincoln Labs.
1. Git clone this repository into a local directory on Supercloud. This is likely in a home directory after logging in via `USERNAME@txe1-login.mit.edu`.
2. Run `setup_supercloud.sh` from the top-level repository. In particular, this should switch you to using a Linux-specific build of the OCaml binaries (which we build on OM) and pip install the `laps_requirements.txt` locally.
3. You can now run experiments via standard SLURM commands [like these](https://supercloud.mit.edu/submitting-jobs). Note that if running interactively or writing your own batch commands, you should call `module load anaconda/2020b` to access the appropriate Python installation.
   - If you are running `evaluate_compression_model_scoring.py`, running your desired command with `--util_generate_cloud_command om` will automatically generate command-line parameters to run on OpenMind.
  - If you are running experiments from `evaluate_compression_model_scoring.py`, you can reference previous reported results [here](https://docs.google.com/spreadsheets/d/11-qKHK_pOyF4lfwhaonRQTZEqHTepPKI4MTdWYAU9hM/edit#gid=0).
