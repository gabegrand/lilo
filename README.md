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
These are instructions for setting up on the Supercloud computing cluster managed by Lincoln Labs.
1. Git clone this repository into a local directory on Supercloud. This is likely in a home directory after logging in via `USERNAME@txe1-login.mit.edu`.
2. Run `setup_supercloud.sh` from the top-level repository. In particular, this should switch you to using a Linux-specific build of the OCaml binaries (which we build on OM) and pip install the `laps_requirements.txt` locally.
3. You can now run experiments via standard SLURM commands [like these](https://supercloud.mit.edu/submitting-jobs). Note that if running interactively or writing your own batch commands, you should call `module load anaconda/2020b` to access the appropriate Python installation.
   - If you are running `evaluate_compression_model_scoring.py`, running your desired command with `--util_generate_cloud_command supercloud` will automatically generate command-line parameters to run on Supercloud.
  - If you are running experiments from `evaluate_compression_model_scoring.py`, you can reference previous reported results [here](https://docs.google.com/spreadsheets/d/11-qKHK_pOyF4lfwhaonRQTZEqHTepPKI4MTdWYAU9hM/edit#gid=0).

## Quickstart.
#### Experiments.
Experiments are run and managed through Config files. Example config files can be found in `experiments/configs`.

The Python entrypoint to these is `run_experiment.py`. Example usage: `python run_experiment.py --config_dir experiments/configs --config_file dreamcoder_compositional_graphics_200_human.json`.

#### DreamCoder compression API.
We implement a rewritten API over the original DreamCoder compressor for ease of use and adaptation into new library learning and library ranking functions.
If you are just looking to work with this API, you may find it more useful to examine the following:

1. The Python entrypoint to run compressor-specific tests is `evaluate_compression_model_scoring.py`. Example usage: `python evaluate_compression_model_scoring.py -k test_discrimination_original_final_libraries_full --db_no_model_training` will run a test that benchmarks compressor performance on increasingly large numbers of programs in a training set.
2. We implement a Python wrapper over the compressor that makes API calls in `laps_grammar.py`. See: `_send_receive_compressor_api_call` for an example of the calling syntax, and `_get_compressed_grammmar_and_rewritten_frontiers` for a wrapper function that gets a compressed grammar and rewritten frontiers for a set of train and test programs.
3. Finally, the OCaml implementation itself is in `ocaml/solvers/compression_rescoring_api.ml` and `ocaml/solvers/compression_rescoring_api_utils.ml`. The former specifies all of the API functions; the latter implements them.

