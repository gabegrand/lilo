# LILO

# Installation

To run LILO, you'll need an environment with:

- Python 3.7 (for backwards compatibility with DreamCoder)
- OCaml (required by DreamCoder)
- Rust (required by Stitch)

The easiest way to get this environment is to use the Docker container provided in this repository. If you'd like to build a local environment, see the instructions below.

## LILO Docker container (Recommended)

First, install [Docker](https://docs.docker.com/get-docker/) on your local machine. Then, pull the Docker container from Docker Hub:

```
docker pull gabegrand/lilo
```

Alternatively, you can clone the repo and build the Docker container locally:

```
git clone --recurse-submodules https://github.com/gabegrand/lilo
cd lilo
docker build -t lilo .
```

## Building a local LILO environment

LILO is compatible with both Linux (tested on Debian 12 "bookworm" and Ubuntu 20.04 "Focal Fossa") and MacOS (tested on M1 Mac running MacOS 12 Monterey).

### Prerequisites
- Conda / Miniconda: https://docs.conda.io/en/latest/miniconda.html
- OCaml: https://ocaml.org/install
- Rust: https://www.rust-lang.org/tools/install

### Building the Python environment

To build a local environment, first clone the repo:

```
git clone --recurse-submodules https://github.com/gabegrand/lilo
cd lilo
```

Create the conda environment. Note that you may wish to install the [libmamba](https://conda.github.io/conda-libmamba-solver/getting-started/) solver, which significantly speeds up environment resolution.
```
conda env create -f environment.yml
```

### Building the OCaml binaries

```
cd dreamcoder
make setup-ocaml
make
```

## Testing the environment

As a quick test to confirm that the environment is installed correctly, run:
```
conda activate lilo
python
>>> from run_experiment import *
>>> p = Program.parse("(_rconcat _x _y)")
>>> p.evaluate([])
'xy'
```

# Running experiments

## OpenAI API key

To run experiments with the OpenAI API, you'll need to set the `OPENAI_API_KEY` environment variable to your API key. You can find your API key at https://platform.openai.com/account/api-keys.

```
export OPENAI_API_KEY=<sk-...a123>
```

## Experiments overview


### LILO

### DreamCoder

### LLM Solver

# Data release

We make all data from our experiments public: `s3://lilo-experiments-release/lilo_arxiv_v1/`

To view this data in your browser, you can use the AWS CLI (requires AWS account sign-up): https://s3.console.aws.amazon.com/s3/buckets/lilo-experiments-release

# Citations

```
@article{grand2023lilo,
  title={LILO: Learning Interpretable Libraries by Compressing and Documenting Code},
  author={Grand, Gabriel and Wong, Lionel and Bowers, Matthew and Olausson, Theo X. and Liu, Muxin and Tenenbaum, Joshua B. and Andreas, Jacob},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```

# Acknowledgements

LILO inherits from the LAPS codebase (https://github.com/CatherineWong/laps). We gratefully acknowledge Lionel Wong for the use of their codebase as a foundation and for technical support.

# License

MIT License Copyright (c) 2023 Gabriel Grand

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.