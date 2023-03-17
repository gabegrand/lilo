# setup_osx.sh: set up script | Author: Catherine Wong.
# Setup script for Mac OSX. Should be run upon cloning the directory.

# Download LAPS-DreamCoder submodule dependency, which needs to be unpacked into laps/dreamcoder
git submodule update --init --recursive

# Create a new Conda environment called `laps` with Python 3.7.7
conda env create -f environment.yml
conda activate laps

pip install pregex==1.0.0 --ignore-requires-python

# Install the NLTK word tokenize package.
python -m nltk.downloader 'punkt'
