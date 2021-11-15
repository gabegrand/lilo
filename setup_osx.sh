# setup_osx.sh: set up script | Author: Catherine Wong.
# Setup script for Mac OSX. Should be run upon cloning the directory. 

# Download LAPS-DreamCoder submodule dependency, which needs to be unpacked into laps/dreamcoder
git submodule update --init --recursive

# Pip install the requirements locally.
# Note that this has been tested with Python 3.7.7.
cat laps_requirements.txt |  xargs -n 1 pip install --user
# Install the NLTK word tokenize package.
python -m nltk.downloader 'punkt'