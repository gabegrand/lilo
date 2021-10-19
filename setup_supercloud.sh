# setup_om.sh: set up script | Author: Catherine Wong.
# Setup script for MIT Lincoln Lab's Supercloud computing cluster. Should be run upon cloning the directory. 

# Download LAPS-DreamCoder submodule dependency, which needs to be unpacked into laps/dreamcoder
git submodule update --init --recursive

# Pip install the requirements locally.
module load anaconda/2020b # Note that you should run this each time.
cat laps_requirements.txt |  xargs -n 1 pip install --user
# Install the NLTK word tokenize package.
python -m nltk.downloader 'punkt'

# Use the linux-specific OCaml binaries.
rm ocaml/bin/*
cp ocaml/linux_bin/* ocaml/bin/*