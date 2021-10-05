# setup_om.sh: set up script | Author: Catherine Wong.
# Setup script for MIT BCS's OpenMind computing cluster. Should be run upon cloning the directory. 

# Download LAPS-DreamCoder submodule dependency, which needs to be unpacked into laps/dreamcoder
git submodule update --init --recursive

# Download the singularity container (currently owned by Catherine Wong: zyzzyva@mit.edu).
cp /om2/user/zyzzyva/laps-dev-container.img ../laps-dev-container.img

# Remake the binaries for Linux. This requires using an alternate geomLib/logoLib.
rm -rf ocaml/solvers/geomLib
rm -rf ocaml/solvers/logoLib
mv ocaml/linux_solvers/geomLib_linux ocaml/solvers/geomLib
mv ocaml/linux_solvers/logoLib_linux ocaml/solvers/logoLib
cd ocaml && ../../laps-dev-container.img make compression_rescoring_api && ../../laps-dev-container.img make compression