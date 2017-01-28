#!/usr/bin/env bash
set -x
set -e

MAKE_TARGET=html

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

# Install dependencies with miniconda
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
if [[ ! -f miniconda.sh ]]
then
   wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
   -O miniconda.sh
fi
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
cd ..
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda
popd

# Configure the conda environment and put it in the path using the
# provided versions.
conda create -n $CONDA_ENV_NAME --yes --quiet python=3.5
source activate testenv

# Install pip dependencies.
pip install -r requirements.txt

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make $MAKE_TARGET 2>&1 | tee ~/log.txt

cd -
set +o pipefail

echo "Finished building docs."
echo "Artifacts in $CIRCLE_ARTIFACTS"
