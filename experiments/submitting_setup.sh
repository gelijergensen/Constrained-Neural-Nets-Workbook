# This script clones the repository into a new place, cd's into the repository
# sets up a new virtual environment, and then sets up the repository

echo "Loading modules"
source ~/setup.sh

DATE=$(date +"%Y%m%d-%H%M%S")
echo "Making dir" $SCRATCH/clones/$DATE
mkdir -p $SCRATCH/clones/$DATE
cd $SCRATCH/clones/$DATE

echo "Cloning repository"
git clone git@github.com:gelijergensen/pyinsulate.git

echo "Setting up new virtual environment"
python3 -m venv env
source env/bin/activate

echo "Setting up repository"
cd pyinsulate
pip install -e .
