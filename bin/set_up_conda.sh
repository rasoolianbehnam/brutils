SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ~/.miniconda3/bin/activate
source $SCRIPT_DIR/bashrc.sh ~/brasoolian/
conda env create --force
source ~/.miniconda/bin/activate brutils
python -m pip install -r requirements.txt
set_up_jupyter
python -m ipykernel install --user --name=brutils
