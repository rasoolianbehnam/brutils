source ~/brasoolian/brutils/bin/bashrc.sh ~/brasoolian/
install_miniconda2
source ~/brasoolian/brutils/bin/bashrc.sh ~/brasoolian/
cd ~/brasoolian/brutils/dependencies/conda_envs/
conda env create -f hadoop_original_2022_01_04.yml
source ~/brasoolian/brutils/bin/bashrc.sh ~/brasoolian/
set_up_jupyter
source ~/brasoolian/brutils/bin/bashrc.sh ~/brasoolian/
cp ~/brasoolian/brutils/dependencies/jupyter/notebook.json  ~/brasoolian/.jupyter/nbconfig/notebook.json
cd ~/brasoolian
echo ok