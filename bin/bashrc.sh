HM=$1
unalias python 2>/dev/null
unalias pip 2>/dev/null
########################### ########################### ########################### ###########################
## functions
install_miniconda () {
    mkdir -p ~/.miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/.miniconda3/miniconda.sh
    bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
    rm -rf ~/.miniconda3/miniconda.sh
    # ~/.miniconda3/bin/conda init bash
    # ~/.miniconda3/bin/conda init zsh
}

install_miniconda2 () {
    mkdir -p ~/brasoolian/.miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/brasoolian/.miniconda3/miniconda.sh
    bash ~/brasoolian/.miniconda3/miniconda.sh -b -u -p ~/brasoolian/.miniconda3
    rm -rf ~/brasoolian/.miniconda3/miniconda.sh
    # ~/.miniconda3/bin/conda init bash
    # ~/.miniconda3/bin/conda init zsh
}

install_miniconda_mac () {
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh
}

install_julia () {
    mkdir -p ~/git
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz -O ~/git/julia.tar.gz
    cd ~/git/ && tar xvf julia.tar.gz --directory ./julia
}

function set_up_jupyter() {
  conda install -y -c anaconda git
  conda install -y -c conda-forge git-lfs
  conda install -c conda-forge -y jupyterlab jupyter_contrib_nbextensions jupyter_nbextensions_configurator


  jupyter contrib nbextension install --user

  jupyter nbextension enable notify/notify
  jupyter nbextension enable codefolding/main
  jupyter-nbextension enable execute_time/ExecuteTime
  jupyter-nbextension enable collapsible_headings/main
  jupyter-nbextension enable freeze/main
  jupyter-nbextension enable scratchpad/main
  jupyter-nbextension enable toc2/main
  jupyter-nbextension enable zenmode/main
  jupyter-nbextension enable hide_input/main
  jupyter-nbextension enable hide_input_all/main

  # jupyter vim binder:

  # Create required directory in case (optional)
  mkdir -p $(jupyter --data-dir)/nbextensions
  # Clone the repository
  cd $(jupyter --data-dir)/nbextensions && git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding \
  && jupyter nbextension enable vim_binding/vim_binding && cd -;
}

function echo_vars() {
  jupyter --data-dir
}

function beep() {
  echo -ne '\007'
}
## end functions
########################### ########################### ########################### ###########################
set -o vi
source ~/brasoolian/.miniconda3/bin/activate brutils || source ~/brasoolian/.miniconda3/bin/activate base || echo "environment not found"

########################### ########################### ########################### ###########################
if [ -z "$HM" ]; then
  echo "please enter home"
else
  export IPYTHONDIR=$HM/.ipython
  export JUPYTER_CONFIG_DIR=$HM/.jupyter
  export JUPYTER_RUNTIME_DIR=$HM/.local/share/jupyter
  export JUPYTER_DATA_DIR=$HM/.local/share/jupyter
  export AWS_CONFIG_FILE=$HM/.aws/config
  export AWS_SHARED_CREDENTIALS_FILE=$HM/.aws/credentials
  export PATH=$HM/.local/julia-1.6.2/bin:$PATH
  export JULA_NUM_THREADS=12
  export JULIA_DEPOT_PATH=$HM/.julia
fi