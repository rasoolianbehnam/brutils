DIR=$(dirname "$0")
source $DIR/source.sh
if ! command -v git &> /dev/null; then     conda install -c conda-forge -y git; fi
if ! command -v aws &> /dev/null; then     conda install -c conda-forge -y awscli; fi
if ! command -v julia &> /dev/null; then install_julia; fi