SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR="$(dirname ${SCRIPT_DIR})/data"
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -P $DATA_DIR
tar -xjf ${DATA_DIR}/LJSpeech-1.1.tar.bz2 -C $DATA_DIR
