#!/bin/bash

url_atomic = "https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz"
url_graphwriter = "https://github.com/rikdz/GraphWriter"
url_qagnn = "https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip"
url_text2kg = "https://github.com/cenguix/Text2KGBench"

echo "Downloading and preprocessing atomic..."
if [! -d "./atomic/raw"]; then
    wget &url_atomic
    mkdir "./atomic/raw"
    tar -xvf "atomic_data.tgz" -c "./atomic/raw"
    rm "atomic_data.tgz"


# echo "Downloading and preprocessing graphwriter..."
# git clone $url_graphwriter

# echo "Downloading and preprocessing qagnn..."
# wget $url_qagnn

# echo "Downloading and preprocessing text2kg..."
# git clone $url_text2kg


# echo "Downloading and preprocessing webnlg..."
# cd webnlg && python convert.py