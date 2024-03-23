#!/bin/bash

url_atomic="https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz"
url_graphwriter="https://github.com/rikdz/GraphWriter"
url_qagnn="https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip"
url_text2kg="https://github.com/cenguix/Text2KGBench"

echo "Downloading and preprocessing atomic..."
atomic_raw="./atomic/raw"
if [ ! -d "$atomic_raw" ]; then
    wget "$url_atomic"
    mkdir "$atomic_raw"
    tar -xzvf  "atomic_data.tgz" -C "$atomic_raw"
    rm "atomic_data.tgz"
fi
cd "./atomic" && python convert.py && cd ..
wait

echo "Downloading and preprocessing graphwriter..."
graphwriter_raw="./graph-writer/raw"
if [ ! -d "$graphwriter_raw" ]; then
    git clone "$url_graphwriter"
    cp -r "./GraphWriter/data" "$graphwriter_raw"
    rm -rf "./GraphWriter"
fi
cd "./graph-writer" && python convert.py && cd ..
wait

echo "Downloading and preprocessing qagnn..."
qagnn_raw="./qagnn/raw"
if [ ! -d "$qagnn_raw" ]; then
    if [ -d "../lm-kbc/conceptnet/raw" ]; then
        cp -r "../lm-kbc/conceptnet/raw" "$qagnn_raw"
    else
        wget $url_qagnn
        unzip "data_preprocessed_release.zip" -d "$qagnn_raw"
        rm "data_preprocessed_release.zip"
    fi
fi
cd "./qagnn" && python convert.py && cd ..
wait

echo "Downloading and preprocessing text2kg..."
text2kg_raw="./text2kg/raw"
if [ ! -d "$text2kg_raw" ]; then
    git clone "$url_text2kg"
    cp -r "./Text2KGBench/data" "$text2kg_raw"
    rm -rf "./Text2KGBench"
fi
cd "./text2kg" && python convert.py && cd ..
wait

echo "Downloading and preprocessing webnlg..."
cd "./webnlg" && python convert.py && cd..