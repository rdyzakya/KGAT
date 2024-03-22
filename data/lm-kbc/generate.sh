#!/bin/bash

url_lmkbc2022="https://github.com/lm-kbc/dataset2022"
url_lmkbc2023="https://github.com/lm-kbc/dataset2023"
url_mars="https://github.com/zjunlp/MKG_Analogy"

url_freebase="https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip"
url_freebase_mid2name="https://drive.usercontent.google.com/download?id=0B52yRXcdpG6MaHA5ZW9CZ21MbVk&export=download&authuser=0&confirm=t&uuid=9f09fae2-0885-4aff-bf65-e55c9ebeac19&at=APZUnTW9rF2jBHGa2Iqq5_zrX7kv%3A1709103664110"

url_conceptnet="https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip"

echo "Downloading and preprocessing conceptnet..."
conceptnet_raw="./conceptnet/raw"
if [ ! -d "$conceptnet_raw" ]; then
    if [ -d "../subgraph-gen/qagnn/raw" ]; then
        cp -r "../subgraph-gen/qagnn/raw" "$conceptnet_raw"
    else
        wget $url_conceptnet
        unzip "data_preprocessed_release.zip" -d "$conceptnet_raw"
        rm "data_preprocessed_release.zip"
    fi
fi
cd "./conceptnet" && python convert.py && cd ..

echo "Downloading and preprocessing freebase..."
freebase_raw="./freebase/raw"
if [ ! -d "$freebase_raw" ]; then
    wget "$url_freebase"
    unzip "FB15K-237.2.zip" -d "$freebase_raw"
    rm "FB15K-237.2.zip"
    wget "$url_freebase_mid2name" -O "mid2name.gz"
    gunzip -c "mid2name.gz" > "$freebase_raw/mid2name.tsv"
    rm "mid2name.gz"
fi
cd "./freebase" && python convert.py && cd ..

echo "Downloading and preprocessing lm-kbc2022..."
lmkbc2022_raw="./lm-kbc2022/raw"
if [ ! -d "$lmkbc2022_raw" ]; then
    git clone "$url_lmkbc2022"
    cp -r "./dataset2022/data" "$lmkbc2022_raw"
    rm -rf "./dataset2022"
fi
cd "./lm-kbc2022" && python convert.py & cd ..

echo "Downloading and preprocessing lm-kbc2023..."
lmkbc2023_raw="./lm-kbc2023/raw"
if [ ! -d "$lmkbc2023_raw" ]; then
    git clone "$url_lmkbc2023"
    cp -r "./dataset2023/data" "$lmkbc2023_raw"
    rm -rf "./dataset2023"
fi
cd "./lm-kbc2023" && python convert.py & cd ..

echo "Downloading and preprocessing mars..."
mars_raw="./mars/raw"
if [ ! -d "$mars_raw" ]; then
    git clone "$url_mars"
    cp -r "./MKG_Analogy/MarT/dataset/MarKG" "$mars_raw"
    rm -rf "./MKG_Analogy"
fi
cd "./mars" && python convert.py & cd ..


echo "woiii"