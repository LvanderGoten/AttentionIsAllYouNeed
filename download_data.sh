#!/usr/bin/env bash
EMBEDDING_LIMIT=100000
mkdir MT_Data
wget http://www.statmt.org/europarl/v7/de-en.tgz -P ./MT_Data
tar -xf MT_Data/de-en.tgz -C ./MT_Data
export PYTHONPATH=`pwd`
python3 conversion/europarl.py --de MT_Data/europarl-v7.de-en.de \
--en MT_Data/europarl-v7.de-en.en \
--output MT_Data/europarl-v7.tfrecords \
--output_de_vocab MT_Data/europarl-v7-de-vocab.txt \
--output_en_vocab MT_Data/europarl-v7-en-vocab.txt &
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec -t 0 -qO- | head -n $((EMBEDDING_LIMIT + 1)) > ./MT_Data/wiki.en.vec &
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec -t 0 -qO- | head -n $((EMBEDDING_LIMIT + 1)) > ./MT_Data/wiki.de.vec &
wait < <(jobs -p)
echo "Converting embeddings.."
python3 conversion/separate_embedding.py --embedding MT_Data/wiki.en.vec --limit $EMBEDDING_LIMIT
python3 conversion/separate_embedding.py --embedding MT_Data/wiki.de.vec --limit $EMBEDDING_LIMIT
echo "Done"
