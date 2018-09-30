#!/usr/bin/env bash
sudo nvidia-docker build . -f Dockerfile --tag attention_is_all_you_need
sudo nvidia-docker run -v `realpath MT_Data`:/data/ \
 --rm attention_is_all_you_need \
 --mode train \
 --config config.yml \
 --data /data/europarl-v7.tfrecords \
 --de_vocab /data/wiki.de_tokens.txt \
 --en_vocab /data/wiki.en_tokens.txt \
 --pre_trained_embedding_de /data/wiki.de_embeddings.npy \
 --pre_trained_embedding_en /data/wiki.en_embeddings.npy \
 --temp /data/
