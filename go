#!/bin/bash
docker stop ia; docker rm ia
( cd ia; docker buildx build --rm -t ia . )
if ! docker image inspect nn &>/dev/null; then
  docker stop nn; docker rm nn
  ( cd nn; docker buildx build --rm -t nn . )
fi
export MODELS=`cd ia/graph; pwd`
echo "export MODELS=$MODELS" >> ~/.bashrc
docker compose up