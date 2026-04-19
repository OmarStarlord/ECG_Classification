#!/bin/bash
docker stop ia; docker rm ia
( cd ia; docker buildx build --rm -t ia . )
docker stop nn; docker rm nn
( cd nn; docker buildx build --rm -t nn . )
echo "Beware if you try later with docker compose up"
echo "Before export this variable using :"
echo "export MODELS=\`cd ia/graph; pwd\`"
export MODELS=`cd ia/graph; pwd`
docker compose up