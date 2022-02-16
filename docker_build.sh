#!/usr/bin/env bash
set -e

BASE_IMG="tensorflow_base:1.15.3-cuda10-cudnn7-py3"
TF_CC_IMG="tensorflow_cc:1.15.3-cuda10-cudnn7-py3"
SPIEL_IMG="my_spiel_cc:tf1.15.3-cuda10-cudnn7-py3"

# build tensorflow_base
docker build . -f Dockerfile.tensorflow_base -t ${BASE_IMG} \
--build-arg USE_PYTHON_3_NOT_2

# # build tensorflow_cc
docker build . -f Dockerfile.tensorflow_cc -t ${TF_CC_IMG} \
--build-arg BASE=${BASE_IMG}

# build my_spiel
docker build . -f Dockerfile.from_tensorflow_cc -t ${SPIEL_IMG} \
--build-arg TF_CC=${TF_CC_IMG} \
--build-arg BASE=${BASE_IMG}

