#!/usr/bin/env bash

export CC=gcc
export CXX=g++
export BUILD_WITH_ACPC=ON
work_path=${PWD}
export PYTHONPATH=${work_path}/third_party/open_spiel
export PYTHONPATH=${PYTHONPATH}:${work_path}/build/third_party/open_spiel/python
