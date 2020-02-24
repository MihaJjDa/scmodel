#!/bin/bash
set -e

MPICXX=mpicxx

SRC_PATH="./src"
OBJ_PATH="./obj"
BIN_PATH="./bin"
LOG_PATH="./logs"

echo "Clean"
rm -rf ${OBJ_PATH}/*.o
rm -rf ${BIN_PATH}/*

echo "Compile"
${MPICXX} -O5 -o ${BIN_PATH}/mpiwaves ${SRC_PATH}/mpiwaves.cpp
${MPICXX} -O5 -qsmp=omp -o ${BIN_PATH}/mpiwaves_omp ${SRC_PATH}/mpiwaves.cpp
