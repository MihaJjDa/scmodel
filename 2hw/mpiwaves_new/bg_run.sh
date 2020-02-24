#!/bin/bash
set -e

BIN_PATH="./bin"
LOG_PATH="./logs"

num_threads=1
num_nodes=$(($2*$3*$4))

if [ $5 == "omp" ] 
then
    num_threads=3
fi

echo "running on $num_nodes with $num_threads threads"

out_filename="$5_$1_$2x$3x$4"
out_filename="test"

mpisubmit.bg -n $num_nodes -m smp -env "OMP_NUM_THREADS=$num_threads" -w $6:00\
    --stdout "${LOG_PATH}/$out_filename.out" --stderr "${LOG_PATH}/$out_filename.err"\
    ./${BIN_PATH}/mpiwaves -- $1 $2 $3 $4 1
