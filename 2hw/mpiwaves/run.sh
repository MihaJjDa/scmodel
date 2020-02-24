#!/bin/bash

OMP_NUM_THREADS=$2
mpicxx -O3 -fopenmp main.cpp
mpirun -n $1 ./a.out $3 $4 $5 $6
