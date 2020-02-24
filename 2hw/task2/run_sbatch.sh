#!/bin/bash

BASE=~/_scratch/task2
if [ ! -d $BASE/results ]; then
	mkdir $BASE/results
fi

procs=(1 8 16 32 28)
times=(0-00:10:00 0-00:5:00 0-00:5:00 0-00:5:00 0-00:1:00)

a=$(squeue --user alichek95_1945 --partition=test | grep impi)
if [ "$a" == "" ]; then
	for i in $(seq 1 3); do
		for N in 1000 2000 3000; do
			for j in "${!procs[@]}"; do
				n=${procs[$j]}
				time=${times[$j]}
				file=$N"_"$n"_"$i
				if [ ! -f $BASE/results/$file.txt ]; then
					sbatch -p test -n $n --time=$time -o $BASE/results/$file.txt -e $BASE/results/$file.err impi $BASE/task2 $N
					echo $file submitted
				fi
			done
		done
	done
fi