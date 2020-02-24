#!/bin/bash
if [ ! -d results ]; then
	mkdir results
fi

procs=(1 128 256 512) 
times=(02:00:00 00:15:00 00:10:00 00:05:00)

for omp in yes no; do
	omp=$([ $omp == yes ] && echo _omp || echo )
	for i in $(seq 1 3); do
		for N in 1000 2000; do
			for j in "${!procs[@]}"; do
			n=${procs[$j]}
			time=${times[$j]}
				file=$N"_"$n"_"$i$omp
				if [ ! -f results/$file.txt ] || [ -f results/$file.err ]; then
					mpisubmit.bg -w $time -n $n -m smp -e "OMP_NUM_THREADS=3" --stdout results/$file.txt --stderr results/$file.err task2$omp -- $N
					echo $file submitted
				fi
			done
		done
	done
done