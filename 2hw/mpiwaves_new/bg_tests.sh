#!/bin/bash
set -e

# пути для логов
LOG_PATH="./logs"

# убиваем все имеющиеся задачи у пользователя
llcancel -u $(whoami)

# очищаем папку с логами
echo "cleaning"
rm -rf $LOG_PATH/*

# список параметров: размеры mpi-сеток и максимальное время работы
grid_array=("8 4 4" "8 8 4" "8 8 8")
mpi_time_array=(15 10 5)

echo "computing"

# считаем для перечисленных N
for N in 512 1024 1536
do
    # и для всех параметров
    for i in 0 1 2
    do
        # считаем для single и omp режима
        for mode in "single" "omp"
        do
            # максимальное время выполнения
            mpi_time=${mpi_time_array[i]}

            # размеры mpi-сетки
            grid=${grid_array[i]}

            # путь к файлам stdout и stderr 
            log_filename="${mode}_${N}_$(echo ${grid} | tr " " "x")"

            # число mpi нод
            num_nodes=$(( $(echo ${grid} | tr " " "*") ))

            # бинарник без OpenMP
            binary="./bin/mpiwaves"
            if [ $mode == "omp" ]
            then
                # бинарник с OpenMP
                binary="./bin/mpiwaves_omp"
            fi

            # печатаем информацию о задаче
            echo "================================================"
            echo "binary=$binary"
            echo "nodes=$num_nodes"
            echo "grid=$grid"
            echo "log=$log_filename"
            echo "mpi_time=$mpi_time"

            # сабмитим задачу в очередь
            mpisubmit.bg -n $num_nodes -m smp -env "OMP_NUM_THREADS=3"\
                --stdout "$LOG_PATH/${log_filename}.out" --stderr "$LOG_PATH/${log_filename}.err"\
                -w $mpi_time:00\
                $binary -- $N $grid 1

            # спим 10 секунд
            sleep 10
        done
    done
done