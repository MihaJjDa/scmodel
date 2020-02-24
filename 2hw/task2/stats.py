#!/usr/bin/python
import glob
from collections import defaultdict
from os import path

from joblib import Parallel
from joblib import delayed

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math


def process_line(line):
    return [float(x) for x in line.split()]


def get_stats(file, short):
    time = deviation = p = None
    with open(file, mode='r', encoding='utf8') as file:
        time = float(file.readline())
        deviation = float(file.readline())
        lines = file.readlines()
        if len(lines) == 2001 * 2001 and not short:
            temp = Parallel(n_jobs=-1, verbose=-10)(delayed(process_line)(line) for line in lines)
            p = defaultdict(lambda: defaultdict(float))
            for t in temp:
                p[t[0]][t[1]] = t[2]
    return [time, deviation, p]


def get_all_stats(dir):
    files = glob.glob(dir + '/results/*.txt')
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    p = None
    for file in files:
        info = path.splitext(path.basename(file))[0].split('_')
        size, nproc = [int(x) for x in info[:2]]
        omp = len(info) > 3

        file_stats = get_stats(file, p is not None)

        stats[omp][size][nproc]['time'].append(file_stats[0])
        stats[omp][size][nproc]['deviation'] = file_stats[1]

        p = p or file_stats[2]
    return stats, p

if __name__ == '__main__':
    for comp in ['lomonosov', 'bg']:
        stats, p = get_all_stats(comp)

        type_info = defaultdict(dict)
        for omp, mode_results in sorted(stats.items()):
            type_info[omp]['sizes'] = sorted(mode_results)
            type_info[omp]['nprocs'] = sorted(list(mode_results.values())[0])

        time1 = {}
        with open('%s_stats.tsv' % comp, mode='w', encoding='utf8') as file:
            for size in type_info[False]['sizes']:
                file.write('%d x %d\t%0.6f\n' % (size, size,
                                                 stats[False][size][type_info[False]['nprocs'][0]]['deviation']))
            for omp, mode_results in sorted(stats.items()):
                file.write('%s\n' % ('OMP' if omp else 'NO OMP'))
                for size in type_info[omp]['sizes']:
                    for nproc in type_info[omp]['nprocs']:
                        time = sum(mode_results[size][nproc]['time']) / len(mode_results[size][nproc]['time'])
                        if nproc == 1 and size not in time1:
                            time1[size] = time
                        file.write('%d\t%d x %d\t%0.3f\t%0.6f\n'
                                   % (nproc, size, size, time,
                                      time1[size] / time if size in time1 else 0))

        # p
        if p:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            step = 2.0 / 2000
            X = np.array(sorted(p.keys()))
            Y = np.array(sorted(p[X[0]].keys()))
            X, Y = np.meshgrid(X, Y)
            Z_P = np.array([p[x][y] for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
            surf = ax.plot_surface(X, Y, Z_P, cmap=cm.coolwarm, linewidth=0)

            ax.set_xticks(np.arange(0, 2.1, 0.5))
            ax.set_xlabel('$X$', fontsize=15)
            ax.set_yticks(np.arange(0, 2.1, 0.5))
            ax.set_ylabel('$Y$', fontsize=15)
            ax.set_zticks(np.arange(0, 3.0, 0.5))
            ax.set_zlabel('$Z$', fontsize=15)
            ax.view_init(20, 30)

            fig.colorbar(surf, shrink=0.8, aspect=15, ticks=np.arange(0, 3.0, 0.25))
            fig.savefig('p.png', dpi=600, format='png')

            # phi
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            Z_PHI = np.array([math.exp(1 - x ** 2 * y ** 2) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
            surf = ax.plot_surface(X, Y, Z_PHI, cmap=cm.coolwarm, linewidth=0)

            ax.set_xticks(np.arange(0, 2.1, 0.5))
            ax.set_xlabel('$X$', fontsize=15)
            ax.set_yticks(np.arange(0, 2.1, 0.5))
            ax.set_ylabel('$Y$', fontsize=15)
            ax.set_zticks(np.arange(0, 3.0, 0.5))
            ax.set_zlabel('$Z$', fontsize=15)
            ax.view_init(20, 30)

            fig.colorbar(surf, shrink=0.8, aspect=15, ticks=np.arange(0, 3.0, 0.25))
            fig.savefig('phi.png', dpi=600, format='png')

            # deviation
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            surf = ax.plot_surface(X, Y, Z_PHI - Z_P, cmap=cm.coolwarm, linewidth=0, antialiased=True)

            ax.set_xticks(np.arange(0, 2.1, 0.5))
            ax.set_xlabel('$X$', fontsize=15)
            ax.set_yticks(np.arange(0, 2.1, 0.5))
            ax.set_ylabel('$Y$', fontsize=15)
            ax.set_zlabel('$Z$', fontsize=15)
            ax.view_init(20, 30)

            fig.colorbar(surf, shrink=0.8, aspect=15)
            fig.savefig('deviation.png', dpi=600, format='png')