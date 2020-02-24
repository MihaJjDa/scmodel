#include <mpi.h>
#include <omp.h>

#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>

void sendrecv(int rank, int target, int to, void* adr, int size, MPI_Comm comm)
{
    if (rank != target) {
        MPI_Status status;
        int st = 100000 + rank   * 100 + 0 * 10 + to;
        int rt = 100000 + target * 100 + 0 * 10 + 1-to;
        MPI_Sendrecv_replace(adr, size, MPI_DOUBLE, target, st, target, rt, comm, &status);
    }
}

double solve (double x, double y, double z, double t) {
    return cos(x) * sin (y) * sin(z) * cos(sqrt(3) * t);
}

double phi(double x, double y, double z) {
    return cos(x) * sin (y) * sin(z);
}

int main(int argc, char* argv[]){
    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc <= 4) {
        if (rank == 0)
            printf("./a.out N Gx Gy Gz\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (atoi(argv[2])*atoi(argv[3])*atoi(argv[4]) != size) {
        if (rank == 0)
            printf("Need %d procs, have %d", atoi(argv[2])*atoi(argv[3])*atoi(argv[4]), size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(atoi(argv[1]) % atoi(argv[2])) {
        printf("Bad Gx\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(atoi(argv[1]) % atoi(argv[3])) {
        printf("Bad Gy\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(atoi(argv[1]) % atoi(argv[4])) {
        printf("Bad Gz\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    double start, stop;
    start = MPI_Wtime();

    MPI_Comm comm;
    int N = atoi(argv[1]);
    int dim[3] = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};
    int coord[3];
    int period[3] = {1,1,1};
    double lm, m, md, r, d, value;
    double x, y, z;
    double Uijk, Ui1jk, Ui2jk, Uij1k, Uij2k, Uijk1, Uijk2;
    int n, n1, n2, i, j, k, t;
    int st, rt, crd[3];
    int targetL, targetR, targetU, targetD, targetB, targetF;
    MPI_Status status;
    
    int Nx = N, Ny = N, Nz = N;
    double Lx = 2 * M_PI, Ly = 2 * M_PI, Lz = 2 * M_PI;
    int Dx = Nx / dim[0], Dy = Ny / dim[1], Dz = Nz / dim[2];
    double hx = Lx / (Nx-1), hy = Ly / (Ny-1), hz = Lz / (Nz-1);
    
    double T = 0.05;
    int frames = 20;
    double ht = T / (frames-1);
    
    std::vector<double> uleft, uright;
    uleft.resize(3*Dy*Dz);
    uright.resize(3*Dy*Dz);
    std::vector<double> uup, udown;
    uup.resize(3*Dx*Dz);
    udown.resize(3*Dx*Dz);
    std::vector<double> uforw, uback;
    uback.resize(3*Dx*Dy);
    uforw.resize(3*Dx*Dy);

    std::vector < std::vector < std::vector < std::vector <double> > > > u;
    u.resize(3);
    for (i = 0; i < 3; i++) {
        u[i].resize(Dx);
        for (j = 0; j < Dx; j++) {
            u[i][j].resize(Dy);
            for (k = 0; k < Dy; k++)
                u[i][j][k].resize(Dz);
        }
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Cart_create(MPI_COMM_WORLD, 3, dim, period, 1, &comm);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Cart_coords(comm, rank, 3, coord);
    
    crd[0] = coord[0]-1;
    crd[1] = coord[1];
    crd[2] = coord[2];
    MPI_Cart_rank(comm, crd, &targetL);            
    crd[0] = coord[0]+1;
    crd[1] = coord[1];
    crd[2] = coord[2];
    MPI_Cart_rank(comm, crd, &targetR);            
    crd[0] = coord[0];
    crd[1] = coord[1]-1;
    crd[2] = coord[2];
    MPI_Cart_rank(comm, crd, &targetU);            
    crd[0] = coord[0];
    crd[1] = coord[1]+1;
    crd[2] = coord[2];
    MPI_Cart_rank(comm, crd, &targetD);            
    crd[0] = coord[0];
    crd[1] = coord[1];
    crd[2] = coord[2]-1;
    MPI_Cart_rank(comm, crd, &targetF);            
    crd[0] = coord[0];
    crd[1] = coord[1];
    crd[2] = coord[2]+1;
    MPI_Cart_rank(comm, crd, &targetB);            
    int Ox = Dx * coord[0], Oy = Dy * coord[1], Oz = Dz * coord[2];

    if (rank == 0) {
        printf("N  = %d\n", N);
        printf("Gx = %d\n", dim[0]);
        printf("Gy = %d\n", dim[1]);
        printf("Gz = %d\n", dim[2]);
        printf("T  = %lf\n", ht * frames);
        printf("ht = %lf\n", ht);
        printf("hx = %lf\n", hx);
        printf("hy = %lf\n", hy);
        printf("hz = %lf\n", hz);
        if (ht > fmax(hx, fmax(hy, hz))) 
            printf("ht > max(hx, hy, hz)\n");
    }
    
    d = 0;
    for(t = 0; t < frames; t++){
        if(rank == 0)
            printf("Frame %d\n", t);
        
        n  =  t   %3;
        n1 = (t-1)%3;
        n2 = (t-2)%3;
        m = 0;
        #pragma omp parallel for
        for (i = 0; i < Dx; i++) 
            for (j = 0; j < Dy; j++)
                for (k = 0; k < Dz; k++) {
                    x = (Ox + i) * hx;
                    y = (Oy + j) * hy;
                    z = (Oz + k) * hz;
                    if (t == 0) {
                        value = phi(x,y,z); 
                    } else if (t == 1) {
                        Uijk = u[n1][i][j][k];
                        Ui1jk = i == 0    ? uleft[(n1*Dy+j)*Dz+k]  : u[n1][i-1][j][k];
                        Ui2jk = i == Dx-1 ? uright[(n1*Dy+j)*Dz+k] : u[n1][i+1][j][k];
                        Uij1k = j == 0    ? uup[(n1*Dx+i)*Dz+k]    : u[n1][i][j-1][k];
                        Uij2k = j == Dy-1 ? udown[(n1*Dx+i)*Dz+k]  : u[n1][i][j+1][k];
                        Uijk1 = k == 0    ? uforw[(n1*Dx+i)*Dy+j]  : u[n1][i][j][k-1];
                        Uijk2 = k == Dz-1 ? uback[(n1*Dx+i)*Dy+j]  : u[n1][i][j][k+1];
                        value = Uijk + pow(ht,2)/2 * ((Ui2jk - 2*Uijk + Ui1jk)/pow(hx,2) +
                                                      (Uij2k - 2*Uijk + Uij1k)/pow(hy,2) +
                                                      (Uijk2 - 2*Uijk + Uijk1)/pow(hz,2));
                    } else {
                        Uijk = u[n1][i][j][k];
                        Ui1jk = i == 0    ? uleft[(n1*Dy+j)*Dz+k]  : u[n1][i-1][j][k];
                        Ui2jk = i == Dx-1 ? uright[(n1*Dy+j)*Dz+k] : u[n1][i+1][j][k];
                        Uij1k = j == 0    ? uup[(n1*Dx+i)*Dz+k]    : u[n1][i][j-1][k];
                        Uij2k = j == Dy-1 ? udown[(n1*Dx+i)*Dz+k]  : u[n1][i][j+1][k];
                        Uijk1 = k == 0    ? uforw[(n1*Dx+i)*Dy+j]  : u[n1][i][j][k-1];
                        Uijk2 = k == Dz-1 ? uback[(n1*Dx+i)*Dy+j]  : u[n1][i][j][k+1];
                        value = 2*Uijk - u[n2][i][j][k] + 
                                pow(ht,2) * ((Ui2jk - 2*Uijk + Ui1jk)/pow(hx,2) +
                                             (Uij2k - 2*Uijk + Uij1k)/pow(hy,2) +
                                             (Uijk2 - 2*Uijk + Uijk1)/pow(hz,2));
                    }
        
        
                    if (((Oy+j) == 0) or ((Oy+j) == (N-1)))
                        value = 0;
                    if (((Oz+k) == 0) or ((Oz+k) == (N-1)))
                        value = 0;

                    u[n][i][j][k] = value;
                    
                    lm = fabs(u[n][i][j][k] - solve(x, y, z, t*ht));
                    #pragma omp atomic
                    m += lm;
        }

        #pragma omp parallel for
        for (j = 0; j < Dy; j++)
            for (k = 0; k < Dz; k++) {
                if (dim[0] > 1) {
                    uleft[(n*Dy+j)*Dz+k] = u[n][  0 ][j][k];
                    uright[(n*Dy+j)*Dz+k] = u[n][Dx-1][j][k];
                } else {
                    uright[(n*Dy+j)*Dz+k] = u[n][  0 ][j][k];
                    uleft[(n*Dy+j)*Dz+k] = u[n][Dx-1][j][k];
                }
            }
        #pragma omp parallel for
        for (i = 0; i < Dx; i++)
            for (k = 0; k < Dz; k++) {
                if (dim[1] > 1) {
                    uup[(n*Dx+i)*Dz+k] = u[n][i][0][k];
                    udown[(n*Dx+i)*Dz+k] = u[n][i][Dy-1][k];
                } else {
                    udown[(n*Dx+i)*Dz+k] = u[n][i][0][k];
                    uup[(n*Dx+i)*Dz+k] = u[n][i][Dy-1][k];
                }
            }
        #pragma omp parallel for
        for (i = 0; i < Dx; i++)
            for (j = 0; j < Dy; j++) {
                if (dim[2] > 1) {
                    uforw[(n*Dx+i)*Dy+j] = u[n][i][j][0];
                    uback[(n*Dx+i)*Dy+j] = u[n][i][j][Dz-1];
                } else {
                    uback[(n*Dx+i)*Dy+j] = u[n][i][j][0];
                    uforw[(n*Dx+i)*Dy+j] = u[n][i][j][Dz-1];
                }
            }
        
                    
        if (coord[0]%2 == 0) {
            sendrecv(rank, targetL, 1, &(uleft[n*Dy*Dz]), Dy*Dz, comm);
            sendrecv(rank, targetR, 0, &(uright[n*Dy*Dz]), Dy*Dz, comm);
        } else {
            sendrecv(rank, targetR, 0, &(uright[n*Dy*Dz]), Dy*Dz, comm);
            sendrecv(rank, targetL, 1, &(uleft[n*Dy*Dz]), Dy*Dz, comm);
        }
        if (coord[1]%2 == 0) {
            sendrecv(rank, targetU, 1, &(uup[n*Dx*Dz]), Dx*Dz, comm);
            sendrecv(rank, targetD, 0, &(udown[n*Dx*Dz]), Dx*Dz, comm);
        } else {
            sendrecv(rank, targetD, 0, &(udown[n*Dx*Dz]), Dx*Dz, comm);
            sendrecv(rank, targetU, 1, &(uup[n*Dx*Dz]), Dx*Dz, comm);
        }
        if (coord[2]%2 == 0) {
            sendrecv(rank, targetF, 1, &(uback[n*Dx*Dy]), Dx*Dy, comm);
            sendrecv(rank, targetB, 0, &(uforw[n*Dx*Dy]), Dx*Dy, comm);
        } else {
            sendrecv(rank, targetB, 0, &(uforw[n*Dx*Dy]), Dx*Dy, comm);
            sendrecv(rank, targetF, 1, &(uback[n*Dx*Dy]), Dx*Dy, comm);
        }

        r = 0;
        MPI_Reduce(&m, &r, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Barrier(comm);
        d += r;
    }
    stop = MPI_Wtime();
    stop -= start;
    MPI_Reduce(&stop, &start, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Time: %lf\n", start);
        printf("D:    %le\n", d/N/N/N/frames);
    }

    MPI_Finalize();
    return 0;
}
