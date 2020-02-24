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

// ВАЖНО! везде слово процессор заменяется словом процесс!!!! 
// корректно говорить процесс
// приятнее (мне) говорить процессор
int main(int argc, char* argv[]){
    
    // начало распараллеливания
    MPI_Init(&argc, &argv);
    // номер процессора и число процессоров
    int rank, size;
    //присваивание номера процессора
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //присваивание числа процессоров
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // если мало аргументов, то обломайся
    if (argc <= 4) {
        if (rank == 0)
            printf("./a.out N Gx Gy Gz\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // если требуемое число процессоров не совпадает с фактическим, то обломайся
    if (atoi(argv[2])*atoi(argv[3])*atoi(argv[4]) != size) {
        if (rank == 0)
            printf("Need %d procs, have %d", atoi(argv[2])*atoi(argv[3])*atoi(argv[4]), size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // если размер сетки не делится на размер процессорной сетки, то обломайся
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

    // граничные условия
    //bool is_periodic_x = true, is_periodic_y = false, is_periodic_z = false;

    // переменная для общения между процессорами
    MPI_Comm comm;
    // получение линейной размерности сетки
    int N = atoi(argv[1]);
    // размерности параллелепипедной сетки процессоров
    int dim[3] = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};
    //координаты процессора в сетке процессоров
    int coord[3];
    // переменная для MPI_Cart_create (ниже)
    int period[3] = {1,1,1};
    // переменные для невязок и результата для сетки
    double lm, m, md, r, d, value;
    // переменные вещественных точек сетки
    double x, y, z;
    // переменные для семиточечного оператора
    double Uijk, Ui1jk, Ui2jk, Uij1k, Uij2k, Uijk1, Uijk2;
    // переменные для индексации
    int n, n1, n2, i, j, k, t;
    // переменные для синхронации
    int st, rt, crd[3];
    int targetL, targetR, targetU, targetD, targetB, targetF;
    MPI_Status status;
    
    // размеры сетки 
    int Nx = N, Ny = N, Nz = N;
    // правые границы - 2*pi, M_PI из cmath
    double Lx = 2 * M_PI, Ly = 2 * M_PI, Lz = 2 * M_PI;
    // размеры сетки на процессоре
    int Dx = Nx / dim[0], Dy = Ny / dim[1], Dz = Nz / dim[2];
    // шаги сетки 
    double hx = Lx / (Nx-1), hy = Ly / (Ny-1), hz = Lz / (Nz-1);
    
    // время
    double T = 0.05;
    // количество разбиений по времени
    int frames = 20;
    // шаг времени
    double ht = T / (frames-1);
    
    // Ox -1 1
    std::vector<double> uleft, uright;
    uleft.resize(3*Dy*Dz);
    uright.resize(3*Dy*Dz);
    // Oy -1 1
    std::vector<double> uup, udown;
    uup.resize(3*Dx*Dz);
    udown.resize(3*Dx*Dz);
    // Oz -1 1
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

    //присваивание номера процессора
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //присваивание числа процессоров
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //MPI_Cart_create(MPI_Comm comm_old, int ndims, int *dims, 
    //                int *periods, int reorder, MPI_Comm *comm_cart)
    //IN  comm_old  - родительский коммуникатор
    //IN  ndims     - число размерностей сетки процессоров
    //IN  dims      - массив размерностей сетки процессоров
    //IN  periods   - задание периодичности сетки процессоров (везде задаем - 1)
    //IN  reorder   - задание перенумерации процессоров для оптимизации (везде задаем - 1)
    //OUT comm_cart - новый коммуникатор
    // создание коммуникатора для параллелепипедной топологии
    MPI_Cart_create(MPI_COMM_WORLD, 3, dim, period, 1, &comm);
    //присваивание нового номера процессора
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Cart_coords(MPI_Comm comm, int rank, int ndims, int *coords)
    //IN  comm   - коммуникатор
    //IN  rank   - номер процессора
    //IN  ndim   - число размерностей сетки процессоров
    //OUT coords - координаты процессора в сетке
    // получение своих координат в процессорной сетке
    MPI_Cart_coords(comm, rank, 3, coord);
    // MPI_Cart_rank(MPI_Comm comm, int *coords, int *rank)
    // IN  comm   - коммуникатор
    // IN  coords - координаты искомого процессора
    // OUT rank   - ранк искомого процессора
    // Получение соседних процессоров
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
    //printf("TARGETS %d %d %d %d %d %d\n", targetL, targetR, targetU, targetD, targetF, targetB);
    // сдвиг сетки на процессоре относительно всей сетки
    int Ox = Dx * coord[0], Oy = Dy * coord[1], Oz = Dz * coord[2];
    // printf("O %d %d %d D %d %d %d\n", Ox, Oy, Oz, Dx, Dy, Dz);

    // вспомогательный вывод
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
        // для устойчивости численного метода нужно, чтобы tau < h
        if (ht > fmax(hx, fmax(hy, hz))) 
            printf("ht > max(hx, hy, hz)\n");
    }
    
    d = 0;
    // основное вычисление
    for(t = 0; t < frames; t++){
        if(rank == 0)
            printf("Frame %d\n", t);
        
        n  =  t   %3;
        n1 = (t-1)%3;
        n2 = (t-2)%3;
        //printf("ints %d %d %d\nD %d %d %d\n", n, n1, n2, Dx, Dy, Dz);
        m = 0;
        // распараллеливание только внешнего цикла, 
        // чтобы каждый процесс получал непрерывный кусок данных
        #pragma omp parallel for
        for (i = 0; i < Dx; i++) 
            for (j = 0; j < Dy; j++)
                for (k = 0; k < Dz; k++) {
                    // координаты
                    x = (Ox + i) * hx;
                    y = (Oy + j) * hy;
                    z = (Oz + k) * hz;
                    // первая итерация по первому условию
                    if (t == 0) {
                        value = phi(x,y,z); 
                    // вторая итерация по второму условию
                    }                    else if (t == 1) {
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
                    // остальные итерации по семиточечному оператору
                    } else { // if n > 1
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
        
        
                    // сохраняем граничные условия
            //        if (!is_periodic_x && ((x == 0)))
            //            value = 0;
            //        if (!is_periodic_y && ((y == 0)))
            //            value = 0;
            //        if (!is_periodic_z && ((z == 0)))
            //            value = 0;
 //                   if (((Ox+i) == 0) or ((Ox+i) == (N-1)))
   //                     value = 0;
                    if (((Oy+j) == 0) or ((Oy+j) == (N-1)))
                        value = 0;
                    if (((Oz+k) == 0) or ((Oz+k) == (N-1)))
                        value = 0;

                    // записываем полученное значение
                    u[n][i][j][k] = value;
                    
                    lm = fabs(u[n][i][j][k] - solve(x, y, z, t*ht));
                    #pragma omp atomic
                    m += lm;
        }

        // printf("Dx %d Dy %d Dz %d\n", Dx, Dy, Dz);
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
        
                    
        //MPI_Sendrecv_replace(void* buf, int count, MPI_Datatype datatype,
        //                     int dest, int sendtag, int source, int recvtag,
        //        MPI_Comm comm, MPI_Status *status)
        // INOUT buf      - адрес посылаемых и принимаемых данных
        // IN    count    - число передаваемых элементов
        // IN    datatype - тип передаваемых элементов
        // IN    dest     - номер процесса-получателя
        // IN    sendtag  - идентификатор посылаемого сообщения
        // IN    source   - номер процесса-отправителя
        // IN    recvtag  - идентификатор принимаемого сообщения
        // IN    comm     - коммуникатор
        // OUT   status   - атрибуты принятого сообщения
        // Синхронизация между процессорами путем передачи граничных значений своей сетки
        // printf("HERE %d %d %d %d\n", rank, coord[0], coord[1], coord[2]);
        if (coord[0]%2 == 0) {
            sendrecv(rank, targetL, 1, &(uleft[n*Dy*Dz]), Dy*Dz, comm);
            sendrecv(rank, targetR, 0, &(uright[n*Dy*Dz]), Dy*Dz, comm);
 /*            if (rank != targetL) {
                st = 100000 + rank    * 100 + 0 * 10 + 1;
                rt = 100000 + targetL * 100 + 0 * 10 + 0;
            //     printf("SYNC L %d to %d send %d recv %d\n", rank, targetL, st, rt);
                MPI_Sendrecv_replace(&uleft[n*Dy*Dz], Dy*Dz, MPI_DOUBLE, 
                                     targetL, st, targetL, rt, comm, &status);
            }
             if (rank != targetR) {
                st = 100000 + rank    * 100 + 0 * 10 + 0;
                rt = 100000 + targetR * 100 + 0 * 10 + 1;
              //   printf("SYNC R %d to %d send %d recv %d\n", rank, targetR, st, rt);
                MPI_Sendrecv_replace(&uright[n*Dy*Dz], Dy*Dz, MPI_DOUBLE, 
                                     targetR, st, targetR, rt, comm, &status);
            }*/
        } else {
            sendrecv(rank, targetR, 0, &(uright[n*Dy*Dz]), Dy*Dz, comm);
            sendrecv(rank, targetL, 1, &(uleft[n*Dy*Dz]), Dy*Dz, comm);
/*            if (rank != targetR) {
                st = 100000 + rank    * 100 + 0 * 10 + 0;
                rt = 100000 + targetR * 100 + 0 * 10 + 1;
 //                printf("SYNC R %d to %d send %d recv %d\n", rank, targetR, st, rt);
                MPI_Sendrecv_replace(&uright[n*Dy*Dz], Dy*Dz, MPI_DOUBLE, 
                                     targetR, st, targetR, rt, comm, &status);
            }
            if (rank != targetL) {
                st = 100000 + rank    * 100 + 0 * 10 + 1;
                rt = 100000 + targetL * 100 + 0 * 10 + 0;
   //             printf("SYNC L %d to %d send %d recv %d\n", rank, targetL, st, rt);
                MPI_Sendrecv_replace(&uleft[n*Dy*Dz], Dy*Dz, MPI_DOUBLE, 
                                     targetL, st, targetL, rt, comm, &status);
            }*/
        }
        if (coord[1]%2 == 0) {
            sendrecv(rank, targetU, 1, &(uup[n*Dx*Dz]), Dx*Dz, comm);
            sendrecv(rank, targetD, 0, &(udown[n*Dx*Dz]), Dx*Dz, comm);
  /*          if (rank != targetU) {
                st = 100000 + rank    * 100 + 1 * 10 + 1;
                rt = 100000 + targetU * 100 + 1 * 10 + 0;
                // printf("SYNC U %d to %d send %d recv %d\n", rank, targetU, st, rt);
                MPI_Sendrecv_replace(&(uup[n*Dx*Dz]), Dx*Dz, MPI_DOUBLE, 
                                     targetU, st, targetU, rt, comm, &status);
            }
            if (rank != targetD) {
                st = 100000 + rank    * 100 + 1 * 10 + 0;
                rt = 100000 + targetD * 100 + 1 * 10 + 1;
                // printf("SYNC D %d to %d send %d recv %d\n", rank, targetD, st, rt);
                MPI_Sendrecv_replace(&(udown[n*Dx*Dz]), Dx*Dz, MPI_DOUBLE, 
                                     targetD, st, targetD, rt, comm, &status);
            }*/
        } else {
            sendrecv(rank, targetD, 0, &(udown[n*Dx*Dz]), Dx*Dz, comm);
            sendrecv(rank, targetU, 1, &(uup[n*Dx*Dz]), Dx*Dz, comm);
            /* if (rank != targetD) {
                st = 100000 + rank    * 100 + 1 * 10 + 0;
                rt = 100000 + targetD * 100 + 1 * 10 + 1;
                // printf("SYNC D %d to %d send %d recv %d\n", rank, targetD, st, rt);
                MPI_Sendrecv_replace(&(udown[n*Dx*Dz]), Dx*Dz, MPI_DOUBLE, 
                                     targetD, st, targetD, rt, comm, &status);
            }
            if (rank != targetU) {
                st = 100000 + rank    * 100 + 1 * 10 + 1;
                rt = 100000 + targetU * 100 + 1 * 10 + 0;
                // printf("SYNC U %d to %d send %d recv %d\n", rank, targetU, st, rt);
                MPI_Sendrecv_replace(&(uup[n*Dx*Dz]), Dx*Dz, MPI_DOUBLE, 
                                     targetU, st, targetU, rt, comm, &status);
            }*/
        }
        if (coord[2]%2 == 0) {
            sendrecv(rank, targetF, 1, &(uback[n*Dx*Dy]), Dx*Dy, comm);
            sendrecv(rank, targetB, 0, &(uforw[n*Dx*Dy]), Dx*Dy, comm);
  /*           if (rank != targetF) {
                st = 100000 + rank    * 100 + 2 * 10 + 0;
                rt = 100000 + targetF * 100 + 2 * 10 + 1;
                // printf("SYNC B %d to %d send %d recv %d\n", rank, targetB, st, rt);
                MPI_Sendrecv_replace(&(uforw[n*Dx*Dy]), Dx*Dy, MPI_DOUBLE, 
                                     targetF, st, targetF, rt, comm, &status);
            }
            if (rank != targetB) {
                st = 100000 + rank    * 100 + 2 * 10 + 1;
                rt = 100000 + targetB * 100 + 2 * 10 + 0;
                // printf("SYNC F %d to %d send %d recv %d\n", rank, targetF, st, rt);
                MPI_Sendrecv_replace(&(uback[n*Dx*Dy]), Dx*Dy, MPI_DOUBLE, 
                                     targetB, st, targetB, rt, comm, &status);
            }*/
        } else {
            sendrecv(rank, targetB, 0, &(uforw[n*Dx*Dy]), Dx*Dy, comm);
            sendrecv(rank, targetF, 1, &(uback[n*Dx*Dy]), Dx*Dy, comm);
   /*         if (rank != targetB) {
                st = 100000 + rank    * 100 + 2 * 10 + 1;
                rt = 100000 + targetB * 100 + 2 * 10 + 0;
                // printf("SYNC F %d to %d send %d recv %d\n", rank, targetF, st, rt);
                MPI_Sendrecv_replace(&(uback[n*Dx*Dy]), Dx*Dy, MPI_DOUBLE, 
                                     targetB, st, targetB, rt, comm, &status);
            }
            if (rank != targetF) {
                st = 100000 + rank    * 100 + 2 * 10 + 0;
                rt = 100000 + targetF * 100 + 2 * 10 + 1;
                // printf("SYNC B %d to %d send %d recv %d\n", rank, targetB, st, rt);
                MPI_Sendrecv_replace(&(uforw[n*Dx*Dy]), Dx*Dy, MPI_DOUBLE, 
                                     targetF, st, targetF, rt, comm, &status);
            }*/
        }

        r = 0;
        // int MPI_Reduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, 
        //                MPI_Op op, int root, MPI_Comm comm)
        // IN  sendbuf  - адрес пересылаемых данных;
        // OUT recvbuf  - адрес результирующих данных
        // IN  count    - число элементов
        // IN  datatype - тип элементов
        // IN  op       - операция, по которой выполняется редукция
        // IN  root     - номер процесса-получателя результата
        // IN  comm     - коммуникатор
        // Поиск максимальной невязки и отправка на нулевой процессор
        MPI_Reduce(&m, &r, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
 //       printf("D: %d %d %d %d %le\n", rank, Ox+imax, Oy+jmax, Oz+kmax, m);
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
