#include <mpi.h>
#include <omp.h>

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

/*********************************************************************************
Класс MPIGridFunc
этот класс отвечает за хранение сетки численного метода на узлах MPI

Содержимое:
    N[x,y,z] -- размер сетки численного метода
      
    D[x,y,z] -- размер куска сетки, который хранится на узле MPI
    O[x,y,z] -- смещение верхнего левого угла сетки в данном блоке

    B[x,y,z] -- координаты блока MPI
    G[x,y,z] -- размер MPI сетки

    data -- локальный кусок данных
    extern_data -- внешние данные, 6 граней:
        0 -- данные с блока слева   по OX (-1)
        1 -- данные с блока справа  по OX (+1)
        2 -- данные с блока сверху  по OY (-1)
        3 -- данные с блока снизу   по OY (+1)
        4 -- данные с блока спереди по OZ (-1)
        5 -- данные с блока сзади   по OZ (+1)
          
Методы:
    Init() -- задает параметры
    MPIGridFunc() -- конструкторы

    SyncMPI() -- синхронизирует extern_data

    Get[N,B,D,G,O]() -- возвращает параметры

    Get()           -- возвращает элемент сетки в глобальной индексации
                       или nan, если элемента нет ни в data, ни в extern_data
                       
    GetLocalIndex() -- возвращает элемент сетки в локальной индексации

    Set()           -- устанавливает элемент сетки в глобальной индексации
                  
    SetLocalIndex() -- устанавливает элемент сетки в локальной индексации
                                                 
*********************************************************************************/
/*
class MPIGridFunc{
    // размеры сетки
    int Nx, Ny, Nz;
    // размеры сетки на процессоре
    int Dx, Dy, Dz;
    // смещение сетки на процессоре
    int Ox, Oy, Oz;
    // координаты процессора
    int Bx, By, Bz;
    // размер сетки процессоров
    int Gx, Gy, Gz;
    
    // локальные данные
    std::vector<double> data;
    // рамка локальных данных
    std::vector<double> extern_data[6];

public:
    // пустой конструктор
    MPIGridFunc() {}
    MPIGridFunc(int Nx, int Ny, int Nz, int Gx, int Gy, int Gz, int Bx, int By, int Bz){
     //   this->Init(Nx,Ny,Nz, Gx,Gy,Gz, Bx,By,Bz);
   // void Init(int Nx, int Ny, int Nz, int Gx, int Gy, int Gz, int Bx, int By, int Bz){
        this->Nx = Nx;
        this->Ny = Ny;
        this->Nz = Nz;

        this->Gx = Gx;
        this->Gy = Gy;
        this->Gz = Gz;

        this->Bx = Bx;
        this->By = By;
        this->Bz = Bz;

        if((Nx % Gx)||(Ny % Gy)||(Nz % Gz))
            std::cerr << "Bad grid size" << std::endl;

        Dx = Nx / Gx;
        Dy = Ny / Gy;
        Dz = Nz / Gz;

        Ox = Dx * Bx;
        Oy = Dy * By;
        Oz = Dz * Bz;

        data.resize(Dx * Dy * Dz);

        extern_data[0].resize(Dy*Dz);
        extern_data[1].resize(Dy*Dz);

        extern_data[2].resize(Dx*Dz);
        extern_data[3].resize(Dx*Dz);
        
        extern_data[4].resize(Dx*Dy);
        extern_data[5].resize(Dx*Dy);
   // }
    }
  */  /*
    MPIGridFunc(int N, int G, int Bx, int By, int Bz){
        this->Init(N,N,N, G,G,G, Bx,By,Bz);
    }*/

    //void SyncMPI(MPI_Comm comm){
 //       this->PrepareExternData();

   // void PrepareExternData(){
      /*  for(int j = 0; j < Dy; ++j)
            for(int k = 0; k < Dz; ++k){
                int i = 0, target = (Gx > 1? 0: 1);
                extern_data[target][k + Dz*j] = data[k + Dz*(j + Dy*i)];
                i = Dx-1; target = (Gx > 1? 1: 0);
                extern_data[target][k + Dz*j] = data[k + Dz*(j + Dy*i)];
            }
        for(int i = 0; i < Dx; ++i)
            for(int k = 0; k < Dz; ++k){
                int j = 0, target = (Gy > 1? 2: 3);
                extern_data[target][k + Dz*i] = data[k + Dz*(j + Dy*i)];
                j = Dy-1; target = (Gy > 1? 3: 2);
                extern_data[target][k + Dz*i] = data[k + Dz*(j + Dy*i)];
            }
        for(int i = 0; i < Dx; ++i)
            for(int j = 0; j < Dy; ++j){
                int k = 0, target = (Gz > 1? 4: 5);
                extern_data[target][j + Dy*i] = data[k + Dz*(j + Dy*i)];
                k = Dz-1; target = (Gz > 1? 5: 4);
                extern_data[target][j + Dy*i] = data[k + Dz*(j + Dy*i)];
            }
    //}


        int crd[3];
        int my_rank;
        MPI_Status status;

        MPI_Comm_rank(comm, &my_rank);

        int target[6];
        int delta[6][3] = {
            {-1,0,0},{1,0,0},
            {0,-1,0},{0,1,0},
            {0,0,-1},{0,0,1}
        };

        for(int i = 0; i < 6; i++){
            crd[0] = Bx + delta[i][0];
            crd[1] = By + delta[i][1];
            crd[2] = Bz + delta[i][2];
            
            MPI_Cart_rank(comm,crd,&target[i]);            
        }

        for(int axis = 0; axis < 3; axis++){
            int tp = (axis == 0? Bx : (axis == 1? By : Bz)) % 2;
            for(int tmp = 0; tmp < 2; tmp++){
                tp = 1 - tp;

                int target_idx = 2 * axis + (1 - tp);

                int send_tag = my_rank * 100 + axis * 10 + tp;
                int recv_tag = target[target_idx] * 100 + axis * 10 + (1-tp);

                if(my_rank != target[target_idx]){
                    MPI_Sendrecv_replace(&extern_data[target_idx][0],extern_data[target_idx].size(),
                        MPI_DOUBLE,target[target_idx],send_tag,target[target_idx],recv_tag,
                        comm,&status);
                }
            }
        }
    }

    double GetLocalIndex(int i, int j, int k){
        if((j >= 0)&&(j<Dy)&&(k>=0)&&(k<Dz)){
            if(i == -1)
                return extern_data[0][k + Dz*j];
            if((i >= 0)&&(i < Dx))
                return data[k + Dz*(j + Dy*i)];
            if(i == Dx)
                return extern_data[1][k + Dz*j];
        }
        if((i >= 0)&&(i<Dx)&&(k>=0)&&(k<Dz)){
            if(j == -1)
                return extern_data[2][k + Dz*i];
            if(j == Dy)
                return extern_data[3][k + Dz*i];
        }
        if((i >= 0)&&(i<Dx)&&(j>=0)&&(j<Dy)){
            if(k == -1)
                return extern_data[4][j + Dy*i];
            if(k == Dz)
                return extern_data[5][j + Dy*i];
        }
        return nan("");
    }

    bool SetLocalIndex(int i, int j, int k, double v){
        if((i < 0)||(i >= Dx)||(j < 0)||(j >= Dy)||(k < 0)||(k >= Dz))
            return false;

        data[k + Dz*(j + Dy*i)] = v;
        return true;
    }
*/
/*
    int GetN(int i) {return (i == 0? Nx : (i == 1? Ny : Nz));}
    int GetB(int i) {return (i == 0? Bx : (i == 1? By : Bz));}
    int GetG(int i) {return (i == 0? Gx : (i == 1? Gy : Gz));}
    int GetD(int i) {return (i == 0? Dx : (i == 1? Dy : Dz));}
    int GetO(int i) {return (i == 0? Ox : (i == 1? Oy : Oz));}
*//*
    double Get(int i, int j, int k) {return GetLocalIndex(i - Ox, j - Oy, k - Oz);}
    bool Set(int i, int j, int k, double v) {return SetLocalIndex(i - Ox, j - Oy, k - Oz, v);}    
    */
//};
/*
void PrintMPIGridFunc(MPI_Comm comm, MPIGridFunc& u, int print_rank = -1, bool extended = false){
    bool full_print = (print_rank == -1);
    
    print_rank = (full_print? 0 : print_rank);
    int ext = int(extended);

    int rank;
    MPI_Comm_rank(comm, &rank);

    for(int i = 0 - ext; i < u.GetN(0) + ext; i++){
        if(print_rank == rank)
            std::cout << "[" << std::endl;
        for(int j = 0 - ext; j < u.GetN(1) + ext; j++){
            if(print_rank == rank)
                std::cout << "\t";
            for(int k = 0 - ext; k < u.GetN(2) + ext; k++){
                double loc_value = u.Get(i,j,k);
                loc_value = (isnan(loc_value)? -1e300: loc_value);

                double glob_value = loc_value;
                if(full_print)
                    MPI_Reduce(&loc_value,&glob_value,1,MPI_DOUBLE,MPI_MAX,0,comm);

                if(print_rank == rank){
                    if(glob_value == -1e300)
                        std::cout << "?" << "\t";
                    else
                        std::cout << std::setprecision(3) << glob_value << "\t";
                }
            }
            if(print_rank == rank)
                std::cout << std::endl;
        }
        if(print_rank == rank)
            std::cout << "]" << std::endl;
    }
}
*/
double UAnalytics(double x, double y, double z, double t){
    return sin(x) * sin (y) * cos(z) * cos(sqrt(3) * t);
}

double Phi(double x, double y, double z){
    return sin(x) * sin (y) * cos(z);
}
// ВАЖНО! везде слово процессор заменяется словом процесс!!!! 
// корректно говорить процесс
// приятнее (мне) говорить процессор
int main(int argc, char* argv[]){
    //номер процессора
    int rank;
    //число процессоров
    int size;
    //переменная для общения между процессорами
    MPI_Comm comm;

    // размерности параллелепипедной сетки процессоров
    int dim[3];
    //переменная для MPI_Cart_create (ниже)
    int period[3];
    //координаты процессора в сетке процессоров
    int coord[3];

    //начало распараллеливания
    MPI_Init(&argc, &argv);
    //присваивание номера процессора
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //присваивание числа процессоров
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //размерность кубической сетки процессоров (длина ребра)
    int Gsize;
    //получение длины ребра куба как числа, куб которого равен числу процессоров
    for(Gsize = 0; Gsize*Gsize*Gsize < size; ++Gsize);

    //все размерности сетки процессоров (одинаковые)
    dim[0]=Gsize; dim[1]=Gsize; dim[2]=Gsize; 
    //переменные для MPI_Cart_create (ниже)
    period[0]=1; period[1]=1; period[2]=1;

    // требуемое число процессоров
    int req_size = dim[0] * dim[1] * dim[2]; // требуемое число процессоров
    // если требуемое число процессоров не совпадает с фактическим то обломайся
    if(req_size != size){
        if(rank == 0)
            std::cout << "Need " << req_size << "procs, have " << size << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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

    // если нет размера сетки уравнения то обломайся
    if(argc <= 1){
        if(rank == 0)
            std::cout << "Need at least 1 argument" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //получение размерности сетки (количество элементов сетки) на одном процессоре
    int N;
    if(rank == 0){
        N = atoi(argv[1]);
        std::cout << "N = " << N << std::endl;
        // если размер сетки не делится на размер процессорной сетки, то обломайся
        if(N % Gsize){
            std::cout << N << " %% " << Gsize << " != 0" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    //int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    //INOUT buffer   - адрес начала расположения в памяти рассылаемых данных;
    //IN    count    - число посылаемых элементов
    //IN    datatype - тип посылаемых элементов
    //IN    root     - номер процесса-отправителя
    //IN    comm     - коммуникатор
    //рассылка размера сетки по процессорам
    MPI_Bcast(&N, 1, MPI_INT, 0, comm);

    // размеры сетки 
    int Nx = N, Ny = N, Nz = N;
    // правые границы отрезков куба - 2*pi, M_PI из cmath
    double Lx = 2 * M_PI, Ly = 2 * M_PI, Lz = 2 * M_PI;
    // шаги сетки 
    double hx = Lx / Nx, hy = Ly / Ny, hz = Lz / Nz;

    // граничные условия
    bool is_periodic_x = true, is_periodic_y = false, is_periodic_z = false;


    // количество разбиений по времени
    int frames = 20;
    
    // шаг времени
    /*
    // на сколько долей разбивам временной отрезок
    int T = 1;
    int fps = 200;
    double ht = 1.0 * T / fps;*/
    double ht = 0.05;

    // печать размера сетки
    // для устойчивости численного метода нужно, чтобы tau < h
    if(rank == 0) {
        std::cout << "h   = " << fmax(hx,fmax(hy,hz)) << std::endl << "tau = " << ht << std::endl;
        // если tau > h, то предупреждаем о последствиях
        if(ht > fmax(hx,fmax(hy,hz)))
            std::cout << "Warning! tau > h!" << std::endl;
    }

    int Dx = Nx / dim[0];
    int Dy = Ny / dim[1];
    int Dz = Nz / dim[2];

    int Ox = Dx * coord[0];
    int Oy = Dy * coord[1];
    int Oz = Dz * coord[2];
    // вектор сеточных функций
  //  std::vector<MPIGridFunc> u;
    double u1[frames][Dx][Dy][Dz];
    // Ox -1 1
    double uleft[frames][Dy][Dz], uright[frames][Dy][Dz];
    // Oy -1 1
    double uup[frames][Dx][Dz], udown[frames][Dx][Dz];
    // Oz -1 1
    double uback[frames][Dx][Dy], uforw[frames][Dx][Dy];

    /*    double** uframe[frames][6];
    for (int n = 0; n < frames; n++) {
//    uframe[n][0].resize(Dy*Dz);
//    uframe[n][1].resize(Dy*Dz);
        uframe[n][0] = (double**) malloc(Dy*sizeof(double*));
        for (int i = 0; i < Dy; i++)
            uframe[n][0][i] = (double*) malloc(Dz*sizeof(double));
        uframe[n][1] = (double**) malloc(Dy*sizeof(double*));
        for (int i = 0; i < Dy; i++)
            uframe[n][1][i] = (double*) malloc(Dz*sizeof(double));
    
//    uframe[n][2].resize(Dx*Dz);
//    uframe[n][3].resize(Dx*Dz);
        uframe[n][2] = (double**) malloc(Dx*sizeof(double*));
        for (int i = 0; i < Dy; i++)
            uframe[n][2][i] = (double*) malloc(Dz*sizeof(double));
        uframe[n][3] = (double**) malloc(Dx*sizeof(double*));
        for (int i = 0; i < Dy; i++)
            uframe[n][3][i] = (double*) malloc(Dz*sizeof(double));

//    uframe[n][4].resize(Dx*Dy);
//    uframe[n][5].resize(Dx*Dy);
        uframe[n][4] = (double**) malloc(Dx*sizeof(double*));
        for (int i = 0; i < Dy; i++)
            uframe[n][4][i] = (double*) malloc(Dy*sizeof(double));
        uframe[n][5] = (double**) malloc(Dx*sizeof(double*));
        for (int i = 0; i < Dy; i++)
            uframe[n][5][i] = (double*) malloc(Dy*sizeof(double));
    }    */
    


    //
    bool compute_metrics = true;

    // вспомогательный вывод
    if(rank == 0){
        std::cout << "Running" << std::endl;
        std::cout << "Num procs: " << omp_get_num_procs() << std::endl;
        std::cout << "Num threads: " << omp_get_num_threads() << std::endl;
    }
    // основное вычисление
    for(int n = 0; n < frames; n++){
        // вывод при начале итерации
        if(rank == 0){
            std::cout << "Frame " << n << std::endl;
        }

        // очередная сеточная функция
//        u.push_back(MPIGridFunc(Nx,Ny,Nz,dim[0],dim[1],dim[2],coord[0],coord[1],coord[2]));

        // загрузка размеров блока
      //  int Dx = u[n].GetD(0);
      //  int Dy = u[n].GetD(1);
      //  int Dz = u[n].GetD(2);

        // загрузка смещения блока в сетке
      //  int Ox = u[n].GetO(0);
      //  int Oy = u[n].GetO(1);
      //  int Oz = u[n].GetO(2);

//        int Gx = dim[0];
  //      int Gy = dim[1];
    //    int Gz = dim[2];

      //  int Bx = coord[0];
     //   int By = coord[1];
     //   int Bz = coord[2];
        

        double linf_metrics = 0;
        double lm = 0;


        // распараллеливание только внешнего цикла, 
        // чтобы каждый процесс получал непрерывный кусок данных
        #pragma omp parallel for
/*        for(int idx = 0; idx < Dx * Dy * Dz; ++idx){
            int k = idx % Dz,
                j = (idx / Dz) % Dy,
                i = idx / (Dz * Dy);*/
        for (int i = 0; i < Dx; i++) 
            for (int j = 0; j < Dy; j++)
                for (int k = 0; k < Dz; k++) {
            // координаты
            double x = (Ox + i) * hx, 
                   y = (Oy + j) * hy,
                   z = (Oz + k) * hz;
            // искомое значение
            double value, value1;
            // первая итерация по первому условию
            if (n == 0) {
                value = Phi(x,y,z);
                value1 = Phi(x,y,z); }
            // вторая итерация по второму условию
            else if (n == 1) {
/*                value = u[n-1].GetLocalIndex(i,j,k) + 
                        pow(ht,2)/2 * ((Phi(x+hx,y,z) - 2*Phi(x,y,z) + Phi(x-hx,y,z))/pow(hx,2)+
                                       (Phi(x,y+hy,z) - 2*Phi(x,y,z) + Phi(x,y-hy,z))/pow(hy,2)+
                                       (Phi(x,y,z+hz) - 2*Phi(x,y,z) + Phi(x,y,z-hz))/pow(hz,2));
  */              value1 = u1[n-1][i][j][k] + 
                        pow(ht,2)/2 * ((Phi(x+hx,y,z) - 2*Phi(x,y,z) + Phi(x-hx,y,z))/pow(hx,2)+
                                       (Phi(x,y+hy,z) - 2*Phi(x,y,z) + Phi(x,y-hy,z))/pow(hy,2)+ 
                                       (Phi(x,y,z+hz) - 2*Phi(x,y,z) + Phi(x,y,z-hz))/pow(hz,2));}
            // остальные итерации по семиточечному оператору
            else { // if n > 1
    /*            value = 2*u[n-1].GetLocalIndex(i,j,k) - u[n-2].GetLocalIndex(i,j,k) + 
                        pow(ht,2)*((u[n-1].GetLocalIndex(i+1,j,k) - 
                                    2*u[n-1].GetLocalIndex(i,j,k) + 
                                    u[n-1].GetLocalIndex(i-1,j,k))/pow(hx,2) +
                                   (u[n-1].GetLocalIndex(i,j+1,k) - 
                                    2*u[n-1].GetLocalIndex(i,j,k) + 
                                    u[n-1].GetLocalIndex(i,j-1,k))/pow(hy,2) +
                                   (u[n-1].GetLocalIndex(i,j,k+1) - 
                                    2*u[n-1].GetLocalIndex(i,j,k) + 
                                    u[n-1].GetLocalIndex(i,j,k-1))/pow(hz,2));
      */          double Uijk, Ui1jk, Ui2jk, Uij1k, Uij2k, Uijk1, Uijk2;
                Uijk = u1[n-1][i][j][k];

                if (i == 0)
                    Ui1jk = uleft[n-1][j][k];
                else
                    Ui1jk = u1[n-1][i-1][j][k];

                if (i == Dx-1)
                    Ui2jk = uright[n-1][j][k];
                else
                    Ui2jk = u1[n-1][i+1][j][k];

                if (j == 0)
                    Uij1k = uup[n-1][i][k];
                else
                    Uij1k = u1[n-1][i][j-1][k];

                if (j == Dy-1)
                    Uij2k = udown[n-1][i][k];
                else
                    Uij2k = u1[n-1][i][j+1][k];

                if (k == 0)
                    Uijk1 = uback[n-1][i][j];
                else
                    Uijk1 = u1[n-1][i][j][k-1];

                if (k == Dz-1)
                    Uijk2 = uforw[n-1][i][j];
                else
                    Uijk2 = u1[n-1][i][j][k+1];

                value1 = 2*Uijk - u1[n-2][i][j][k] + 
                        pow(ht,2)*((Ui2jk - 2*Uijk + Ui1jk)/pow(hx,2) +
                                   (Uij2k - 2*Uijk + Uij1k)/pow(hy,2) +
                                   (Uijk2 - 2*Uijk + Uijk1)/pow(hz,2));}

            // сохраняем граничные условия
            if(!is_periodic_x && ((x == 0)||(x == Lx))){
                value = 0; value1 = 0;}
            if(!is_periodic_y && ((y == 0)||(y == Ly))){
                value = 0; value1 = 0;}
            if(!is_periodic_z && ((z == 0)||(z == Lz))){
                value = 0; value1 = 0;}

            // записываем полученное значение
        //    u[n].SetLocalIndex(i,j,k,value);
            u1[n][i][j][k] = value1;


            if(compute_metrics){
          //      linf_metrics = fmax(linf_metrics, 
            //                        fabs(u[n].GetLocalIndex(i,j,k) - UAnalytics(x,y,z,n*ht)));
                lm = fmax(lm, 
                          fabs(u1[n][i][j][k] - UAnalytics(x,y,z,n*ht)));
            }
        }

//        u[n].SyncMPI(comm);
   // void SyncMPI(MPI_Comm comm){
 //       this->PrepareExternData();

   // void PrepareExternData(){
        for(int j = 0; j < Dy; ++j)
            for(int k = 0; k < Dz; ++k){
/*                int i = 0, target = (dim[0] > 1? 0: 1);
                extern_data[target][k + Dz*j] = data[k + Dz*(j + Dy*i)];
                i = Dx-1; target = (dim[0] > 1? 1: 0);
                extern_data[target][k + Dz*j] = data[k + Dz*(j + Dy*i)];*/
                if (dim[0] > 1) {
                    uleft[n][j][k] = u1[n][0][j][k];
                    uright[n][j][k] = u1[n][Dx-1][j][k];
                } else {
                    uright[n][j][k] = u1[n][0][j][k];
                    uleft[n][j][k] = u1[n][Dx-1][j][k];
                }
            }
        for(int i = 0; i < Dx; ++i)
            for(int k = 0; k < Dz; ++k){
 /*               int j = 0, target = (dim[1] > 1? 2: 3);
                extern_data[target][k + Dz*i] = data[k + Dz*(j + Dy*i)];
                j = Dy-1; target = (dim[1] > 1? 3: 2);
                extern_data[target][k + Dz*i] = data[k + Dz*(j + Dy*i)];*/
                if (dim[1] > 1) {
                    uup[n][i][k] = u1[n][i][0][k];
                    udown[n][i][k] = u1[n][i][Dy-1][k];
                } else {
                    udown[n][i][k] = u1[n][i][0][k];
                    uup[n][i][k] = u1[n][i][Dy-1][k];
                }
            }
        for(int i = 0; i < Dx; ++i)
            for(int j = 0; j < Dy; ++j){
/*                int k = 0, target = (dim[2] > 1? 4: 5);
                extern_data[target][j + Dy*i] = data[k + Dz*(j + Dy*i)];
                k = Dz-1; target = (dim[2] > 1? 5: 4);
                extern_data[target][j + Dy*i] = data[k + Dz*(j + Dy*i)];*/
                if (dim[2] > 1) {
                    uback[n][i][j] = u1[n][i][j][0];
                    uforw[n][i][j] = u1[n][i][j][Dz-1];
                } else {
                    uforw[n][i][j] = u1[n][i][j][0];
                    uback[n][i][j] = u1[n][i][j][Dz-1];
                }
            }
    //}

        int send_tag, recv_tag;
        int crd[3];
        MPI_Status status;
        int targetL, targetR, targetU, targetD, targetB, targetF;
 /*       int target[6], 
        int delta[6][3] = {
            {-1,0,0},{1,0,0},
            {0,-1,0},{0,1,0},
            {0,0,-1},{0,0,1}
        };

        for(int i = 0; i < 6; i++){
            crd[0] = coord[0] + delta[i][0];
            crd[1] = coord[1] + delta[i][1];
            crd[2] = coord[2] + delta[i][2];
            // MPI_Cart_rank(MPI_Comm comm, int *coords, int *rank)
            // IN  comm   - коммуникатор
            // IN  coords - координаты искомого процессора
            // OUT rank   - ранк искомого процессора
            MPI_Cart_rank(comm,crd,&target[i]);            
        }*/
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
        MPI_Cart_rank(comm, crd, &targetB);            
        crd[0] = coord[0];
        crd[1] = coord[1];
        crd[2] = coord[2]+1;
        MPI_Cart_rank(comm, crd, &targetF);            
/*
        for(int axis = 0; axis < 3; axis++){
            int tp = coord[axis] % 2;
            for(int tmp = 0; tmp < 2; tmp++){
                tp = 1 - tp;

                int target_idx = 2 * axis + (1 - tp);

                int send_tag = rank * 100 + axis * 10 + tp;
                int recv_tag = target[target_idx] * 100 + axis * 10 + (1-tp);

                if(rank != target[target_idx]){
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
                    MPI_Sendrecv_replace(&extern_data[target_idx][0],
                                         extern_data[target_idx].size(),
                                         MPI_DOUBLE,
                                         target[target_idx],
                                         send_tag,
                                         target[target_idx],
                                         recv_tag,
                                         comm,
                                         &status);
                }
            }
        }*/
        if (rank != targetL) {
            send_tag = rank    * 100 + 0 * 10 + 1;
            recv_tag = targetL * 100 + 0 * 10 + 0;
            MPI_Sendrecv_replace(&(uleft[n]),  Dy*Dz, MPI_DOUBLE, targetL, send_tag, 
                                                                  targetL, recv_tag, comm, &status);
        }
        if (rank != targetL) {
            send_tag = rank    * 100 + 0 * 10 + 0;
            recv_tag = targetR * 100 + 0 * 10 + 1;
            MPI_Sendrecv_replace(&(uright[n]), Dy*Dz, MPI_DOUBLE, targetR, send_tag,
                                                                  targetR, recv_tag, comm, &status);
        }
        if (rank != targetU) {
            send_tag = rank    * 100 + 1 * 10 + 1;
            recv_tag = targetU * 100 + 1 * 10 + 0;
            MPI_Sendrecv_replace(&(uup[n]),    Dx*Dz, MPI_DOUBLE, targetU, send_tag,
                                                                  targetU, recv_tag, comm, &status);
        }
        if (rank != targetD) {
            send_tag = rank    * 100 + 1 * 10 + 0;
            recv_tag = targetD * 100 + 1 * 10 + 1;
            MPI_Sendrecv_replace(&(udown[n]),  Dx*Dz, MPI_DOUBLE, targetD, send_tag,
                                                                  targetD, recv_tag, comm, &status);
        }
        if (rank != targetB) {
            send_tag = rank    * 100 + 2 * 10 + 1;
            recv_tag = targetB * 100 + 2 * 10 + 0;
            MPI_Sendrecv_replace(&(uback[n]),  Dx*Dy, MPI_DOUBLE, targetB, send_tag,
                                                                  targetB, recv_tag, comm, &status);
        }
        if (rank != targetF) {
            send_tag = rank    * 100 + 2 * 10 + 0;
            recv_tag = targetF * 100 + 2 * 10 + 1;
            MPI_Sendrecv_replace(&(uforw[n]),  Dx*Dy, MPI_DOUBLE, targetF, send_tag,
                                                                  targetF, recv_tag, comm, &status);
        }
   // }
        
        if(compute_metrics){
            double result = 0;
            double result1 = 0;
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
  //          MPI_Reduce(&linf_metrics, &result, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&lm, &result1, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if(rank == 0){
                std::cout << "L_inf = " << result << std::endl;
                std::cout << "L_inf = " << result1 << std::endl;}
        }      
        // PrintMPIGridFunc(comm,u[n],-1,false);
    }

    MPI_Finalize();
    return 0;
}
