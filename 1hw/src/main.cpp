#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda.h>
#include "cublas_v2.h"
#include "magma_v2.h"
#include "cpp/src/linalg.h"

alglib::complex_2d_array gen_matrix(int n)
{
    alglib::complex_2d_array a; 
    alglib::cmatrixrndcond(n, 1, a);
    return a;
}

void ae_to_magma_sqr_matrix(magma_int_t n, magmaFloatComplex *A, alglib::complex_2d_array b)
{
    int i, j;
    #define A(i_, j_) A[ (i_) + (j_)*n ]
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A(i,j) = MAGMA_C_MAKE(b(i,j).x, b(i,j).y);
    #undef A
    return;
}

void magma_lu_unpack_to_ae(int n, magmaFloatComplex *A, 
                           alglib::complex_2d_array *l, alglib::complex_2d_array *u)
{
    int i, j;
    alglib::complex ae_c_one, ae_c_zero;
    ae_c_one.x = 1;
    ae_c_one.y = 0;
    ae_c_zero.x = 0;
    ae_c_zero.y = 0;
    #define A(i_, j_) A[ (i_) + (j_)*n ]
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            if (i < j) {
                (*u)(i, j).x = A(i, j).x;
                (*u)(i, j).y = A(i, j).y;
                (*l)(i, j) = ae_c_zero;
            } else if (i == j) {
                (*u)(i, j).x = A(i, j).x;
                (*u)(i, j).y = A(i, j).y;
                (*l)(i, i) = ae_c_one;
            }
            else {
                (*u)(i, j) = ae_c_zero;
                (*l)(i, j).x = A(i, j).x;
                (*l)(i, j).y = A(i, j).y;
            }
    #undef A
    return;
}

float disperancy(int n, alglib::complex_2d_array a, alglib::complex_2d_array b)
{
    int i, j;
    float max = 0, tmp;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            tmp = hypot(a(i,j).x-b(i,j).x, a(i,j).y-b(i,j).y);
            if (max < tmp)
                max = tmp;
        }
    return max;
}

void magma_pivmatr(alglib::complex_2d_array A, alglib::complex_2d_array* b, 
                   magma_int_t n, magma_int_t* ipiv)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) 
              (*b)(ipiv[i],j) = A(i,j);
}

int main(int argc, char* argv[])
{
    magma_init();
    printf("NUM_GPU %d\n", magma_num_gpus());
    if (argc < 2)
        fprintf(stdout, "Bad argc\n");
    else {
        srand(time(NULL));
        int i, n = atoi(argv[1]);
        float dis;
        bool debug = argc > 2;
        bool print = debug and n <= 3;

        alglib::complex_2d_array ae_a = gen_matrix(n);, ae_l, ae_u, ae_r, ae_rpiv; 
        alglib::complex ae_c_one, ae_c_zero;
        ae_c_one.x = 1; 
        ae_c_one.y = 0;
        ae_c_zero.x = 0; 
        ae_c_zero.y = 0;
        ae_l.setlength(n, n);
        ae_u.setlength(n, n);
        ae_r.setlength(n, n);
        ae_rpiv.setlength(n, n);
    
        if (print) printf("AE_A\n%s\n", ae_a.tostring(4).c_str());
        
        magma_setdevice(0);
        real_Double_t time;
        magma_int_t ngpu = magma_num_gpus(), m_n = n;
        magma_int_t m_n2 = m_n*m_n, nb = magma_get_cgetrf_nb(m_n,m_n);
        magma_int_t n_local, info, ldn_local;
        magma_int_t *ipiv = (magma_int_t*)malloc(m_n*sizeof(magma_int_t));
        magma_int_t *newpiv = (magma_int_t*)malloc(m_n*sizeof(magma_int_t));
        magmaFloatComplex *m_a;
        magma_cmalloc_cpu(&m_a, m_n2);
        magmaFloatComplex_ptr d_la[ngpu];
        magma_queue_t queue[ngpu];
        
        ae_to_magma_sqr_matrix(m_n, m_a, ae_a);
        
        if (print) {printf("M_A BEFORE\n"); magma_cprint(m_n, m_n, m_a, m_n);}
    
        if (debug) {
            time = magma_wtime();
            magma_cgetrf(m_n, m_n, m_a, m_n, ipiv, &info);
            time = magma_wtime() - time;
        } else {
            //основной алгоритм
            for (i = 0; i < ngpu; i++)
                magma_queue_create(i, &queue[i]);

            for (i = 0; i < ngpu; i++) {
                n_local = ((m_n/nb)/ngpu)*nb;
                if (i < (m_n/nb)%ngpu)
                n_local += nb;
                else if (i == (m_n/nb)%ngpu)
                    n_local += m_n%nb;
                ldn_local = ((n_local+31)/32)*32;
                magma_setdevice(i);
                magma_cmalloc(&d_la[i], m_n*ldn_local);
            }

            magma_csetmatrix_1D_col_bcyclic(ngpu, m_n, m_n, nb, m_a, m_n, d_la, m_n, queue);
            time = magma_sync_wtime(NULL);
            magma_cgetrf_mgpu(ngpu, m_n, m_n, d_la, m_n, ipiv, &info);
            time = magma_sync_wtime(NULL) - time;
            magma_cgetmatrix_1D_col_bcyclic(ngpu, m_n, m_n, nb, d_la, m_n, m_a, m_n, queue);
        } 
    
        if (print) {printf ("M_A AFTER\n"); magma_cprint(m_n, m_n, m_a, m_n);}
        
        magma_setdevice(0);
        printf("INFO %d\n", info);
        
        if (print) {printf("IPIV "); 
                         for (i = 0; i < m_n; i++) 
                             printf("%d ", ipiv[i]);
                         printf("\n");}

        magma_swp2pswp(MagmaNoTrans, m_n, ipiv, newpiv);

        if (print) {printf("NEWPIV "); 
                    for (i = 0; i < m_n; i++)
                        printf("%d ", newpiv[i]);
                    printf("\n");}
        
        magma_lu_unpack_to_ae(m_n, m_a, &ae_l, &ae_u);
        
        if (print) {printf("AE_A\n%s\n", ae_a.tostring(4).c_str());
                    printf("AE_L\n%s\n", ae_l.tostring(4).c_str());
                    printf("AE_U\n%s\n", ae_u.tostring(4).c_str());}
        
        alglib::cmatrixgemm(n, n, n, 
                            ae_c_one, ae_l, 0, 0, 0, ae_u, 0, 0, 0, ae_c_zero, ae_r, 0, 0);
        
        if (print) printf("AE_R\n%s\n", ae_r.tostring(4).c_str());
     
        magma_pivmatr(ae_r, &ae_rpiv, m_n, newpiv);
        
        if (print) printf("AE_R1\n%s\n", ae_rpiv.tostring(4).c_str());
        
        dis = disperancy(n, ae_a, ae_rpiv);
        printf("N %d\nNGPU %d\nDIS %f\n", n, ngpu, dis);
        printf("TIME %f\n", time);
    
        free(ipiv);
        magma_free_cpu(m_a);
        if (!debug) {
            for (i = 0; i < ngpu; i++)
                magma_free(d_la[i]);    
            for (i = 0; i < ngpu; i++)
                magma_queue_destroy(queue[i]);
        }
    }
    magma_finalize();
    return 0;
}

