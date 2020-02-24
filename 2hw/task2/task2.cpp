#include <iostream>
#include <iomanip>
#include "mpi.h"
#include <math.h>
#include "omp.h"
#include <stdio.h>

using namespace std;

int x_real_size, y_real_size, x_size, y_size, abs_size,
	up_proc, down_proc, left_proc, right_proc,
	up_in, down_in, left_in, right_in,
	up_in_abs, down_in_abs, down_border_abs,
	proc_rank;

double *x, *y, 
	*left_shadow_send, *left_shadow_recv,
	*right_shadow_send, *right_shadow_recv,
	*up_shadow_send, *up_shadow_recv,
	*down_shadow_send, *down_shadow_recv;

double step, step_square, laplas_step_coef;

//IN: u = [x_size x y_size] - with shadow
//IN: v = [x_size x y_size] - with shadow
inline double dot_product(double *u, double *v) {
	double result = 0.0;
	#pragma omp parallel for reduction(+: result)
	for (int i = up_in_abs; i <= down_in_abs; i += y_size) {
		double *u_shifted = &u[i], *v_shifted = &v[i];
		for (int j = left_in; j <= right_in; ++j) {
			result += step_square * u_shifted[j] * v_shifted[j];
		}
	}
	return result;
}

inline double function_phi(double x, double y) {
	return exp(1 - x * x * y * y);
}

inline double function_F(double x, double y) {
	double x_square = x * x, y_square = y * y, xy_square = x_square * y_square;
	return 2 * (x_square + y_square) * (1 - 2 * xy_square) * exp(1 - xy_square);
}

inline void init_0(double *m) {
	#pragma omp parallel for
	for (int i = 0; i < abs_size; i += y_size) {
		double *m_shifted = &m[i];
		for (int j = 0; j < y_size; ++j) {
			m_shifted[j] = 0;
		}
	}
}

//IN/OUT: p = [x_size x y_size] - with shadow
void init_p(double *p) {
	if (up_proc < 0) {
		double *p_shifted = &p[y_size];
		#pragma omp parallel for
		for (int k = 1; k <= y_real_size; ++k) {
			p_shifted[k] = function_phi(x[0], y[k - 1]);
		}
	}

	if (down_proc < 0) {
		double *p_shifted = &p[down_border_abs];
		#pragma omp parallel for
		for (int k = 1; k <= y_real_size; ++k) {
			p_shifted[k] = function_phi(x[x_real_size - 1], y[k - 1]);
		}
	}

	if (left_proc < 0) {
		double *p_shifted = &p[1];
		#pragma omp parallel for
		for (int i = 1; i <= x_real_size; ++i) {
			p_shifted[i * y_size] = function_phi(x[i - 1], y[0]);
		}
	}

	if (right_proc < 0) {
		double *p_shifted = &p[y_real_size];
		#pragma omp parallel for
		for (int i = 1; i <= x_real_size; ++i) {
			p_shifted[i * y_size] = function_phi(x[i - 1], y[y_real_size - 1]);
		}
	}
}

//IN/OUT: F = [x_size x y_size] - with shadow
inline void calc_F(double *F) {
	#pragma omp parallel for
	for (int i = up_in - 1; i < down_in; ++i) {
		double *F_shifted = &F[(i + 1) * y_size];
		for (int l = left_in, j = left_in - 1; l <= right_in; ++l, ++j) {
			F_shifted[l] = function_F(x[i], y[j]);
		}
	}
}

//IN/OUT: laplas = [x_size x y_size] - with shadow
//IN: p = [x_size x y_size] - with shadow
inline void calc_minus_laplas(double *minus_laplas, double *p) {
	#pragma omp parallel for
	for (int k = up_in; k <= down_in; ++k) {
		int row_abs_coord = k * y_size;
		double *minus_laplas_shifted = &minus_laplas[row_abs_coord], *p_shifted = &p[row_abs_coord];
		for (int j = left_in; j <= right_in; ++j) {
			minus_laplas_shifted[j] 
				= laplas_step_coef * (4 * p_shifted[j] - p_shifted[j - y_size] - p_shifted[j + y_size] - p_shifted[j - 1] - p_shifted[j + 1]);
		}
	}
}

//IN/OUT: r = [x_size x y_size] - with shadow
//IN: laplas_p = [x_size x y_size] - with shadow
//IN: F = [x_size x y_size] - with shadow
inline void calc_r(double *r, double *minus_laplas_p, double *F) {
	#pragma omp parallel for
	for (int k = up_in; k <= down_in; ++k) {
		int row_abs_coord = k * y_size;
		double *r_shifted = &r[row_abs_coord], *minus_laplas_p_shifted = &minus_laplas_p[row_abs_coord], *F_shifted = &F[row_abs_coord];
		for (int j = left_in; j <= right_in; ++j) {
			r_shifted[j] = minus_laplas_p_shifted[j] - F_shifted[j];
		}
	}
}

//IN: m = [x_size x y_size] - with shadow
void create_shadows(double *m) {
	if (up_proc >= 0) {
		double *m_shifted = &m[y_size + 1];
		#pragma omp parallel for
		for (int j = 0; j < y_real_size; ++j) {
			up_shadow_send[j] = m_shifted[j];
		}
	}

	if (down_proc >= 0) {
		double *m_shifted = &m[down_border_abs + 1];
		#pragma omp parallel for
		for (int j = 0; j < y_real_size; ++j) {
			down_shadow_send[j] = m_shifted[j];
		}
	}

	if (left_proc >= 0) {
		double *m_shifted = &m[1];
		#pragma omp parallel for
		for (int i = 0; i < x_real_size; ++i) {
			left_shadow_send[i] = m_shifted[(i + 1) * y_size];
		}
	}

	if (right_proc >= 0) {
		double *m_shifted = &m[y_real_size];
		#pragma omp parallel for
		for (int i = 0; i < x_real_size; ++i) {
			right_shadow_send[i] = m_shifted[(i + 1) * y_size];
		}
	}
}

//IN/OUT: m = [x_size x y_size] - with shadow
void copy_shadows(double *m) {
	if (up_proc >= 0) {
		double *m_shifted = &m[1];
		#pragma omp parallel for
		for (int j = 0; j < y_real_size; ++j) {
			m_shifted[j] = up_shadow_recv[j];
		}
	}

	if (down_proc >= 0) {
		double *m_shifted = &m[(x_real_size + 1) * y_size + 1];
		#pragma omp parallel for
		for (int j = 0; j < y_real_size; ++j) {
			m_shifted[j] = down_shadow_recv[j];
		}
	}

	if (left_proc >= 0) {
		#pragma omp parallel for
		for (int i = 0; i < x_real_size; ++i) {
			m[(i + 1) * y_size] = left_shadow_recv[i];
		}
	}

	if (right_proc >= 0) {
		#pragma omp parallel for
		for (int i = 0; i < x_real_size; ++i) {
			m[(i + 2) * y_size - 1] = right_shadow_recv[i];
		}
	}
}

void swap_shadows(double *m, MPI_Comm comm) {
	create_shadows(m);

	MPI_Request left_send_req = MPI_REQUEST_NULL, right_send_req = MPI_REQUEST_NULL, up_send_req = MPI_REQUEST_NULL, down_send_req = MPI_REQUEST_NULL,
		left_recv_req = MPI_REQUEST_NULL, right_recv_req = MPI_REQUEST_NULL, up_recv_req = MPI_REQUEST_NULL, down_recv_req = MPI_REQUEST_NULL;

	if (down_proc >= 0) {
		MPI_Irecv(down_shadow_recv, y_real_size, MPI_DOUBLE, down_proc, MPI_ANY_TAG, comm, &down_recv_req);
	}
	if (right_proc >= 0) {
		MPI_Irecv(right_shadow_recv, x_real_size, MPI_DOUBLE, right_proc, MPI_ANY_TAG, comm, &right_recv_req);
	}

	if (up_proc >= 0) {
		MPI_Isend(up_shadow_send, y_real_size, MPI_DOUBLE, up_proc, 0, comm, &up_send_req);
	}
	if (left_proc >= 0) {
		MPI_Isend(left_shadow_send, x_real_size, MPI_DOUBLE, left_proc, 0, comm, &left_send_req);
	}
	if (down_proc >= 0) {
		MPI_Isend(down_shadow_send, y_real_size, MPI_DOUBLE, down_proc, 0, comm, &down_send_req);
	}
	if (right_proc >= 0) {
		MPI_Isend(right_shadow_send, x_real_size, MPI_DOUBLE, right_proc, 0, comm, &right_send_req);
	}

	if (up_proc >= 0) {
		MPI_Irecv(up_shadow_recv, y_real_size, MPI_DOUBLE, up_proc, MPI_ANY_TAG, comm, &up_recv_req);
	}
	if (left_proc >= 0) {
		MPI_Irecv(left_shadow_recv, x_real_size, MPI_DOUBLE, left_proc, MPI_ANY_TAG, comm, &left_recv_req);
	}

	MPI_Request requests[] = { left_send_req, right_send_req, up_send_req, down_send_req, left_recv_req, right_recv_req, up_recv_req, down_recv_req };
	MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

	copy_shadows(m);
}

inline void calc_next_iter(double *next, double *first_m, double coef, double *second_m) {
	#pragma omp parallel for
	for (int i = y_size; i <= down_border_abs; i += y_size) {
		double *next_shifted = &next[i], *first_m_shifted = &first_m[i], *second_m_shifted = &second_m[i];
		for (int j = 1; j <= y_real_size; ++j) {
			next_shifted[j] = first_m_shifted[j] + coef * second_m_shifted[j];
		}
	}
}

inline void copy_matrix(double *src, double *dest) {
	#pragma omp parallel for
	for (int i = y_size; i <= down_border_abs; i += y_size) {
		double *src_shifted = &src[i], *dest_shifted = &dest[i];
		for (int j = 1; j <= y_real_size; ++j) {
			src_shifted[j] = dest_shifted[j];
		}
	}
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int NPROC, FROM = 0, TO = 2;
	double eps_square = 1e-4 * 1e-4;
	cout << fixed;

	MPI_Comm_size(MPI_COMM_WORLD, &NPROC);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	int N = 100;
	if (argc > 1) {
		sscanf(argv[1], "%d", &N);
	}
	int	log2n = log2(NPROC);
	step = (double)(TO - FROM) / N, step_square = step * step, laplas_step_coef = 1 / step_square;
	++N;

	int proc_per_x = log2n % 2 == 0 ? sqrt(NPROC) : sqrt(NPROC / 2), proc_per_y = log2n % 2 == 0 ? proc_per_x : 2 * proc_per_x;

	int periods[] = { 0, 0 }, dims[] = { proc_per_x, proc_per_y }, coords[2];
	MPI_Comm comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm);
	MPI_Comm_rank(comm, &proc_rank);
	MPI_Cart_coords(comm, proc_rank, 2, coords);
	MPI_Cart_shift(comm, 0, 1, &up_proc, &down_proc);
	MPI_Cart_shift(comm, 1, 1, &left_proc, &right_proc);

	int points_per_proc_x = N / dims[0], points_per_proc_y = N / dims[1],
		start_x_abs = coords[0] * points_per_proc_x, start_y_abs = coords[1] * points_per_proc_y,
		end_x_abs = coords[0] < dims[0] - 1 ? start_x_abs + points_per_proc_x - 1 : N - 1,
		end_y_abs = coords[1] < dims[1] - 1 ? start_y_abs + points_per_proc_y - 1 : N - 1;

	x_real_size = end_x_abs - start_x_abs + 1;
	y_real_size = end_y_abs - start_y_abs + 1;
	x_size = x_real_size + 2;
	y_size = y_real_size + 2;
	abs_size = x_size * y_size;

	up_in = 1 + (up_proc < 0); up_in_abs = up_in * y_size;
	down_in = x_real_size - (down_proc < 0); down_in_abs = down_in * y_size;
	left_in = 1 + (left_proc < 0);
	right_in = y_real_size - (right_proc < 0);

	down_border_abs = x_real_size * y_size;

	MPI_Barrier(comm);
	
	x = new double[x_real_size]; y = new double[y_real_size];
	for (int k = start_x_abs, i = 0; i < x_real_size; ++k, ++i) {
		x[i] = FROM + step * k;
	}
	for (int k = start_y_abs, j = 0; j < y_real_size; ++k, ++j) {
		y[j] = FROM + step * k;
	}

	double *F = new double[abs_size], 
		*p = new double[abs_size],
		*minus_laplas = new double[abs_size],
		*r = new double[abs_size],
		*next_p = new double[abs_size],
		*diff = new double[abs_size],
		*g = new double[abs_size];

	left_shadow_send = new double[x_real_size]; left_shadow_recv = new double[x_real_size];
	right_shadow_send = new double[x_real_size]; right_shadow_recv = new double[x_real_size];
	up_shadow_send = new double[y_real_size]; up_shadow_recv = new double[y_real_size];
	down_shadow_send = new double[y_real_size]; down_shadow_recv = new double[y_real_size];
	
	double local_time = MPI_Wtime();

	calc_F(F);

	init_0(p);
	init_p(p);

	init_0(r);

	double coef, local_numerator, global_numerator, local_denumenator, global_denumenator, local_deviation, global_deviation;

	int iteration = 0;
	do {
		if (iteration > 0) {
			if (iteration == 1) {
				copy_matrix(g, r);
			}

			copy_matrix(p, next_p);
		}

		swap_shadows(p, comm);
		calc_minus_laplas(minus_laplas, p);
		
		calc_r(r, minus_laplas, F);

		swap_shadows(r, comm);
		calc_minus_laplas(minus_laplas, r);

		if (iteration == 0) {
			local_numerator = dot_product(r, r);
			local_denumenator = dot_product(minus_laplas, r);
		} else {
			local_numerator = dot_product(minus_laplas, g);

			swap_shadows(g, comm);
			calc_minus_laplas(minus_laplas, g);
			local_denumenator = dot_product(minus_laplas, g);

			MPI_Allreduce(&local_numerator, &global_numerator, 1, MPI_DOUBLE, MPI_SUM, comm);
			MPI_Allreduce(&local_denumenator, &global_denumenator, 1, MPI_DOUBLE, MPI_SUM, comm);
			coef = -global_numerator / global_denumenator;
			calc_next_iter(g, r, coef, g);

			local_numerator = dot_product(r, g);

			swap_shadows(g, comm);
			calc_minus_laplas(minus_laplas, g);
			local_denumenator = dot_product(minus_laplas, g);
		}

		MPI_Allreduce(&local_numerator, &global_numerator, 1, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(&local_denumenator, &global_denumenator, 1, MPI_DOUBLE, MPI_SUM, comm);
		coef = -global_numerator / global_denumenator;
		calc_next_iter(next_p, p, coef, iteration == 0 ? r : g);

		#pragma omp parallel for
		for (int i = y_size; i <= down_border_abs; i += y_size) {
			double *diff_shifted = &diff[i], *next_p_shifted = &next_p[i], *p_shifted = &p[i];
			for (int j = 1; j <= y_real_size; ++j) {
				diff_shifted[j] = next_p_shifted[j] - p_shifted[j];
			}
		}

		local_deviation = dot_product(diff, diff);
		MPI_Allreduce(&local_deviation, &global_deviation, 1, MPI_DOUBLE, MPI_SUM, comm);

		++iteration;
	} while (global_deviation >= eps_square);

	local_time = MPI_Wtime() - local_time;
	double global_time;
	MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	if (proc_rank == 0) {
		cout << setprecision(6) << global_time << endl;
	}

	for (int k = y_size, i = 0; i < x_real_size; k += y_size, ++i) {
		for (int l = 1, j = 0; j < y_real_size; ++l, ++j) {
			next_p[k + l] -= function_phi(x[i], y[j]);
		}
	}

	local_deviation = dot_product(next_p, next_p);
	MPI_Allreduce(&local_deviation, &global_deviation, 1, MPI_DOUBLE, MPI_SUM, comm);
	if (proc_rank == 0) {
		cout << fixed << setprecision(10) << sqrt(global_deviation) << endl;
	}
			
	MPI_Barrier(comm);

	if (N - 1 == 2000 && NPROC == 512) {
		cout << setprecision(3);
		for (int i = 0, k = y_size; i < x_real_size; ++i, k += y_size) {
			for (int j = 0, l = 1; j < y_real_size; ++j, ++l) {
				cout << x[i] << " " << y[j] << " " << p[k + l] << endl;
			}
		}
	}

	MPI_Finalize();
	return 0;
}