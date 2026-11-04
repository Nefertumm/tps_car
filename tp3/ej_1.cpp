#include <iostream>
#include <vector>
#include <omp.h>


void sum_rows(std::vector<double> const& matriz, long long N, double& suma)
{
    #pragma omp for reduction(+:suma)
    for (long long i = 0; i < N; ++i)
        for (long long j = 0; j < N; ++j)
            suma += matriz[i * N + j];
}

void sum_columns(std::vector<double> const& matriz, long long N, double& suma)
{
    #pragma omp for reduction(+:suma)
    for (long long j = 0; j < N; ++j)
        for (long long i = 0; i < N; ++i)
            suma += matriz[i * N + j];
}

int main()
{
    constexpr long long N = 50000;
    std::vector<double> matrix(N * N, 1.0);

    double sum_local = 0.0;
    double t_start, t_end;
    double t_seq_filas, t_seq_cols, t_par_filas, t_par_cols;

    // Secuencial
    t_start = omp_get_wtime();
    sum_rows(matrix, N, sum_local);
    t_end = omp_get_wtime();
    t_seq_filas = t_end - t_start;
    sum_local = 0.0;

    t_start = omp_get_wtime();
    sum_columns(matrix, N, sum_local);
    t_end = omp_get_wtime();
    t_seq_cols = t_end - t_start;
    sum_local = 0.0;

    // Paralelo
    t_start = omp_get_wtime();
    #pragma omp parallel
    {
        sum_rows(matrix, N, sum_local);
    }
    t_end = omp_get_wtime();
    t_par_filas = t_end - t_start;
    sum_local = 0.0;

    t_start = omp_get_wtime();
    #pragma omp parallel
    {
        sum_columns(matrix, N, sum_local);
    }
    t_end = omp_get_wtime();
    t_par_cols = t_end - t_start;



    std::cout << "Tiempos" << std::endl;
    std::cout << "Seq Filas: " << t_seq_filas << " s | Seq Columnas: " << t_seq_cols << " s" << std::endl;
    std::cout << "Par Filas: " << t_par_filas << " s | Par Columnas: " << t_par_cols << " s" << std::endl;

    std::cout << "Speedup Filas: " << t_seq_filas / t_par_filas << std::endl;
    std::cout << "Speedup Columnas: " << t_seq_cols / t_par_cols << std::endl;

    return 0;
}