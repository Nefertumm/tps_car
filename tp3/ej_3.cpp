#include <omp.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>

void mat_mult(std::vector<double> const& A, std::vector<double> const& B, std::vector<double>& C, int N)
{
    #pragma omp for
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

int main()
{
    std::array<int, 5> matrix_sizes = {128, 256, 512, 1024, 2048};
    std::array<int, 4> num_cores = {2, 4, 8, 16};

    for (int const& size : matrix_sizes)
    {
        std::vector<double> A(size * size, 1.0);
        std::vector<double> B(size * size, 1.0);
        std::vector<double> C(size * size, 0.0);

        // Secuencial
        double start_time = omp_get_wtime();
        mat_mult(A, B, C, size);
        double end_time = omp_get_wtime();
        double t_seq = end_time - start_time;
        std::cout << "Sequential: " << "Size: " << size << ", Time: " << t_seq << " seconds" << std::endl;

        // Parallel
        for (int const& cores : num_cores)
        {
            omp_set_num_threads(cores);
            std::fill(C.begin(), C.end(), 0.0);
            double start_time = omp_get_wtime();
            #pragma omp parallel default(none) shared(A, B, C, size) // es buena práctica hacerlo así en  vez de #pragma omp for default(shared)
            {
                mat_mult(A, B, C, size);
            }
            double end_time = omp_get_wtime();

            double t_par = end_time - start_time;
            double speedup = t_seq / t_par;
            std::cout << "Parallel: " << "Size: " << size << ", Cores: " << cores << ", Time: " << t_par << " seconds, Speedup: " << speedup << std::endl;
        }
    }
    return 0;
}