#include <mpi.h>
#include <iostream>
#include <memory>
#include <vector>

void mat_mult(const double* local_A, const double* B, double* local_C, int N, int local_rows)
{
    for (int i = 0; i < local_rows; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < N; ++k)
            {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int N = 2048;

    if (N % size != 0)
    {
        if (rank == 0)
        {
            std::cerr << "El tamaño N debe ser múltiplo de la cantidad de procesos.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_rows = N / size;
    int local_elements = local_rows * N;
    int total_elements = N * N;

    std::unique_ptr<double[]> A;
    std::unique_ptr<double[]> C;
    std::unique_ptr<double[]> B(new double[total_elements]);

    std::unique_ptr<double[]> local_A(new double[local_elements]);
    std::unique_ptr<double[]> local_C(new double[local_elements]);

    if (rank == 0)
    {
        A.reset(new double[total_elements]);
        C.reset(new double[total_elements]);

        for (int i = 0; i < total_elements; ++i)
        {
            A[i] = 1.0;
            B[i] = 2.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    MPI_Bcast(B.get(), total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(A.get(), local_elements, MPI_DOUBLE, local_A.get(), local_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    mat_mult(local_A.get(), B.get(), local_C.get(), N, local_rows);

    MPI_Gather(local_C.get(), local_elements, MPI_DOUBLE, C.get(), local_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Procesadores: " << size << "\n";
        std::cout << "Matriz: " << N << "x" << N << "\n";
        std::cout << "Tiempo de ejecución: " << max_time << " segundos\n";
        std::cout << "Verificación C[0]: " << C[0] << " (Esperado: " << N * 2.0 << ")\n";
    }

    MPI_Finalize();
    return 0;
}