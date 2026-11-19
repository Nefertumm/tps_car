#include <mpi.h>
#include <iostream>

void mat_mult(const double* local_A, const double* B, double* local_C, int N, int local_rows)
{
    int local_elements = local_rows * N;
    for (int idx = 0; idx < local_elements; ++idx)
    {
        local_C[idx] = 0.0;
    }

    for (int i = 0; i < local_rows; ++i)
    {
        for (int k = 0; k < N; ++k)
        {
            // Pequeña optimización para aprovechar el caché en vez de acceder repetidamente a local_A[i * N + k]
            double a_ik = local_A[i * N + k];
            for (int j = 0; j < N; ++j)
            {
                local_C[i * N + j] += a_ik * B[k * N + j];
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

    const int num_tests = 3;
    const int sizes[num_tests] = {512, 1024, 2048};

    for (int t = 0; t < num_tests; ++t)
    {
        int N = sizes[t];

        if (N % size != 0)
        {
            if (rank == 0)
            {
                std::cerr << "El tamanio N (" << N << ") debe ser multiplo de la cantidad de procesos (" << size << ").\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int local_rows = N / size;
        int local_elements = local_rows * N;
        int total_elements = N * N;

        double* A = NULL;
        double* C = NULL;
        double* B = new double[total_elements];
        double* local_A = new double[local_elements];
        double* local_C = new double[local_elements];

        if (rank == 0)
        {
            A = new double[total_elements];
            C = new double[total_elements];

            for (int i = 0; i < total_elements; ++i)
            {
                A[i] = 1.0;
                B[i] = 2.0;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        MPI_Bcast(B, total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(A, local_elements, MPI_DOUBLE, local_A, local_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        mat_mult(local_A, B, local_C, N, local_rows);

        MPI_Gather(local_C, local_elements, MPI_DOUBLE, C, local_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double t_end = MPI_Wtime();
        double local_time = t_end - t_start;
        double max_time = 0.0;

        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::cout << "-----------------------------------\n";
            std::cout << "Procesadores: " << size << "\n";
            std::cout << "Matriz: " << N << "x" << N << "\n";
            std::cout << "Tiempo de ejecucion: " << max_time << " segundos\n";
            std::cout << "Verificacion C[0]: " << C[0] << " (Esperado: " << N * 2.0 << ")\n";
        }

        delete[] B;
        delete[] local_A;
        delete[] local_C;
        if (rank == 0)
        {
            delete[] A;
            delete[] C;
        }
    }

    MPI_Finalize();
    return 0;
}