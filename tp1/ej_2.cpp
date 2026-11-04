#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2)
    {
        std::cerr << "Se requieren al menos 2 procesos" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    constexpr int m_size = 1 << 24;
    constexpr int N_iters = 10;
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[m_size]);

    std::vector<double> local_row(size, 0.0);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == j) continue;

            MPI_Barrier(MPI_COMM_WORLD);

            double sum_times = 0.0;
            for (int k = -1; k < N_iters; k++)
            {
                double t_start = MPI_Wtime();

                if (rank == i)
                {
                    MPI_Send(buffer.get(), m_size, MPI_BYTE, j, 0, MPI_COMM_WORLD);
                    MPI_Recv(buffer.get(), m_size, MPI_BYTE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else if (rank == j)
                {
                    MPI_Recv(buffer.get(), m_size, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buffer.get(), m_size, MPI_BYTE, i, 0, MPI_COMM_WORLD);
                }

                double t_end = MPI_Wtime();
                if (k >= 0 && rank == i)
                    sum_times += (t_end - t_start) / 2.0; // divido por 2 porque es ida y vuelta...
            }
            if (rank == i)
            {
                double avg_comm = sum_times / N_iters;
                double bw = m_size / avg_comm / (1024.0 * 1024.0); // MB/s
                local_row[j] = bw;
            }
        }
    }

    std::unique_ptr<double[]> full_matrix;
    if (rank == 0)
    {
        full_matrix.reset(new double[size * size]);

        std::copy(local_row.begin(), local_row.end(), full_matrix.get()); // "pegamos" la local_row de master en la matriz

        for (int k = 1; k < size; k++)
            MPI_Recv(&full_matrix[k * size], size, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
        MPI_Send(local_row.data(), size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Matriz completa de ancho de banda (MB/s):" << std::endl;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                std::cout << std::setw(10) << full_matrix[i * size + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}