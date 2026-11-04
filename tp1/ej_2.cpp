#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <stdint.h> // Para uint8_t

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

    // Reemplazo de constexpr por const de C clásico
    const int m_size = 1 << 24;
    const int N_iters = 10;
    
    // Puntero crudo con new
    uint8_t* buffer = new uint8_t[m_size];

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
                    MPI_Send(buffer, m_size, MPI_BYTE, j, 0, MPI_COMM_WORLD);
                    MPI_Recv(buffer, m_size, MPI_BYTE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else if (rank == j)
                {
                    MPI_Recv(buffer, m_size, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buffer, m_size, MPI_BYTE, i, 0, MPI_COMM_WORLD);
                }

                double t_end = MPI_Wtime();
                if (k >= 0 && rank == i)
                {
                    sum_times += (t_end - t_start) / 2.0; 
                }
            }
            if (rank == i)
            {
                double avg_comm = sum_times / N_iters;
                double bw = m_size / avg_comm / (1024.0 * 1024.0); // MB/s
                local_row[j] = bw;
            }
        }
    }

    // Puntero crudo para la matriz completa
    double* full_matrix = NULL;
    
    if (rank == 0)
    {
        full_matrix = new double[size * size];

        std::copy(local_row.begin(), local_row.end(), full_matrix); 

        for (int k = 1; k < size; k++)
        {
            MPI_Recv(&full_matrix[k * size], size, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        // En C++98/03 se usa &vector[0] en lugar de vector.data()
        MPI_Send(&local_row[0], size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

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

    // Liberación estricta de memoria antes de finalizar
    delete[] buffer;
    if (rank == 0)
    {
        delete[] full_matrix;
    }

    MPI_Finalize();
    return 0;
}