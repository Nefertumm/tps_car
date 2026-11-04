#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdint.h> // Para uint8_t en compiladores antiguos

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2)
    {
        std::cerr << "Este programa requiere al menos 2 procesos." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Arreglo C-style (compatible con C++98/03)
    const int num_sizes = 9;
    const int message_sizes[num_sizes] = {1 << 0, 1 << 10, 1 << 14, 1 << 20, 1 << 24, 1 << 27, 1 << 28, 1 << 29, 1 << 30};
    const int N = 20;

    // Puntero crudo en lugar de std::unique_ptr
    uint8_t* buffer = NULL;

    if (rank == 0 || rank == 1)
    {
        buffer = new uint8_t[message_sizes[num_sizes - 1]];
    }

    for (int s = 0; s < num_sizes; ++s)
    {
        int m_size = message_sizes[s];
        double sum_times = 0.0;
        
        for (int i = -1; i < N; i++)
        {
            double t_start = MPI_Wtime();
            
            if (rank == 0) 
            {
                MPI_Send(buffer, m_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, m_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else if (rank == 1) 
            {
                MPI_Recv(buffer, m_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, m_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
            
            double t_end = MPI_Wtime();
            double t_comm = (t_end - t_start) / 2.0; 
            
            if (i >= 0 && rank == 0) 
            {
                sum_times += t_comm;
            }
        }

        if (rank == 0)
        {
            double avg_time = sum_times / N;
            std::cout << "Message size: " << m_size << " bytes, Average time: " << avg_time << " seconds" << std::endl;
        }
    }

    // Liberación manual de memoria
    if (rank == 0 || rank == 1)
    {
        delete[] buffer;
    }

    MPI_Finalize();
    return 0;
}