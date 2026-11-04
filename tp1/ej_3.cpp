#include <mpi.h>
#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2 || size % 2 != 0)
    {
        if (rank == 0)
            std::cerr << "Se requieren al menos 2 procesos y un número par de procesos" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    constexpr int m_size = 1 << 24;
    constexpr int N_iters = 10;

    int half_size = size / 2;
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[m_size]);

    double total_max_time = 0.0;

    for (int k = -1; k < N_iters; k++)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        double t_start = MPI_Wtime();

        if (rank < half_size)
        {
            int dest = rank + half_size;
            MPI_Send(buffer.get(), m_size, MPI_BYTE, dest, 0, MPI_COMM_WORLD);
        }
        else
        {
            int src = rank - half_size;
            MPI_Recv(buffer.get(), m_size, MPI_BYTE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double t_end = MPI_Wtime();
        double local_time = t_end - t_start;

        double max_time = local_time;
        if (rank == 0)
        {
            for (int p = 1; p < size; p++)
            {
                double recv_time = 0.0;
                MPI_Recv(&recv_time, 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                max_time = std::max(max_time, recv_time);
            }
        }
        else
        {
            MPI_Send(&local_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }

        if (k >= 0)
            total_max_time += max_time;
    }

    if (rank == 0)
    {
        double avg_time = total_max_time / N_iters;
        double total_bytes = static_cast<double>(m_size) * half_size;

        double bisection_bw_bytes = total_bytes / avg_time;
        double bisection_bw_Mbs = bisection_bw_bytes / (1024.0 * 1024.0);
        double bisection_bw_Mbps = bisection_bw_bytes * 8.0 / 1e6;

        std::cout << "Procesadores (n): " << size << std::endl;
        std::cout << "Ancho de banda de disección: " << bisection_bw_Mbs << " MB/s (" << bisection_bw_Mbps << " Mbps)" << std::endl;
    }

    MPI_Finalize();
    return 0;
}