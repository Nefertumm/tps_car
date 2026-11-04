#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <memory>

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
    //                                        1 byte, 1 KB   , 16 KB  , 1 MB   , 16 MB  , 128 MB , 256 MB , 512 MB , 1 GB
    std::array<int, 9> const message_sizes = {1 << 0, 1 << 10, 1 << 14, 1 << 20, 1 << 24, 1 << 27, 1 << 28, 1 << 29, 1 << 30};
    constexpr int N = 20;
    double latency, bandwidth;

    std::unique_ptr<uint8_t[]> buffer;
    // Básicamente lo hice así porque previamente lo implementé con std::vector<int>, pero tenía un problema con el overhead de resize, y utilizando reserve no
    // lo solucionaba del todo bien porque técnicamente si después haces vector.data() y lo pasas a MPI, MPI podría usar memoria out of bounds...
    // En este ejemplo puntual no importa demasiado porque se le pasa mugre nada más al buffer para hacer benchmarking, pero es importante tenerlo en cuenta y no lo quería hacer
    // mal de un arranque
    if (rank == 0 || rank == 1)
        buffer.reset(new uint8_t[message_sizes.back()]);

    for (int const &m_size : message_sizes)
    {
        double sum_times = 0.0;
        for (int i = -1; i < N; i++)
        {
            double t_start = MPI_Wtime();
            if (rank == 0) {
                MPI_Send(buffer.get(), m_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer.get(), m_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else if (rank == 1) {
                MPI_Recv(buffer.get(), m_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.get(), m_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
            double t_end = MPI_Wtime();

            double t_comm = (t_end - t_start) / 2.0; // divido por 2 porque es ida y vuelta...
            if (i >= 0 && rank == 0) // skipeamos -1, warmup
                sum_times += t_comm;
        }

        if (rank == 0)
        {
            double avg_time = sum_times / N;
            std::cout << "Message size: " << m_size << " bytes, Average time: " << avg_time << " seconds" << std::endl;
        }

    }

    MPI_Finalize();
    return 0;
}
