#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <stdint.h> // C++98 en lugar de <cstdint>

int mybcast_tree(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int vrank = (rank - root + size) % size;

    int mask = 1;
    while (mask < size)
        mask <<= 1;

    mask >>= 1;

    for (; mask > 0; mask >>= 1)
    {
        int chunk = mask << 1;

        if (vrank % chunk == 0)
        {
            int vdest = vrank + mask;
            if (vdest < size)
            {
                int dest = (vdest + root) % size;
                MPI_Send(buffer, count, datatype, dest, 0, comm);
            }
        }
        else if (vrank % chunk == mask)
        {
            int vsrc = vrank - mask;
            int src = (vsrc + root) % size;
            MPI_Recv(buffer, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
        }
    }

    return MPI_SUCCESS;
}

int mybcast_sequential(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root)
    {
        for (int i = 0; i < size; i++)
        {
            if (i != root)
                MPI_Send(buffer, count, datatype, i, 0, comm);
        }
    }
    else
    {
        MPI_Recv(buffer, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int count = 1 << 24;
    const int N_iters = 10;

    uint8_t* buffer = new uint8_t[count];

    double t_start, t_end, local_time, max_time;

    if (rank == 0)
        std::fill(buffer, buffer + count, 42);

    // Warmup previo
    mybcast_sequential(buffer, count, MPI_BYTE, 0, MPI_COMM_WORLD);
    mybcast_tree(buffer, count, MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Parte secuencial
    double total_seq_time = 0.0;
    for (int iter = 0; iter < N_iters; iter++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t_start = MPI_Wtime();
        mybcast_sequential(buffer, count, MPI_BYTE, 0, MPI_COMM_WORLD);
        t_end = MPI_Wtime();
        local_time = t_end - t_start;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            total_seq_time += max_time;
    }

    // Parte en árbol
    double total_tree_time = 0.0;
    for (int iter = 0; iter < N_iters; iter++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t_start = MPI_Wtime();
        mybcast_tree(buffer, count, MPI_BYTE, 0, MPI_COMM_WORLD);
        t_end = MPI_Wtime();
        local_time = t_end - t_start;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            total_tree_time += max_time;
    }

    if (rank == 0)
    {
        std::cout << "Procesadores: " << size << "\n";
        std::cout << "Tiempo secuencial (avg): " << (total_seq_time / N_iters) * 1000.0 << " ms\n";
        std::cout << "Tiempo en arbol (avg): " << (total_tree_time / N_iters) * 1000.0 << " ms\n";
    }

    delete[] buffer;
    MPI_Finalize();
    return 0;
}