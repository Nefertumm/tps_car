#include <mpi.h>
#include <iostream>
#include <memory>
#include <algorithm>

void mybcast_tree(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int mask = 1;
    while (mask < size)
        mask <<= 1;

    mask >>= 1;

    for (; mask > 0; mask >>= 1)
    {
        int chunk = mask << 1;

        if (rank % chunk == 0)
        {
            int dest = rank + mask;
            if (dest < size)
                MPI_Send(buffer, count, datatype, dest, 0, comm);
        }

        else if (rank % chunk == mask)
        {
            int src = rank - mask;
            MPI_Recv(buffer, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
        }
    }
}

void mybcast_sequential(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
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
        MPI_Recv(buffer, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int count = 1024 * 1024;
    std::unique_ptr<int[]> buffer(new int[count]);
    constexpr int N_iters = 10;

    double t_start, t_end, local_time, max_time;

    if (rank == 0)
        std::fill(buffer.get(), buffer.get() + count, 42);

    // warmup previo (no sé si está bien, pero me hace sentido siguiendo el hilo de los ejercicios que hemos hecho)
    mybcast_sequential(buffer.get(), count, MPI_INT, 0, MPI_COMM_WORLD);
    mybcast_tree(buffer.get(), count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double total_seq_time = 0.0;
    // Parte secuencial
    for (int iter = 0; iter < N_iters; iter++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t_start = MPI_Wtime();
        mybcast_sequential(buffer.get(), count, MPI_INT, 0, MPI_COMM_WORLD);
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
        mybcast_tree(buffer.get(), count, MPI_INT, 0, MPI_COMM_WORLD);
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
        std::cout << "Tiempo en árbol (avg): " << (total_tree_time / N_iters) * 1000.0 << " ms\n";
    }

    MPI_Finalize();

    return 0;
}