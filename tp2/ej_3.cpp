#include <mpi.h>
#include <iostream>
#include <cmath>
#include <memory>

bool is_prime(int n)
{
    if (n <= 1) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    int limit = std::sqrt(n);
    for (int i = 3; i <= limit; i += 2)
        if (n % i == 0) return false;
    return true;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 5)
    {
        if (rank == 0)
            std::cerr << "Este ejercicio requiere exactamente 5 procesos.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int max_num = 1e7;
    int partitions[] = {1000, 10000, 100000, 1000000, 2000000};

    std::unique_ptr<double[]> all_calc_times;
    std::unique_ptr<double[]> all_comm_times;

    if (rank == 0)
    {
        all_calc_times.reset(new double[size]);
        all_comm_times.reset(new double[size]);
        std::cout << "Analizando primos hasta " << max_num << "...\n";
    }

    // Iteramos por cada tamaño de partición para evaluarlas todas en una corrida
    for (int p = 0; p < 5; ++p)
    {
        int chunk_size = partitions[p];
        int local_primes = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start_calc = MPI_Wtime();

        for (int base = 0; base < max_num; base += size * chunk_size)
        {
            int start = base + rank * chunk_size + 1;
            int end = start + chunk_size - 1;

            if (start > max_num) break;
            if (end > max_num) end = max_num;

            for (int num = start; num <= end; ++num)
                if (is_prime(num))
                    local_primes++;
        }

        double t_end_calc = MPI_Wtime();
        double calc_time = t_end_calc - t_start_calc;

        double t_start_comm = MPI_Wtime();

        int global_primes = 0;
        MPI_Reduce(&local_primes, &global_primes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        double t_end_comm = MPI_Wtime();
        double comm_time = t_end_comm - t_start_comm;

        MPI_Gather(&calc_time, 1, MPI_DOUBLE, all_calc_times.get(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&comm_time, 1, MPI_DOUBLE, all_comm_times.get(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::cout << "\n--- Partición estática: " << chunk_size << " ---\n";
            std::cout << "Primos encontrados: " << global_primes << "\n";
            for (int i = 0; i < size; ++i)
            {
                std::cout << "P" << i << " -> Cálculo: " << all_calc_times[i]
                          << "s | Sync/Comm: " << all_comm_times[i] << "s\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}