#include <omp.h>
#include <iostream>
#include <iomanip>
#include <array>
#include <cmath>
#include <string>

bool is_prime(long long n)
{
    if (n < 2) return false;
    long long m = static_cast<long long>(sqrt(n));
    for (long long i = 2; i <= m; i++)
        if (n % i == 0) return false;
    return true;
}

/*
Variables privadas:
    Los índices de los for (i) son privados para cada hilo.
Variables shared:
    max_num es compartida entre todos los hilos.
Variables reduction:
    global_primes es una variable de reducción con la operación suma.
*/

void test_prime_schedule(long long max_num, int num_cores, char schedule_type, double t_seq)
{
    omp_set_num_threads(num_cores);
    long long global_primes = 0;
    std::string schedule_name;
    double start_time = omp_get_wtime();

    if (schedule_type == 's')
    {
        schedule_name = "static";
        #pragma omp parallel for reduction(+:global_primes) schedule(static)
        for (long long i = 2; i <= max_num; i++)
            if (is_prime(i)) global_primes++;
    }
    else if (schedule_type == 'd')
    {
        schedule_name = "dynamic";
        #pragma omp parallel for reduction(+:global_primes) schedule(dynamic)
        for (long long i = 2; i <= max_num; i++)
            if (is_prime(i)) global_primes++;
    }
    else if (schedule_type == 'g')
    {
        schedule_name = "guided";
        #pragma omp parallel for reduction(+:global_primes) schedule(guided)
        for (long long i = 2; i <= max_num; i++)
            if (is_prime(i)) global_primes++;
    }
    double end_time = omp_get_wtime();
    double t_par = end_time - start_time;

    double speedup = t_seq / t_par;
    double efficiency = (speedup / num_cores) * 100.0;
    std::cout << "Schedule type: " << schedule_name << ", Cores: " << num_cores << ", Time: " << std::fixed << std::setprecision(6) << t_par << "s, Primes: " << global_primes << ", Speedup: " << speedup << ", Efficiency: " << std::setprecision(4) << efficiency << std::endl;
}

int main()
{
    constexpr long long max_num = 1e7;
    std::cout << "Max number: " << max_num << std::endl;
    std::array<int, 4> const cores = {2, 4, 8, 16};

    double t_start = omp_get_wtime();
    long long primes_seq = 0;
    for (long long i = 2; i <= max_num; i++)
        if (is_prime(i)) primes_seq++;
    double t_end = omp_get_wtime();
    double t_seq = t_end - t_start;
    std::cout << "Sequential Time: " << std::fixed << std::setprecision(6) << (t_seq) << "s, Primes: " << primes_seq << std::endl;

    for (auto const& c : cores)
    {
        test_prime_schedule(max_num, c, 's', t_seq);
        test_prime_schedule(max_num, c, 'd', t_seq);
        test_prime_schedule(max_num, c, 'g', t_seq);
    }
}