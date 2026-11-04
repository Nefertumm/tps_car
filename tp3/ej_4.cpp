#include <omp.h>
#include <iostream>
#include <array>
#include <iomanip>

inline double f(double x)
{
    return 4.0 / (1.0 + x * x);
}

void calc_pi(long long num_steps, double step, int cores, double& time, double& pi, char scheme)
{
    double sum = 0.0;
    omp_set_num_threads(cores);
    double t_start;

    if (scheme == 'r')
    {
        t_start = omp_get_wtime();
        #pragma omp parallel for reduction(+:sum) default(none) shared(num_steps, step)
        for (long long i = 0; i < num_steps; ++i)
        {
            double x = (i + 0.5) * step;
            sum += f(x);
        }
        time = omp_get_wtime() - t_start;
    }
    else if (scheme == 'c')
    {
        t_start = omp_get_wtime();
        #pragma omp parallel default(none) shared(num_steps, step, sum)
        {
            double local_sum = 0.0;
            #pragma omp for
            for (long long i = 0; i < num_steps; ++i)
            {
                double x = (i + 0.5) * step;
                local_sum += f(x);
            }
            #pragma omp atomic
                sum += local_sum;
        }
        time = omp_get_wtime() - t_start;
    }
    else if (scheme == 'o')
    {
        t_start = omp_get_wtime();
        #pragma omp parallel for ordered default(none) shared(num_steps, step, sum)
        for (long long i = 0; i < num_steps; ++i)
        {
            double x = (i + 0.5) * step;
            #pragma omp ordered
                sum += f(x);
        }
        time = omp_get_wtime() - t_start;
    }

    pi = sum * step;
}

int main()
{
    std::array<int, 4> num_cores = {2, 4, 8, 16};
    constexpr long long num_steps = 1e9;
    double step = 1.0 / static_cast<double>(num_steps);
    double t_seq, pi_seq;
    calc_pi(num_steps, step, 1, t_seq, pi_seq, 'r');
    std::cout << "Sequential: time = " << t_seq << ", pi = " << std::setprecision(10) << pi_seq << "s" << std::endl;
    for (int cores : num_cores)
    {
        double t_par, pi_par;
        calc_pi(num_steps, step, cores, t_par, pi_par, 'r');
        double speedup = t_seq / t_par;
        double efficiency = speedup / cores * 100.0;
        std::cout << "Parallel (" << cores << " cores, reduction): time = " << t_par << "s, pi = " << std::setprecision(10) << pi_par << ", speedup = " << speedup << ", efficiency = " << std::setprecision(4) << efficiency << "%" << std::endl;
        calc_pi(num_steps, step, cores, t_par, pi_par, 'c');
        speedup = t_seq / t_par;
        efficiency = speedup / cores * 100.0;
        std::cout << "Parallel (" << cores << " cores, critical): time = " << t_par << "s, pi = " << std::setprecision(10) << pi_par << ", speedup = " << speedup << ", efficiency = " << std::setprecision(4) << efficiency << "%" << std::endl;
        calc_pi(num_steps, step, cores, t_par, pi_par, 'o');
        speedup = t_seq / t_par;
        efficiency = speedup / cores * 100.0;
        std::cout << "Parallel (" << cores << " cores, ordered): time = " << t_par << "s, pi = " << std::setprecision(10) << pi_par << ", speedup = " << speedup << ", efficiency = " << std::setprecision(4) << efficiency << "%" << std::endl;
    }
    return 0;
}