#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cctype>
#include <string>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long local_A_count = 0;
    long long local_total_bases = 0;
    long long chunk_size = 0;
    
    // Reemplazo de std::unique_ptr por un puntero crudo
    char* buffer = NULL;

    if (rank == 0)
    {
        std::ifstream file("./homo_sapiens_chromosome_1.fasta", std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "Error al abrir el archivo!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // skipeamos el header antes que nada.
        std::string header;
        std::getline(file, header);
        long long start_pos = file.tellg();
        file.seekg(0, std::ios::end);
        long long end_pos = file.tellg();
        long long data_size = end_pos - start_pos;
        file.seekg(start_pos, std::ios::beg);
        
        long long base_chunk = data_size / size;
        long long remainder = data_size % size;

        chunk_size = base_chunk;
        buffer = new char[chunk_size];
        file.read(buffer, chunk_size);

        for (int p = 1; p < size; p++)
        {
            long long p_chunk = base_chunk + (p == size - 1 ? remainder : 0);
            
            // Puntero temporal crudo en lugar de unique_ptr
            char* temp_buffer = new char[p_chunk];

            file.read(temp_buffer, p_chunk);
            MPI_Send(&p_chunk, 1, MPI_LONG_LONG, p, 0, MPI_COMM_WORLD);
            MPI_Send(temp_buffer, p_chunk, MPI_CHAR, p, 1, MPI_COMM_WORLD);
            
            // Liberación manual indispensable tras enviar
            delete[] temp_buffer;
        }
        file.close();
    }
    else
    {
        MPI_Recv(&chunk_size, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        buffer = new char[chunk_size];
        MPI_Recv(buffer, chunk_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Cambiado de 'long' a 'long long' para evitar overflow en archivos gigantes
    for (long long i = 0; i < chunk_size; i++)
    {
        char c = buffer[i];

        if (!std::isalpha(c))
            continue;

        char c_upper = std::toupper(c);
        if (c_upper == 'A')
            local_A_count++;
        if (c_upper != 'N')
            local_total_bases++;
    }

    long long global_A_count = local_A_count;
    long long global_total_bases = local_total_bases;

    if (rank == 0)
    {
        for (int p = 1; p < size; p++)
        {
            long long p_A_count, p_total_bases;

            MPI_Recv(&p_A_count, 1, MPI_LONG_LONG, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&p_total_bases, 1, MPI_LONG_LONG, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            global_A_count += p_A_count;
            global_total_bases += p_total_bases;
        }

        double percentage_A = (static_cast<double>(global_A_count) / global_total_bases) * 100;
        std::cout << "Utilizando " << size << " procesos." << std::endl;
        std::cout << "Bases totales: " << global_total_bases << std::endl;
        std::cout << "Bases A: " << global_A_count << std::endl;
        std::cout << "Porcentaje de bases A: " << percentage_A << "%" << std::endl;
    }
    else
    {
        MPI_Send(&local_A_count, 1, MPI_LONG_LONG, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&local_total_bases, 1, MPI_LONG_LONG, 0, 3, MPI_COMM_WORLD);
    }

    // Liberación estricta del buffer principal
    delete[] buffer;

    MPI_Finalize();
    return 0;
}