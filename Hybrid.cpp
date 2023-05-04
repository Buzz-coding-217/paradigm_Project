#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <pthread.h>
#include <stdio.h>
#include <fstream>
#include <mpi.h>

using namespace std;
std::fstream myfile("document.txt");

// Box Muller Transformation, it alternatively creates data.
double generate_random_number(double mean, double std)
{
    static double z0, z1;
    static bool generate_new_pair = true;

    // calculating data alternatively
    if (generate_new_pair)
    {
        // create variables from 0 to 1
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;

        // creating variables for the data
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * M_PI * u2;

        // Calculating the cos and sin of the fuction of data
        z0 = r * cos(theta);
        z1 = r * sin(theta);
        generate_new_pair = false;
        return z0 * std + mean;
    }
    else
    {
        generate_new_pair = true;
        return z1 * std + mean;
    }
}

// Creating struct for Thread Data
struct ThreadData
{
    int start_index;
    int end_index;
    double mean;
    double std;
    std::vector<double> *temperatures;
};

// Pthread function to create data
void *generate_weather_data(void *arg)
{
    ThreadData *data = static_cast<ThreadData *>(arg);
    for (int i = data->start_index; i < data->end_index; i++)
    {
        (*data->temperatures)[i] = generate_random_number(data->mean, data->std);
    }
    pthread_exit(NULL);
}

int main(int argc, char **argv)
{
    // Initialsing MPI
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Setting values for different MPI units
    double mean = 4.24;
    double std = 2.11;
    if (rank == 1)
    {
        mean = 3.64;
        std = 1.70;
    }
    else if (rank == 2)
    {
        mean = 3.1;
        std = 2.2;
    }

    // Generate random weather data using pthreads
    int num_days = 1000;
    int num_threads = 4;
    std::vector<double> temperatures(num_days);
    std::vector<double> wind_speeds(num_days);
    std::vector<double> precipitations(num_days);

    std::chrono::time_point<std::chrono::system_clock> start, ending;
    start = std::chrono::system_clock::now();

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int chunk_size = num_days / num_threads;
    for (int i = 0; i < num_threads; i++)
    {
        thread_data[i].start_index = i * chunk_size;
        thread_data[i].end_index = (i + 1) * chunk_size;
        thread_data[i].mean = mean;
        thread_data[i].std = std;
        thread_data[i].temperatures = &temperatures;
        pthread_create(&threads[i], NULL, generate_weather_data, &thread_data[i]);
    }
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    ending = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = ending - start;

    // Send the vector to process 0
    if (rank == 1)
    {
        MPI_Send(temperatures.data(), temperatures.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 0)
    {
        // Receive the vector from process 1 & 2
        MPI_Recv(wind_speeds.data(), wind_speeds.size(), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(precipitations.data(), precipitations.size(), MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &status);
        std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
        std::cout << "Temperature (Celsius), Wind Speed (km/h), Precipitation (mm/day)" << std::endl;
        for (int i = 0; i < num_days; i++)
        {
            myfile << temperatures[i] << "," << wind_speeds[i] << "," << precipitations[i] << "\n";
            std::cout << temperatures[i] << ", " << wind_speeds[i] << ", " << precipitations[i] << std::endl;
        }
        myfile.close();
    }
    else if (rank == 2)
    {
        // Receive the vector from process 1 & 2
        MPI_Send(temperatures.data(), temperatures.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}