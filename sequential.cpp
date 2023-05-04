#include <iostream>
#include <random>
#include <vector>
#include <chrono>

// Box Muller Transformation, it alternatively creates data. 
double generate_random_number(double mean, double std) {
  static double z0, z1;
  static bool generate_new_pair = true;

    // calculating data alternatively
  if (generate_new_pair) {
    // create variables from 0 to 1
    double u1 = (double) rand() / RAND_MAX;
    double u2 = (double) rand() / RAND_MAX;

    // creating variables for the data
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    // Calculating the cos and sin of the fuction of data
    z0 = r * cos(theta);
    z1 = r * sin(theta);
    generate_new_pair = false;
    return z0 * std + mean;
  } else {
    generate_new_pair = true;
    return z1 * std + mean;
  }
}

int main()
{
    // Define the mean and standard deviation of the weather variables
    double mean_temp = 4.24;         
    double std_temp = 2.11;           
    double mean_wind_speed = 3.64;
    double std_wind_speed = 1.70;  
    double mean_precipitation = 3.1;
    double std_precipitation = 2.2; 

    // Generate random weather data
    int num_days = 1000; // number of days to simulate
    std::vector<double> temperatures(num_days);
    std::vector<double> wind_speeds(num_days);
    std::vector<double> precipitations(num_days);

    std::chrono::time_point<std::chrono::system_clock> start, ending;
    start = std::chrono::system_clock::now();
    
    for (int i = 0; i < num_days; i++)
    {
        temperatures[i] = generate_random_number(mean_temp, std_temp);
        wind_speeds[i] = generate_random_number(mean_wind_speed, std_wind_speed);
        precipitations[i] = generate_random_number(mean_precipitation, std_precipitation);
    }
    ending = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(ending - start) * 100;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " microseconds\n";

    // Print the generated weather data
    for (int i = 0; i < num_days; i++)
    {
        std::cout << "Day " << i + 1 << ": Temperature = " << temperatures[i] << " C, Wind Speed = " << wind_speeds[i] << " km/h, Precipitation = " << precipitations[i] << " mm/day" << std::endl;
    }
    return 0;
}