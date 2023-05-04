import csv
import pandas as pd

input_file = 'seattle_weather.csv'
output_file = 'clean-data.csv'

# Read data from input_file and write cleaned data to output_file using csv library
with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:

    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    header = next(reader)
    writer.writerow(header)

    for row in reader:
        if row[5] == 'sun':
            row[5] = 1
        else:
            row[5] = 0
        writer.writerow(row)

# Read cleaned data from output_file using pandas library
df = pd.read_csv(output_file)

# Replace 0 values in 'precipitation' column with the mean value
mean_value = round(df['precipitation'].mean(), 2)
df['precipitation'] = df['precipitation'].apply(
    lambda x: mean_value if x == 0 else x)

# Calculate average temperature and select desired columns
df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2
df = df[['temp_avg', 'wind', 'precipitation', 'weather']]

# Write final cleaned data to output_file
df.to_csv(output_file, index=False)
