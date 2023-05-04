import numpy as np
import csv
from re import X
import time
import concurrent.futures

# Function to transform continous data to probablity whether the variable is used to predict the result
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(theta, x):
    prob = sigmoid(np.dot(x, theta))
    return prob


# Creating variables
data = []
results = []
test = []
random = []

# Loading data to variables
with open('CleanData2.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)

    for row in reader:
        data.append(row[0:3])
        results.append([row[3]])


with open('test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)

    for row in reader:
        random.append([row[0], row[1], row[2]])
with open('testing.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        test.append([row[0], row[1], row[2], row[3]])

# Loading data as float values
data_f = []
for row in data:
    float_row = [float(element) for element in row]
    data_f.append(float_row)

results_f = []
for row in results:
    float_row = [int(element) for element in row]
    results_f.append(float_row)

test_f = []
for row in test:
    float_row = [float(element) for element in row]
    test_f.append(float_row)

random_f = []
for row in random:
    float_row = [float(element) for element in row]
    random_f.append(float_row)

# Converting data to np array
X = np.array(data_f)
y = np.array(results_f)
z = np.array(random_f)

# Normalize the training data
X_norm = (X - X.mean()) / X.std()

# Add bias column to training data
X_bias = np.append(np.ones((X_norm.shape[0], 1)), X_norm[:, [0, 2, 1]], axis=1)

# Initialize theta vector with zeros
theta = np.zeros((X_bias.shape[1], 1))

# Train logistic regression model using gradient descent
learning_rate = 0.1
num_iterations = 1000

# Applying the gradient descent to data
for i in range(num_iterations):
    z = np.dot(X_bias, theta)
    h = sigmoid(z)
    gradient = np.dot(X_bias.T, (h - y)) / y.size
    theta -= learning_rate * gradient

start = time.time()

# Calculating Accuracy
count = 0
# Define a function to process each chunk of data
def process_chunk(chunk):
    count = 0
    for data in chunk:
        temp, wind, precipitation, label = data

        # Normalize the input features
        X_input = np.array([[temp, precipitation, wind]])
        X_input_norm = (X_input - X.mean()) / X.std()

        # Add bias column to input features
        X_input_bias = np.append(
            np.ones((X_input_norm.shape[0], 1)), X_input_norm[:, [0, 2, 1]], axis=1)

        # Predict weather
        value = 1 if predict(theta, X_input_bias) > 0.3 else 0
        if value == label:
            count += 1

    return count

# Define the size of each chunk
chunk_size = 100

# Split the data into chunks of size chunk_size
chunks = [test_f[i:i+chunk_size] for i in range(0, len(test_f), chunk_size)]

# Process the chunks in parallel using a thread pool
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

    # Combine the results from each chunk
    count = sum(f.result() for f in futures)

# Calculating the Accuracy
accuracy = (count / len(test_f)) * 100
print("Accuracy - %" + str(accuracy))

# Testing the model on data produced through Monte carlo Simulation
def predict_weather(random_f):
    temp = random_f[0]
    wind = random_f[1]
    precipitation = random_f[2]

    # Normalize the input features
    X_input = np.array([[temp, precipitation, wind]])
    X_input_norm = (X_input - X.mean()) / X.std()

    # Add bias column to input features
    X_input_bias = np.append(
        np.ones((X_input_norm.shape[0], 1)), X_input_norm[:, [0, 2, 1]], axis=1)

    # Predict weather
    prediction = "sunny" if predict(theta, X_input_bias) > 0.3 else "not sunny"

    return prediction

if _name_ == '_main_':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(predict_weather, random_f[i]) for i in range(len(random_f))]
        for future in concurrent.futures.as_completed(futures):
            prediction = future.result()
            print(f"The predicted weather is: {prediction}")


end = time.time()
execution_time = end - start
print("Execution Time - " + str(execution_time))