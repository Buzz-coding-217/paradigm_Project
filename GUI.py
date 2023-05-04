import numpy as np
import csv
from re import X
import time
import sys
import numpy as np
import csv
from re import X
import time
import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QGridLayout

class MainWindow(QWidget):
    def _init_(self):
        super()._init_()

        # Setting up the window
        self.setWindowTitle("Logistic Regression Model")
        self.setGeometry(300, 200, 1600, 600)  # Modified geometry to accommodate 4 graphs
        self.setStyleSheet("background-color: #f8f9fa;")

        # Creating layout for the window
        
        self.layout = QGridLayout()
        self.layout.setContentsMargins(50, 50, 50, 50)
        self.layout.setSpacing(30)
        self.setLayout(self.layout)

        # Adding start button to the layout
        self.start_btn = QPushButton("Start")
        self.start_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.start_btn.setStyleSheet("background-color: #007bff; color: white; border: none; padding: 10px 20px; font-size: 18px;")
        self.start_btn.clicked.connect(self.run_model)
        self.layout.addWidget(self.start_btn, 0, 0, alignment=Qt.AlignCenter)

        # Adding card to the layout
        self.card_layout = QVBoxLayout()
        self.card_layout.setContentsMargins(40, 40, 40, 40)
        self.card_layout.setSpacing(20)

        self.theta_label = QLabel("Theta:")
        self.theta_label.setFont(QFont('Arial', 16))
        self.card_layout.addWidget(self.theta_label)

        self.theta_values_layout = QHBoxLayout()
        self.theta_values_layout.setContentsMargins(0, 0, 0, 0)
        self.theta_values_layout.setSpacing(10)

        self.card_layout.addLayout(self.theta_values_layout)

        self.slope_label = QLabel("Slope:")
        self.slope_label.setFont(QFont('Arial', 16))
        self.card_layout.addWidget(self.slope_label)
        self.slope_value = QLabel()
        self.slope_value.setFont(QFont('Arial', 12))
        self.card_layout.addWidget(self.slope_value)

        self.accuracy_label = QLabel("Accuracy:")
        self.accuracy_label.setFont(QFont('Arial', 16))
        self.card_layout.addWidget(self.accuracy_label)

        self.accuracy_value = QLabel()
        self.accuracy_value.setFont(QFont('Arial', 12))
        self.card_layout.addWidget(self.accuracy_value)

        self.layout.addLayout(self.card_layout, 0, 1)

        # Adding graphs to the layout
        self.graph_layout1 = QVBoxLayout()
        self.graph_layout1.setContentsMargins(0, 0, 0, 0)
        self.graph_layout1.setSpacing(0)

        self.graph_widget1 = pg.PlotWidget(title="Classification Plot 1")
        self.graph_widget1.setLabel('left', 'Feature 2', units='y')
        self.graph_widget1.setLabel('bottom', 'Feature 1', units='x')
        self.graph_widget1.setBackground('w')
        self.graph_layout1.addWidget(self.graph_widget1)

        self.layout.addLayout(self.graph_layout1, 0, 2)

        self.graph_layout2 = QVBoxLayout()
        self.graph_layout2.setContentsMargins(0, 0, 0, 0)
        self.graph_layout2.setSpacing(0)

        self.graph_widget2 = pg.PlotWidget(title="Classification Plot 2")
        self.graph_widget2.setLabel('left', 'Feature 2', units='y')
        self.graph_widget2.setLabel('bottom', 'Feature1', units='x')
        self.graph_widget2.setBackground('w')
        self.graph_layout2.addWidget(self.graph_widget2)
        self.layout.addLayout(self.graph_layout2, 1, 2)

        self.graph_layout3 = QVBoxLayout()
        self.graph_layout3.setContentsMargins(0, 0, 0, 0)
        self.graph_layout3.setSpacing(0)

        self.graph_widget3 = pg.PlotWidget(title="Classification Plot 3")
        self.graph_widget3.setLabel('left', 'Feature 2', units='y')
        self.graph_widget3.setLabel('bottom', 'Feature 1', units='x')
        self.graph_widget3.setBackground('w')
        self.graph_layout3.addWidget(self.graph_widget3)

        self.layout.addLayout(self.graph_layout3, 0, 3)

        self.graph_layout4 = QVBoxLayout()
        self.graph_layout4.setContentsMargins(0, 0, 0, 0)
        self.graph_layout4.setSpacing(0)

        self.graph_widget4 = pg.PlotWidget(title="Classification Plot 4")
        self.graph_widget4.setLabel('left', 'Feature 2', units='y')
        self.graph_widget4.setLabel('bottom', 'Feature 1', units='x')
        self.graph_widget4.setBackground('w')
        self.graph_layout4.addWidget(self.graph_widget4)

        self.layout.addLayout(self.graph_layout4, 1, 3)

    def run_model(self):
        # Load data
        theta, accuracy, slope = calculate()
        self.update_model(np.array(theta), slope, accuracy)


    def update_model(self, theta, slope, accuracy):
        # Update the GUI elements displaying the model parameters
        self.theta_label.setText("Theta: {}".format(theta))
        self.slope_label.setText("Slope: {}".format(slope))
        self.accuracy_label.setText("Accuracy: {}%".format(accuracy))

        # Update the internal variables representing the model parameters
        global model_theta, model_slope, model_accuracy
        model_theta = 0.7
        model_slope = -0.7
        model_accuracy = 79

        # Generate some sample data
        x = np.linspace(0, 1, 100)
        y = model_theta + model_slope * x + np.random.randn(100) * 0.1

        # Update the four plots
        self.graph_widget1.plot(x, y, pen='r', clear=True)
        self.graph_widget2.plot(x, y, pen='g', clear=True)
        self.graph_widget3.plot(x, y, pen='b', clear=True)
        self.graph_widget4.plot(x, y, pen='y', clear=True)




# Function to transform continous data to probablity whether the variable is used to predict the result
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(theta, x):
    prob = sigmoid(np.dot(x, theta))
    return prob

def calculate():
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
    for i in range(0, len(test_f)):
        temp = test_f[i][0]
        wind = test_f[i][1]
        precipitation = test_f[i][2]

        # Normalize the input features
        X_input = np.array([[temp, precipitation, wind]])
        X_input_norm = (X_input - X.mean()) / X.std()

        # Add bias column to input features
        X_input_bias = np.append(
            np.ones((X_input_norm.shape[0], 1)), X_input_norm[:, [0, 2, 1]], axis=1)

        # Predict weather
        value = 1 if predict(theta, X_input_bias) > 0.3 else 0
        if value == test_f[i][3]:
            count += 1

    # Calculating the Accuracy
    accuracy = (count / len(test_f)) * 100
    print("Accuracy - %" + str(accuracy))

    # Testing the model on data produced through Monte carlo Simulation
    for i in range(0, len(random_f)):
        temp = random_f[i][0]
        wind = random_f[i][1]
        precipitation = random_f[i][2]

        # Normalize the input features
        X_input = np.array([[temp, precipitation, wind]])
        X_input_norm = (X_input - X.mean()) / X.std()

        # Add bias column to input features
        X_input_bias = np.append(
            np.ones((X_input_norm.shape[0], 1)), X_input_norm[:, [0, 2, 1]], axis=1)

        # Predict weather
        prediction = "sunny" if predict(theta, X_input_bias) > 0.3 else "not sunny"

        print(f"The predicted weather is: {prediction}")

    end = time.time()
    execution_time = end - start
    print("Execution Time - " + str(execution_time))

    # Extract the coefficients from the trained logistic regression model
    slope = theta[1] / theta[2]

    # Return the theta vector, accuracy, and slope
    return theta, accuracy, slope

if _name_ == '_main_':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()