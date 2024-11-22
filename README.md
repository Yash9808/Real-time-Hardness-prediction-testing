Real-Time Hardness Prediction System - README
Overview
This Python project implements a real-time hardness prediction system using machine learning classifiers. It utilizes sensor data to classify objects into hardness categories: Hard, Soft, or Flexible (H, S, F). The program integrates hardware interfacing through Arduino and Raspberry Pi GPIO for data acquisition, preprocessing, and prediction.

Features
Machine Learning: Supports multiple classifiers, including:
Logistic Regression
Support Vector Machines (SVM)
Decision Tree
Random Forest
k-Nearest Neighbors (k-NN)
Artificial Neural Networks (ANN)
Real-Time Data Collection:
Collects force, vibration, and position data using Arduino sensors.
Hardware Integration:
Uses Raspberry Pi GPIO for pneumatics control.
Serial communication via Arduino to retrieve sensor data.
Prediction:
Predicts hardness in real time.
Provides accuracy for each classifier.
Visualization:
Generates plots using matplotlib and seaborn for insights.
Requirements
Software
Python 3.8+
Required Libraries:
numpy
pandas
matplotlib
seaborn
scikit-learn
mpmath
RPi.GPIO
nanpy
Hardware
Arduino:
Sensors for force, vibration, and position.
Raspberry Pi:
GPIO connections for pneumatics control.
Sensors:
Analog sensors for force, vibration, and position measurement.
Setup and Installation
Hardware Setup:

Connect sensors to the Arduino.
Ensure the Arduino communicates with Raspberry Pi via /dev/ttyACM0.
Setup Raspberry Pi GPIO pins for pneumatics control:
Pin 21, Pin 4, Pin 3.
Software Installation:

Install Python and the required libraries:
bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn nanpy RPi.GPIO
Prepare Data:

Add your sensor data in a CSV file named All_data_spike_volt.csv. The file should include:
F-volt (Force Voltage)
V-volt (Vibration Voltage)
P-volt (Position Voltage)
Object (Target Label: H, S, or F)
Usage
Step 1: Data Preparation
The program loads data from All_data_spike_volt.csv.
Data is split into training and testing sets.
Features are scaled using StandardScaler.
Step 2: Training Classifiers
The program trains multiple classifiers and calculates their accuracy.
Step 3: Real-Time Prediction
Grasp the object to collect sensor data.
The Arduino collects real-time sensor data and sends it to the Raspberry Pi.
Sensor data is averaged and used for prediction.
Step 4: Display Results
Predicted outcomes for different classifiers are displayed along with their accuracies.
Example Output
plaintext
Copy code
Grasp the object to collect data for testing:
...

Ref:--> Hard: 0, Soft: 1, Flex: 2

Logistic Regression Predicted Outcome: Hard (Accuracy: 85.00%)
SVM Predicted Outcome: Hard (Accuracy: 87.00%)
Decision Tree Predicted Outcome: Soft (Accuracy: 80.00%)
Random Forest Predicted Outcome: Hard (Accuracy: 90.00%)
k-NN Predicted Outcome: Hard (Accuracy: 82.00%)
ANN Predicted Outcome: Soft (Accuracy: 88.00%)
GPIO and Pneumatics Control
The program activates the following GPIO pins to control the pneumatic components during data collection:

Pin 21
Pin 4
Pin 3
Notes
Ensure the Arduino is correctly connected to the Raspberry Pi before running the program.
Adjust GPIO pin settings according to your hardware configuration.
Update device='/dev/ttyACM0' if your Arduino is connected to a different port.
Future Enhancements
Extend predictions to include additional object categories.
Incorporate deep learning models for improved accuracy.
Add graphical visualizations for prediction results in real time.
