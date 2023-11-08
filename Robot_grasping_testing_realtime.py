import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import time
import os
import RPi.GPIO as GPIO
import serial
from nanpy import ArduinoApi,SerialManager
from mpmath import *

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
import seaborn as sns
sns.set_style("whitegrid")

conn = SerialManager(device='/dev/ttyACM0' )
a = ArduinoApi(connection=conn)

force = 1
vibration = 2
position = 3
val1=0
val2=0
val3=0
conn.close()
conn.open()
data1=[]
data2=[]
data3=[]

# Load the data from the CSV file
data32 = pd.read_csv('All_data_spike_volt.csv')

# Define your features (independent variables) and target variable (what you want to predict)
X = data32[['F-volt', 'V-volt', 'P-volt']].values
y = data32['Object'].values
# Convert string labels to numeric values
label_mapping = {'H': 0, 'S': 1, 'F': 2}
y = np.array([label_mapping[label] for label in y])

# Convert string labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train different classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'k-NN': KNeighborsClassifier(n_neighbors=5),  # You can adjust k as needed
    'ANN': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

accuracies = {}

for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy * 100



#1ST PNUMATIC
GPIO.setup(21, GPIO.OUT)
GPIO.output(21, GPIO.OUT)
GPIO.output(21, GPIO.LOW)

GPIO.setup(4, GPIO.OUT)
GPIO.output(4, GPIO.OUT)
GPIO.output(4, GPIO.LOW)

GPIO.setup(3, GPIO.OUT)
GPIO.output(3, GPIO.OUT)
GPIO.output(3, GPIO.LOW)


# Create a dictionary to store the user input

# Take user input for testing
print("Grasp the object to collect data for testing:")
try:
    for i in range(1,10):
        time.sleep(1)
        f_v = ((a.analogRead(force)/ 1023.0)* (5.0)) #float(input("F-volt: "))
        data1.append(f_v)
        v_v = ((a.analogRead(vibration)/ 1023.0) * (5.0)) #float(input("V-volt: "))
        data2.append(v_v)
        p_v = ((a.analogRead(position)/ 1023.0)* (5.0)) #float(input("P-volt: "))
        data3.append(p_v)

except KeyboardInterrupt:
    conn.close()
    print("serial connection closed")
    plt.close()

GPIO.output(4, GPIO.HIGH)
GPIO.output(3, GPIO.HIGH)
GPIO.output(21, GPIO.HIGH)

# Take user input for testing
print("Enter values for testing:")
f_volt = float(np.mean(data1))
v_volt = float(np.mean(data2))
p_volt = float(np.mean(data3))

user_input = {'F-volt': f_volt, 'V-volt': v_volt, 'P-volt': p_volt}

print('Ref:--> Hard: 0, Soft: 1, FLex: 2')
# Predict the outcome for user input using the trained classifiers
predictions = {}


for name, classifier in classifiers.items():
    input_data = np.array([[user_input['F-volt'], user_input['V-volt'], user_input['P-volt']]])
    input_data_scaled = scaler.transform(input_data)
    prediction = label_encoder.inverse_transform(classifier.predict(input_data_scaled))[0]
    predictions[name] = prediction

# Display the predicted outcomes for different classifiers
for name, prediction in predictions.items():
    print(f'{name} Predicted Outcome: {prediction} (Accuracy: {accuracies[name]:.2f}%)')
