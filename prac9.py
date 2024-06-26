# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 18:48:11 2024

@author: appar
"""

import math
import random
import pandas as pd
import numpy as np
# Function to encode class labels into numeric values
def encode_class(mydata):
    classes = []
    for i in range(len(mydata)):
        if mydata[i][-1] not in classes:
            classes.append(mydata[i][-1])
    for i in range(len(classes)):
        for j in range(len(mydata)):
            if mydata[j][-1] == classes[i]:
                mydata[j][-1] = i
    return mydata
# Function to split data into training and testing sets
def splitting(mydata, ratio):
    train_num = int(len(mydata) * ratio)
    train = []
    test = list(mydata)
    while len(train) < train_num:
        index = random.randrange(len(test))
        train.append(test.pop(index))
    return train, test
# Function to group data by class
def groupUnderClass(mydata):
    data_dict = {}
    for i in range(len(mydata)):
        if mydata[i][-1] not in data_dict:
            data_dict[mydata[i][-1]] = []
            data_dict[mydata[i][-1]].append(mydata[i])
            return data_dict
# Function to calculate mean and standard deviation for a list of numbers
def MeanAndStdDev(numbers):
    avg = np.mean(numbers)
    stddev = np.std(numbers)
    return avg, stddev
    # Function to calculate mean and standard deviation for each attribute of each class
def MeanAndStdDevForClass(mydata):
    info = {}
    data_dict = groupUnderClass(mydata)
    for classValue, instances in data_dict.items():
        info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*instances)]
    return info
# Function to calculate Gaussian probability
def calculateGaussianProbability(x, mean, stdev):
    epsilon = 1e-10
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
    return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo
# Function to calculate probabilities for each class
def calculateClassProbabilities(info, test):
    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std_dev = classSummaries[i]
            x = test[i]
            probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
    return probabilities
# Function to predict class for a test data point
def predict(info, test):
    probabilities = calculateClassProbabilities(info, test)
    bestLabel = max(probabilities, key=probabilities.get)
    return bestLabel
# Function to make predictions for the entire test set
def getPredictions(info, test):
    predictions = [predict(info, instance) for instance in test]
    return predictions
# Function to calculate accuracy rate
def accuracy_rate(test, predictions):
    correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
    return (correct / float(len(test))) * 100.0
# Load data using pandas
filename = 'C:/caar/College/Practicals/AI Sem 4/archive/diabetes.csv' # Add the correct file path
df = pd.read_csv(filename)
mydata = df.values.tolist()
# Encode classes and convert attributes to float
mydata = encode_class(mydata)
for i in range(len(mydata)):
    for j in range(len(mydata[i]) - 1):
        mydata[i][j] = float(mydata[i][j])
# Split the data into training and testing sets
ratio = 0.7
train_data, test_data = splitting(mydata, ratio)
print('Total number of examples:', len(mydata))
print('Training examples:', len(train_data))
print('Test examples:', len(test_data))
# Train the model
info = MeanAndStdDevForClass(train_data)
# Test the model
predictions = getPredictions(info, test_data)
accuracy = accuracy_rate(test_data, predictions)
print('Accuracy of the model:', accuracy)