#######################################################################################################################################
# This file contains code in order to implement the KNN model for AUCSC 460
# It takes the diabetes dataset and performs the KNN procudure on the data where k=3
# User can also choose to compare all odd Ks from 1 to 100 in order to compare their accuracy
# Source for the diabetes.csv dataset can be found here: https://github.com/dylan-slack/TalkToModel/blob/main/data/diabetes.csv
#
# Class: AUCSC 460
# Name: Zachary Kelly
# Student ID: 1236421
# Date: February 14th, 2024
#######################################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statistics import mode

class KNearestNeighboursClassifier():

    #initializing k
    def __init__(self, k) :

        self.k = k

    #Storing the training set
    def store(self, trainingSetX, trainingSetY):

        self.trainingSetX = trainingSetX
        self.trainingSetY = trainingSetY

        #Number of training examples, number of training features
        self.trainingExamples, self.trainingFeatures = trainingSetX.shape

    
    #Making predicitons based on the training set
    def makePrediction(self, testingSetX):

        self.testingSetX = testingSetX

        #Number of test examples, number of test features
        self.testExamples, self.testFeatures = testingSetX.shape

        #Initializing the prediciton array
        predictions = np.zeros(self.testExamples)

        #iterate through test examples and predict their diabetes status
        for i in range(self.testExamples):

            #grabbing the row from the testing set to be used this iteration
            testSetSelection = self.testingSetX[i]

            #Finding the K nearest neighbours to the current testing example
            neighbours = np.zeros(self.k)
            neighbours = self.nearestNeighbours(testSetSelection)

            #Selecting from neightbours array the most common value and assigning it to the predicitons array
            predictions[i] = mode(neighbours)
        
        return predictions

    #calculating the K number of nearest neighbours based on some row from the X testing set
    def nearestNeighbours(self, testSetSelection):

        #initializing the distance array
        cosineDistances = np.zeros(self.trainingExamples)

        #iterating through all the trainingExamples and calculating distance from testing set selection
        for i in range(self.trainingExamples):

            distance = self.calculateCosineDistance(testSetSelection, self.trainingSetX[i])
            cosineDistances[i] = distance

        #sort cosine distances by index using argsort()
        sortedIndices = np.argsort(cosineDistances)

        #I HAVE TAKEN EACH EXAMPLE FROM THE TESTING SET X (20%) AND CALCED ITS COSINE DISTANCE TO EACH EXAMPLE IN THE TRAINING SET X(80%)
        #I HAVE THE SMALLEST COSINE DISTANCES INDICIES, AND I USE THOSE TO TAKE THE DIABETES VALUE FROM THE TRAINING SET Y (80%)

        closestValues = []

        #Append to the closestValues array the values from trainingSetY using the indicies from the sortedIndicies array.
        for i in range(self.k):
            
            closestValues.append(self.trainingSetY[sortedIndices[i]])
        
        return closestValues

    #first calculates cossine simularity then converts to, and returns, cosince distance
    def calculateCosineDistance(self, A, B):

        dotProduct = np.dot(A, B)
        normA = np.linalg.norm(A)
        normB = np.linalg.norm(B)

        #calculating cosine simularity
        cosineSimilarity = dotProduct / (normA * normB)

        #calculating cossine distance
        cosineDistance = 1 - cosineSimilarity

        return cosineDistance



# DRIVER CODE #

def main():

    print("This program will perform the KNN test on the Diabetes dataset where K is equal to 3")
    print("If you also wish to plot the accuracy of K from 1-100 (only odd numbers) enter 1, otherwise enter 0.")

    while True:

        try:

            flag = int(input("Enter 1 or 0: "))

        except ValueError:

            print()
            print("Sorry that input is incorrect.")
            print("Please input 1 to plot the accuracy of K from 1-100 (only odd numbers), otherwise 0")
            continue
        
        if flag == 1 or flag == 0:
            print()
            break

        else:

            print()
            print("Sorry that input is incorrect.")
            print("Please input 1 to plot the accuracy of K from 1-100 (only odd numbers), otherwise 0")


    #reading in data from the dataset
    df = pd.read_csv('diabetes.csv')

    #storing the results from the dataset
    y = df['y'].values

    #removing the results from the dataset from what will be the training and testing datasets
    x = df.drop(['y'], axis=1).values

    #partitioning the data into the testing and training datasets
    trainingDataX, testingDataX, trainingDataY, testingDataY = train_test_split( x,y, test_size = 0.2)

    #Standardizing the data to have zero mean and variance
    sScaler = StandardScaler().fit(trainingDataX)
    trainingDataX = sScaler.transform(trainingDataX)
    testingDataX = sScaler.transform(testingDataX)

    #Creating model
    modelKNN = KNearestNeighboursClassifier(k = 3)

    #Storing the training data
    modelKNN.store(trainingDataX, trainingDataY)

    #Making predictions for the 
    testSetPrediction = modelKNN.makePrediction(testingDataX)

    #labels for the classification report
    targetNames = ['Does not have Diabetes', 'Has Diabetes']

    print("CLASSIFICATION REPORT FOR KNN MODEL ON DIABETES DATASET WHEN K=3\n")
    print(classification_report(testingDataY, testSetPrediction, target_names=targetNames))

    #flag for determining if user wants ks 1-100 plotted
    if flag == 1:

        setOfAccuracies = []
        ks = []

        #populating the k array
        for i in range(1, 100):
            if i % 2 == 1:
                ks.append(i)

        #iterating through ks to be plotted and adding accuracies to the setOfAccuracies array
        for k in ks:
            kTestingModel = KNearestNeighboursClassifier(k = k)
            kTestingModel.store(trainingDataX, trainingDataY)
            kTestSetPrediction = kTestingModel.makePrediction(testingDataX)

            correctlyClassified = 0
            count = 0

            #calculating accuracy for each given k
            for i in range(np.size(kTestSetPrediction)):
                if testingDataY[i] == kTestSetPrediction[i]:
                    correctlyClassified = correctlyClassified + 1
                count = count + 1
            
            accuracyOfModel = (correctlyClassified / count) * 100

            setOfAccuracies.append(accuracyOfModel)

        #plotting the accuracies
        fig, ax = plt.subplots()
        ax.plot(ks, setOfAccuracies, "-o")
        ax.set(xlabel="k",
        ylabel="Accuracy",
        title="Performance of KNN model on the Diabetes Dataset for all odd Ks 1-100")
        plt.show()

if __name__ == "__main__":

    main()