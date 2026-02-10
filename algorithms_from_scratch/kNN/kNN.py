import pandas as pd
import numpy as np
from typing import List
import math
import csv
from pathlib import Path


dataset = pd.read_csv('iris.csv')


dataset.info()


print(dataset.head)




def euclidean_calculator(list1: List[float], list2: List[float]):
    length = len(list1)

    result = 0.0

    for  i in range(length):
        num1 = list1[i]
        num2 = list2[i]

        distance = num1-num2
        distance_square = distance * distance
        result += distance_square


    return math.sqrt(result)

def load_csv(filename: Path):
    dataset = []
    with open(filename,'r') as file:

        csv_reader = csv.reader(file)

        next(csv_reader)

        for row in csv_reader:

            if not row:
                continue

            cleaned_row = []

            for i in range(4):
                cleaned_row.append(float(row[i]))

            cleaned_row.append(row[4])

            dataset.append(cleaned_row)

    return dataset

manual_dataset = load_csv('iris.csv')


            
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(manual_dataset, test_size= 0.2, random_state=0)

        

def get_neighbors(train_data, test_row, num_neighbors):
    distances = []

    test_features = test_row[:-1]

    for row in train_data:
        train_features = row[:-1]
        

        distance = euclidean_calculator(train_features, test_features)
        distances.append((row,distance))

    distances.sort(key= lambda x:x[1])
    neighbors = [t[0] for t in distances[:num_neighbors]]
    return neighbors
    

def predict_classification(train, test_row, num_neighbors):

    neighbors = get_neighbors(train,test_row,num_neighbors)
    labels = []

    for neighbor in neighbors:
        labels.append(neighbor[-1])

    most_common_label = max(set(labels), key=labels.count)

    return most_common_label

def get_accuracy(train, test, num_neighbors):
    if not test:
        return 0.0


    true_counter = 0
    
    for row in test:
        predicted = predict_classification(train,row, num_neighbors)
        actual = row[-1]
        if predicted == actual:
            true_counter += 1
    

    return(true_counter/len(test))

def main():
    k = 3
    accuracy = get_accuracy(train_set, test_set, k)
    print(f"Accuracy (k={k}): {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()







