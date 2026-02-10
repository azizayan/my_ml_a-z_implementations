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


def manhattan_calculator(list1: List[float], list2: List[float]):
    length = len(list2)

    result = 0.0

    for i in range(length):
        num1 = list1[i]
        num2 = list2[i]

        distance = math.abs(num1-num2)
        result += distance

    return result

def load_csv(filename: Path):
    dataset = []
    with open(filename,'r') as file:

        csv_reader = csv.reader(file)

        next(csv_reader)

        for row in csv_reader:

            if not row:
                continue

            

        

def get_neighbors(train_data, test_row, num_neighbors):
    

        

        
    

    

