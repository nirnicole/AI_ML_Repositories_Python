"""
Course: Biological computation
Name: Nir nicole
Module: Game Of Life utilities
"""
import random

###################

def rleTobinarymat(rle, lines=10, columns=10):

    Matrix = [[0 for x in range(lines)] for y in range(columns)]
    line=0
    column=0
    count=1

    for char in rle:
        if char=='o':
            for i in range(count):
                Matrix[line][column+i]=1
            column = column + count
            count = 1
        elif char=='b':
            column=column+count
            count=1;
        elif char=='$':
            line=line+1
            column=0;
        elif char=='!':
            break
        elif int(char) < 10:
            count = int(char)

    return Matrix

def importDataSet(path):

    data_set =list()
    f = open(path, "r")
    Lines = f.readlines()

    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        rle = line.strip()
        #print("Line{}: {}".format(count, rle))
        data_set.append(rle)
    return data_set

def printData(data_set):
    for rle in data_set:
        mat =rleTobinarymat(rle)
        print(rle)
        for i in range(len(mat)):
            print(mat[i]);

def translateData(data_set):

    data = list()

    for rle in data_set:
        result = list()
        mat =rleTobinarymat(rle)
        for i in range(len(mat)):
            result = result + mat[i]
        data.append(result)

    return data

def get_dataset(path):

    data = importDataSet(path)
    data = translateData(data)
    random.shuffle(data)

    return data

def split_sets(oscilators_data, nonoscilators_data, training_size, validation_size, test_size):

    train_cut = int(training_size/2)
    validation_cut = int(validation_size/2)
    test_cut = int(test_size/2)

    test1 = oscilators_data[:test_cut]
    test2 = nonoscilators_data[:test_cut]

    validation1 = oscilators_data[test_cut:test_cut+validation_cut]
    oscilators_data = oscilators_data[test_cut+validation_cut:train_cut+test_cut+validation_cut]
    validation2 = nonoscilators_data[test_cut:test_cut+validation_cut]
    nonoscilators_data = nonoscilators_data[test_cut+validation_cut:train_cut+test_cut+validation_cut]

    predictions1 = [1] * len(oscilators_data)
    predictions2 = [0] * len(nonoscilators_data)
    train_predictions = predictions1 + predictions2

    validation1_predictions = [1] * len(validation1)
    validation2_predictions = [0] * len(validation2)
    validation_predictions = validation1_predictions + validation2_predictions

    test1_predictions = [1] * len(test1)
    test2_predictions = [0] * len(test2)
    test_predictions = test1_predictions + test2_predictions

    test_data = test1 + test2
    validation_data = validation1 + validation2
    train_data = oscilators_data + nonoscilators_data

    return train_data, train_predictions, validation_data, validation_predictions, test_data, test_predictions
