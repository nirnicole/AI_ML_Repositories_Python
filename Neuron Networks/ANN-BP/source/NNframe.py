"""
Course: Biological computation
Name: Nir nicole
Module: ANN implementaion
"""
import numpy as np
import random

class NeuralNetwork:
    """
    This class outlines the structure of a Neural Network.
    """
    def __init__(self, n_inputs, n_hidden, n_outputs,layer_count=1, learning_rate=0.3, iterations=100, trace=False,binar_activation=False):

        # Seed the random number generator.
        random.seed(1)

        #initialize network properies
        self.input_size = n_inputs
        self.hidden_size = n_hidden
        self.output_size = n_outputs
        self.learning_rate = learning_rate
        self.epoch_count = iterations
        self.trace = trace
        self.binar_activation = binar_activation
        self.layers = list()
        self.initialize_layers(layer_count)

    # layer would have neuorns with wieghts inputing to them, first layer is implicit and bias neuron is added implicitly as well, as a weigth.
    def initialize_layers(self, size):

        #implicit input layer(a hidden layer), notice the adding weigth for bias
        self.layers.append([{'weights': [random.random() for weight in range(self.input_size+1)]} for weight in range(self.hidden_size)])

        #extra hidden layers
        for i in range(size-1):
            self.layers.append([{'weights': [random.random() for weight in range(self.hidden_size+1)]} for weight in range(self.hidden_size+1)])

        #output layer
        self.layers.append([{'weights': [random.random() for weight in range(self.hidden_size+1)]} for weight in range(self.output_size)])

    # training the network for a fixed number of epochs with BP
    def train(self, data_set, prediction_set, validation_data, validation_predictions, batch_ratio):
        """
        first we zip together data and predictions,
        then we iterate over the whole epoch with small batchs while shuffling the epoch each training iteration.
        than for every batch:
            1.ff -feed forword an input to calculate uotput (calculate activation and transfer forward)
            2.calculate error - Calculate the error, The difference between the desired output and the predicted output.
            3.bp - back propagate error and train the net(transfer the derivative and bp the error diffrences)
            4.update weights
        """
        stats = list()
        trian_stats = list()
        val_stats = list()

        # first we attach the data to the predictions so it wont get lost during shuffles
        train_set = self.attach_predictions(data_set, prediction_set)
        validation_set = self.attach_predictions(validation_data, validation_predictions)

        # batch size = the number of training examples in one forward/backward pass.
        batch_size = int(len(train_set) / batch_ratio)

        # one epoch = one "forward pass and backward pass" of all the training examples through the net.
        for epoch in range(self.epoch_count):
            train_loss = 0.0                                           #accomulated trainig error for displaying.
            val_loss = 0.0                                             #accomulated validation error for displaying.
            DATA = 0                                                    #naming for convinience only.
            PREDICTION = 1

            #initialize epoch and batch indices
            random.shuffle(train_set)
            batch_startindex = 0
            batch_endindex = batch_size

            # while epoch have not been fully coverd
            while batch_startindex < len(train_set):
                for set_index in range(batch_startindex ,batch_endindex):
                    train_expected = [0]*self.output_size                         #create a set of expected lightoff out_neurons.
                    expected_output_neuron = train_set[set_index][PREDICTION]
                    train_expected[expected_output_neuron] = 1                    #light up the predicted out_neuron.

                    outputs = self.feed_forward(train_set[set_index][DATA])
                    train_loss += self.calculate_error(train_expected, outputs)
                    self.Backpropagate(train_expected)
                    self.update_weights( train_set[set_index][DATA])
                batch_startindex += batch_size
                batch_endindex += batch_size

            #calculate and display errors during epochs
            for set_index in range(len(validation_set)):
                val_expected = [0] * self.output_size
                expected_output_neuron = validation_set[set_index][PREDICTION]
                val_expected[expected_output_neuron] = 1

                val_outputs = self.feed_forward(validation_set[set_index][DATA])
                val_loss += self.calculate_error(val_expected, val_outputs)

            val_loss *= (float(len(train_set)/len(validation_set)))             # normelize losses by ratio of val and train sets
            trian_stats.append(train_loss)
            val_stats.append(val_loss)
            if self.trace:
                print('\t=>\tEpoch = %d;\t\tTraining loss = %.3f;\t\tValidation loss = %.3f' % (epoch+1, train_loss,val_loss))

        stats.append(trian_stats)
        stats.append(val_stats)
        return stats

    # utility to attach the training set to its predictions
    def attach_predictions(self, data_set, prediction_set):
        train_set = []

        for index in range(len(data_set)):
            attached = []
            attached.append(data_set[index])
            attached.append(prediction_set[index])
            train_set.append(attached)

        return train_set

    # Forward the input through the network neurons and return the ff outputs for error tracking
    def feed_forward(self, input_neurons):

        inputs = input_neurons

        for layer in self.layers:
            outputs = []
            for neuron in layer:
                neuron['output'] = self.activate_neuron(neuron['weights'], inputs)
                outputs.append(neuron['output'])
            inputs = outputs
        ff_output_neurons = inputs

        return ff_output_neurons

    # error calculation for display purposes -> function =  1\2 * (expected-output)^2
    def calculate_error(self, expected, outputs):
        accomulated_error = 0.0
        for neuron_index in range(self.output_size):
            accomulated_error += pow((expected[neuron_index] - outputs[neuron_index]), 2)
        accomulated_error /= 2.0

        return accomulated_error

    # Backpropagate errors and store in neurons
    def Backpropagate(self, expected_outputs):
        """
        starting from the last layer,
        bp the delta backwards to the input layer.
        handle the last layer by the expected outcomes,
        every other layer's neuron error is calculated by the next layer neurons weigths
        that coneccted to it.
        the delta is propagating from the end to the input weigths through the net.
        """
        for layer_index in reversed(range(len(self.layers))):

            current_layer = self.layers[layer_index]
            current_layer_size = len(current_layer)
            error_set = []

            #if it is the ouput layer, handle by expected outcomes.
            if layer_index == len(self.layers) - 1:
                for neuron_index in range(current_layer_size):
                    neuron = current_layer[neuron_index]
                    error_set.append(expected_outputs[neuron_index] - neuron['output'])
            else:
                for neuron_index in range(current_layer_size):
                    neuron_error = 0.0
                    next_layer = self.layers[layer_index + 1]
                    for next_neuron in next_layer:
                        edge_weight = next_neuron['weights'][neuron_index]      #vertices of the same edge
                        edge_sensitivity = next_neuron['delta']
                        neuron_error += (edge_weight * edge_sensitivity)
                    error_set.append(neuron_error)

            #calculate neuron delta using claculated error for every neuron in the layer.
            for neuron_index in range(current_layer_size):
                neuron = current_layer[neuron_index]
                sensitivity = self.sigmoid_derivative(neuron['output'])
                error = error_set[neuron_index]
                neuron['delta'] = error * sensitivity

    # Update network weights with error accomulated
    def update_weights(self, input_neurons):
        """
        this function execute the "steps" of the optimization process.
        """
        for layer_index in range(len(self.layers)):
            # notice first layer special case.
            if layer_index == 0:
                prev_layer = input_neurons
            else:
                prev_layer = [neuron['output'] for neuron in self.layers[layer_index - 1]]

            #calculating the size of the step torward a gradient descent into a vally.
            for neuron in self.layers[layer_index]:
                for prev_neuron_index in range(len(prev_layer)):
                    step_size = prev_layer[prev_neuron_index] * neuron['delta'] * self.learning_rate
                    neuron['weights'][prev_neuron_index] += step_size
                #update weigth of the implicit biased neuron
                step_size = 1 * neuron['delta'] * self.learning_rate
                neuron['weights'][-1] += step_size

    # transfer the accomulated activision of a neuron using an activation function(here its sigmoid)
    def activate_neuron(self,weights, inputs):

        total_activation = 0
        bias = weights[-1]

        for index in range(len(weights) - 1):
            total_activation += (weights[index] * inputs[index])
        total_activation += bias                                 #assuming bias always connected to a 1.

        # i allowed only 2 options here for activation functions, more can be added!
        if not self.binar_activation:
            return self.sidmoid(total_activation)
        else:
            return self.binary_step(total_activation)

    # The Sigmoid function, we pass the weighted sum of the inputs
    # through this function to normalise them between 0 and 1.
    def sidmoid(self,x):
        return 1 / (1 + np.exp(-x))

    # This is the gradient of the Sigmoid curve,
    # indicates how confident we are about the existing weight.
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # also called Heaviside step function, activats binary - a non-differential function.
    def binary_step(self, x):
        if x<0:
            return 0
        else:
            return 1

    # classify a configuration on the trained network
    def oscillator_prediction(self, data):

        output_neurons = self.feed_forward( data)
        certainty = max(output_neurons)
        neuron_fired = output_neurons.index(certainty)

        return neuron_fired, certainty

    # input samples by string from the user
    def test_net(self, data, predictions, show_process=False):
        got = 0
        total_certainty = 0
        for line in range(len(data)):
            classification, certainty = self.oscillator_prediction(data[line])
            if show_process:
                print('\t=>\tExpected = %d;  Got = %d;  certainty = %0.2f' % (predictions[line], classification, 100 * certainty), "%")
            if predictions[line] == classification:
                got = got + 1
            elif show_process:
                matrix = []
                for i in range(0, len(data[line]), 10):
                    matrix.append(data[line][i:i + 10])
                for i in range(len(matrix)):
                    print(matrix[i])

            total_certainty += certainty

        print('\t=>\tsuccess: %d/%d' % (got, len(predictions)))
        print("\t=>\tsuccess rate: ", '%0.2f' % (100 * got / len(predictions)), "%")
        print("\t=>\tavrage confidence rate: ", '%0.2f' % (100 * total_certainty / len(predictions)), "%")

        results = []
        results.append(got)
        results.append(len(predictions))
        results.append(total_certainty / len(predictions))

        return results

    # utility to prints the net clearly
    def print_net(self):
        print("\nPrinting network:\n- notice that a bias neuron is implied as an extra weigth in the output layer.")
        for layer in range(len(self.layers)):
            print('\t=>\tLayer %d: ' % layer)
            for neuron in range(len(self.layers[layer])):
                print('\t\tNeuron %d ' % (neuron+1))
                for attribute, value in self.layers[layer][neuron].items():
                    print('\t\t{} : {}'.format(attribute, value))
                print("\n")
