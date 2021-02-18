import numpy as np

class NeuralNetwork:
    """
    This class outlines the structure of a Neural Network.

    Network flow:
    1. Take inputs.
    2. Add bias (if required).
    3. Assign random weights in the hidden layer and the output layer.
    4. Run the code for training.
    5. Find the error in prediction.
    6. Update the weight values of the hidden layer and output layer by gradient descent algorithm.
    7. Repeat the training phase with updated weights.
    8. Make predictions.
    9. Test new unseen inputs.

    takes:
    - input_size -> base population size.
    - training_set_size -> number of inputs for training.
    - testing_set_size -> number of inputs for testing.
    - layers -> default for now is 1 hidden layer, we cover multiple layer networks in deep learning.
    - bias -> boolean indicator for adding a biased weight.
    - batches -> number of batches.
    - learning_rate -> rate of adjustments to the weights.

    """
    def __init__(self, input_size, training_set_size, testing_set_size, layers=1, use_bias=True, learning_rate=0.3, trace=False):

        # Seed the random number generator.
        np.random.seed(1)

        self.input_size = input_size
        self.output_count = 1
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.training_set_size = training_set_size
        self.testing_set_size = testing_set_size
        self.layers_count = layers
        self.trace = trace

        self.create_layers()

        self.training_set_inputs, self.training_set_outputs = self.initialize_set(self.training_set_size)
        self.testing_set_inputs, self.testing_set_outputs = self.initialize_set(self.testing_set_size)

    # inner class that represent a neurons layer
    class LayerDense:
        def __init__(self, layer_neurons_count, bias):

                self.biases = np.ones((1, 1))
                self.synaptic_weights = self.initialize_weights(layer_neurons_count)
                if bias:
                    print("\nMAMA MIA")
                    self.synaptic_weights = np.append(self.synaptic_weights, self.biases, axis=0)

        # randomally assign 0 to 1 float values for initialization.
        def initialize_weights(self, size_of_layer):
            weights = 2 * np.random.random((size_of_layer, 1)) - 1
            return weights

        # pass the data out throug the layer's neurons
        def forward(self, inputs):
            self.output = NeuralNetwork.sidmoid(np.dot(inputs.astype(float),self.synaptic_weights))

    def create_layers(self):
        #
        self.layers = [self.LayerDense(self.input_size, self.use_bias) for _ in range(self.layers_count)]

    def initialize_set(self, size):

        #Input neurons layer
        inputs = self.input_neurons(size)

        if self.use_bias:
            inputs = np.insert(inputs, self.input_size, 1, axis=1)

        #Output neorons layer
        outputs = self.output_neurons(inputs)

        return inputs, outputs.T

    def input_neurons(self, size):
        """
        must implement individualy
        """
        self.raiseNotDefined()

    def output_neurons(self, inputs):
        """
        must implement individualy
        """
        self.raiseNotDefined()

    # training the network through a process of trial and error, while
    # Adjusting the synaptic weights each time.
    def train(self, iterations):
        """
        1. Pass the training set through our neural network as input for the first layer.
        2. Calculate the error, The difference between the desired output
        and the predicted output).
        3. Calculate the delta, By looking at the weights in a layer we can
        determine by how much a layer contributed to the error in next layer.
        4. Calculate how much to adjust the layer's weights by.

        param: number of training sessions watned
        iterations: int
        return: none
        """

        for iter in range(iterations):
            prev_input = self.training_set_inputs

            # stream values throug the layer
            self.layers[0].forward(prev_input)
            layer_outputs = self.layers[0].output

            # trace performance
            if not self.trace and iter%(iterations/100) == 0:
                self.error_stats(self.training_set_outputs,layer_outputs,iter)

            #hidden layers
            #if self.layers_count>1:
                #for prev_layer in range(self.layers_count):
                    #Implement hopefully next

            # calculate distance from desired outcome
            layer_error = self.training_set_outputs - layer_outputs
            layer_delta = layer_error * self.sigmoid_derivative(layer_outputs)

            # adjust accordingly
            layer_adjustments = self.learning_rate * np.dot(prev_input.T, layer_delta)
            self.layers[0].synaptic_weights += layer_adjustments

    # The neural network thinks, therefor - it exists.
    def think(self, inputs):
        """
        run new unkown data throug it.
        """
        self.layers[0].forward(inputs)
        return self.layers[0].output

    # input samples by string from the user
    def test_net(self, input_string):

        arr = [[float(c)] for c in input_string]
        output = self.think(np.array(arr).T)

        print(">>>\tClassified as:\t", self.classify_result(output))
        return output

    def classify_result(self, subject):
        """
        must implement individualy
        """
        self.raiseNotDefined()

    # The Sigmoid function, we pass the weighted sum of the inputs
    # through this function to normalise them between 0 and 1.
    def sidmoid(x):
        return 1 / (1 + np.exp(-x))

    # This is the gradient of the Sigmoid curve,
    # indicates how confident we are about the existing weight.
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def raiseNotDefined(self):
        """
        basic notification for an undefined method.
        """
        print("Method not implemented: %s" % inspect.stack()[1][3])
        sys.exit(1)

    # calculate errors
    def error_stats(self,desired, results, iteration):
        layer_errors = desired - results
        count = round(float(np.mean(np.abs(layer_errors)) *100),2)
        print(f">>>\tIteration {iteration}:\tfound {count}% errors.")


#optional activations
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

