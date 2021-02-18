import NNframe
import numpy as np

INPUT_SIZE = 21
TRAINING_SIZE = 100
TEST_SIZE = 100
TRAINING_ITERATIONS = 5000

class Perceptron(NNframe.NeuralNetwork):

    def __init__(self, input_size, training_set_size, testing_set_size, layers=1, use_bias=True, learning_rate=0.3):
        NNframe.NeuralNetwork.__init__(self,input_size, training_set_size, testing_set_size, layers, use_bias, learning_rate)

    def input_neurons(self, size):

        return np.array([np.random.choice([0, 1], size=self.input_size) for _ in range(size)])

    def output_neurons(self, inputs):

        return np.array([[self.most_frequent_element(bit) for bit in inputs]])

    def most_frequent_element(self, arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]

    def classify_result(self, subject):

        return "0" if subject < 0.5 else "1"

def run_perceptron_demo(input_size = INPUT_SIZE, training_set_size = TRAINING_SIZE, testing_set_size = TEST_SIZE, train_amount = TRAINING_ITERATIONS):

    print("\n>>>\tThis is a demonstration of a perceptromn simulation!\n")

    print(">>>\tInitializing Perceptron...")
    percep = Perceptron(input_size, training_set_size, testing_set_size)
    print(f">>>\tPerceptron created with a set of {training_set_size}  input elements for training.")

    print("\n>>>\tSet for trainig is:")
    print(percep.training_set_inputs)

    print("\n>>>\tExcpected outputs are:")
    print(percep.training_set_outputs.T)

    print(f"\n>>>\tStarting training session at a size of {training_set_size} with {train_amount} repeats...")
    # Train the neural network using the training set.
    percep.train(train_amount)
    print(">>>\tTraining ended successfully!")

    print("\n>>>\tSynaptic weights of output neurons after training:")

    out = [neuron[0] for neuron in percep.layers[0].output]
    print(out)

    arr = percep.training_set_outputs
    finalarr = []
    for x in arr:
        for y in x:
            finalarr.append(str(y))

    print("\n>>>\tExpected outcome:\n", finalarr)
    out = [percep.classify_result(neuron) for neuron in percep.layers[0].output]
    print("\n>>>\tTrained neurons results (output neurons after classifications):\n", out)

    print("\n>>>\tTesting unseen nested testing set...")
    arr = percep.testing_set_outputs
    finalarr = []
    for x in arr:
        for y in x:
            finalarr.append(str(y))

    print(">>>\tExpected outcome:\n", finalarr)
    out_raw = percep.think(percep.testing_set_inputs)
    out = [neuron[0] for neuron in out_raw]
    print("\n>>>\tOutput neurons wieghts:\n", out)
    out = [percep.classify_result(neuron) for neuron in out_raw]
    print("\n>>>\tOutput neurons results (output neurons after classifications):\n", out)

    print(f"\n>>>\tBias value:\t{percep.use_bias}.")
    print(f">>>\tOutput neurons errors for a {testing_set_size} test size after learning a {training_set_size} training size:")
    percep.error_stats(percep.testing_set_outputs, out_raw, 1)

    print("\n>>>\tTest unseen input...")
    s =  "1010101010001000001111"
    percep.test_net(str(input(f"Please insert 21 bit:\n(you can copy paste from here ->  {s} )\n\t")))

    print("\n>>>\tEnding session...")

if __name__ == "__main__":
    run_perceptron_demo()

