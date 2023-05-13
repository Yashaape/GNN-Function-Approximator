# Beginning of GNN code
import numpy as np
import random
import pickle
import math


class Node:

    def __init__(self, lastlayer = None):
        self.lastlayer = lastlayer
        self.collector = 0.0   #sum of weighted inmputs to neuron
        self.connections = []  #list of neurons in previous layer connected to it's neuron
        self.weights = [] #associated with each connection to previous layer
        self.delta = 0.0 #Used to store error in the output of the neuron for back propagation


net_structure = np.array([4, 2, 1])  # 4 input layers, 2 hidden layers, 1 output
input_data = np.arange(-10, 10, 0.1)  # range of values for training
output_layer = None
net = []


def generate_data(start=-10, end=10, step=0.1):
    x = np.arange(start, end, step)
    y = (np.cos(x) + 1) / 2
    return x, y


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2


#@param: network_structure: a numpy list that gives the number of nodes for each layer of the network
#@return: net: a list of lists representing the initialized network
def initialize_network(network_structure):
    global output_layer
    for i in range(len(network_structure)):
        layer = []
        if i == 0:
            for j in range(network_structure[0]):
                layer.append(Node())  # appends new node obj to layer list
            net.append(layer)
        else:
            prev_layer_size = network_structure[i-1]
            for j in range(network_structure[i]):
                node = Node()
                node.connections = net[-1]
                node.weights = [{'weights': np.random.uniform(-1, 1)} for _ in range(prev_layer_size)]
                layer.append(node)
                print(node.weights)
            net.append(layer)
    output_layer = net[-1]
    #print(output_layer)
    return net


def activation(neuron):
    sum_of_weights = 0.0
    for index in range(len(neuron.connections)):
        con = neuron.connections[index]
        weight = neuron.weights[index]['weights'] # access value from dictionary
        sum_of_weights += con.collector * weight
        #print(sum_of_weights)
    #print(sigmoidal(sum_of_weights))
    return tanh(sum_of_weights)


def forward_propagation(network):
    for layer in network:
        for node in layer:
            node.collector = activation(node)
    return network[-1][0].collector


def back_propagation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i == len(network) - 1:
            #for output layer
            for j in range(len(layer)):
                node = layer[j]
                output = node.collector
                error = expected - output
                errors.append(error)
        else:
            #for hidden layers
            for j in range(len(layer)):
                node = layer[j]
                error = 0.0
                for neuron in network[i + 1]:
                    error += neuron.weights[j]['weights'] * neuron.delta
                errors.append(error)

        for j in range(len(layer)):
            node = layer[j]
            delta = errors[j] * tanh_deriv(node.collector)
            node.delta = delta

def update_weights(network, lr):
    for i in range(len(network)):
        layer = network[i]
        for node in layer:
            for j in range(len(node.connections)):
                con = node.connections[j]
                weight = node.weights[j]
                weight['weights'] += lr * node.delta * con.collector


def train_network(network, train, lr, n_epochs, target_error):
    epoch = 0
    while epoch < n_epochs:
        error_sum = 0.0
        for data in train:
            input_values = data['input']
            expected_output = data['output']

            #Forward Prop:
            for i in range(len(input_values)):
                network[0][i].collector = input_values[i]
            output = forward_propagation(network)

            #calculate error:
            error = np.mean((expected_output - output)**2)
            error_sum += error

            #back prop:
            back_propagation(network, expected_output)

            #update weights:
            update_weights(network, lr)

        #calculate average error for this epoch
        avg_error = error_sum / len(train)
        print("Epoch: {}, l_rate: {}, Error: {}".format(epoch, lr, avg_error))

        # Check if target error is reached:
        if avg_error <= target_error:
            print("Target error reached: {}".format(target_error))
            break
        epoch += 1
    #epoch +=1
    if epoch == n_epochs:
        print("Maximum number of epochs reached. Training stopped.")


def main():
    global output_layer
    print("Network Structure:")
    print(net_structure)

    net = initialize_network(net_structure)
    #nd = node()  This was the cause of my sum being incorrect creating the node outside the loop causes sum errors

    print("network: ", net)
    print("output layer: ", output_layer) #Output layer aka last layer
    print()

    print("Inputs:")
    print(input_data, "\n")

    train_data = [{'input': [x], 'output': [(np.cos(x) + 1) / 2]} for x in input_data]
    print("Training.......................................................")

    train_network(net, train_data, lr=0.01, n_epochs=100, target_error=0.001)

    print("Output Layer: ")
    print(net[-1][0].collector)
    #print(len(input_data))
    #print(sum(input_data))



    with open('doneANN.pickle', 'wb') as handle:
        pickle.dump(net, handle)


main()