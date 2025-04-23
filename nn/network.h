#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

class NeuralNetwork
{
public:
    // Constructor
    NeuralNetwork(const vector<int> &layers); // example: {784, 128, 10} for MNIST

    // Predict the output for a given input
    vector<float> predict(const vector<float> &input);

    // Train the network with given inputs and labels
    void train(const vector<vector<float>> &inputs, const vector<int> &labels, int epochs, float learningRate);

    // Evaluate the network on a dataset
    float evaluate(const vector<vector<float>> &inputs, const vector<int> &labels, const string &filename);

    // Save the model to a bin file.
    // The model is saved as a binary file containing the weights and biases
    void save(const string &filename);

    // Load the model from a bin file
    // The model is loaded from a binary file containing the weights and biases
    int load(const string &filename);

private:
    vector<vector<vector<float>>> weights; // [layer][neuron][input]
    vector<vector<float>> biases;          // [layer][neuron]

    // Forward pass
    // The forward pass computes the output of the network for a given input
    vector<vector<float>> forward(const vector<float> &input);

    // Backward pass
    // The backward pass computes the gradients and updates the weights and biases
    void backward(const vector<float> &input, const vector<float> &expected, float learningRate);
};

#endif // NETWORK_H