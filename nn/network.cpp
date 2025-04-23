#include <iostream>
#include <vector>
#include <cmath>
#include "network.h"
#include "activation.h"
#include "../mnist/mnist_reader.h"
#include "../util/utils.h"
#include <random>

using namespace std;

NeuralNetwork::NeuralNetwork(const vector<int> &layers)
{
    srand(time(0)); // Seed for random number generation
    for (size_t i = 1; i < layers.size(); i++)
    {
        int inputs = layers[i - 1];
        int neurons = layers[i];
        vector<vector<float>> layerWeights(neurons, vector<float>(inputs));

        for (auto &neuronWeights : layerWeights)
        {
            for (auto &weight : neuronWeights)
            {
                weight = ((float)rand() / RAND_MAX) * 2 - 1; // range [-1, 1]
                // you can use different initialization methods here
            }
        }

        vector<float> layerBiases(neurons);
        for (auto &bias : layerBiases)
        {
            bias = ((float)rand() / RAND_MAX) * 2 - 1; // range [-1, 1]
            // you can use different initialization methods here
        }
        weights.push_back(layerWeights);
        biases.push_back(layerBiases);
    }
}

vector<vector<float>> NeuralNetwork::forward(const vector<float> &input)
{
    vector<vector<float>> activations;
    activations.push_back(input);

    vector<float> current = input;

    // Forward pass
    // Each layer's output is the next layer's input
    // The last layer uses softmax activation
    for (size_t layer = 0; layer < weights.size(); ++layer)
    {
        const auto &W = weights[layer];
        const auto &B = biases[layer];
        vector<float> Z(W.size());

        for (size_t neuron = 0; neuron < W.size(); ++neuron)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < W[neuron].size(); ++i)
            {
                sum += W[neuron][i] * current[i];
            }
            sum += B[neuron];
            Z[neuron] = sum;
        }
        vector<float> A(Z.size());
        if (layer == weights.size() - 1)
            A = softmax(Z);
        else
            for (size_t i = 0; i < Z.size(); ++i)
                A[i] = sigmoid(Z[i]);

        activations.push_back(A);
        current = A;
    }

    return activations;
}

// The backward pass uses the chain rule to compute the gradients
void NeuralNetwork::backward(const vector<float> &input, const vector<float> &expected, float learningRate)
{
    vector<vector<float>> activations = forward(input);
    vector<vector<float>> preActivations;
    preActivations.push_back(input);

    // Precompute the pre-activations for each layer
    vector<float> current = input;
    for (size_t layer = 0; layer < weights.size(); ++layer)
    {
        const auto &W = weights[layer];
        const auto &B = biases[layer];
        vector<float> Z(W.size());

        for (size_t neuron = 0; neuron < W.size(); ++neuron)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < W[neuron].size(); ++i)
            {
                sum += W[neuron][i] * current[i];
            }
            sum += B[neuron];
            Z[neuron] = sum;
        }
        preActivations.push_back(Z);
        current = Z;
    }

    vector<float> output = activations.back();
    vector<vector<float>> delta(weights.size());
    delta[weights.size() - 1] = vector<float>(output.size());

    for (size_t i = 0; i < output.size(); ++i)
    {
        delta[weights.size() - 1][i] = output[i] - expected[i];
    }

    // Backpropagation
    // Compute the gradients for each layer
    for (int layer = weights.size() - 2; layer >= 0; --layer)
    {
        delta[layer] = vector<float>(weights[layer].size());
        for (size_t j = 0; j < weights[layer].size(); ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < weights[layer + 1].size(); ++k)
            {
                sum += weights[layer + 1][k][j] * delta[layer + 1][k];
            }
            delta[layer][j] = sum * sigmoid_derivative(preActivations[layer][j]);
        }
    }

    // Update weights and biases
    // The weights are updated using the gradients computed during backpropagation
    for (size_t layer = 0; layer < weights.size(); ++layer)
    {
        for (size_t j = 0; j < weights[layer].size(); ++j)
        {
            const auto &previous_activations = (layer == 0) ? input : activations[layer - 1];

            for (size_t i = 0; i < weights[layer][j].size(); ++i)
            {
                weights[layer][j][i] -= learningRate * delta[layer][j] * previous_activations[i];
            }

            biases[layer][j] -= learningRate * delta[layer][j];
        }
    }
}

vector<float> NeuralNetwork::predict(const vector<float> &input)
{
    // The prediction is the output of the last layer
    // The input is passed through the network
    return forward(input).back();
}

// Evaluate the network on a dataset
// The evaluation computes the accuracy of the network on a given dataset
// The accuracy is the number of correct predictions divided by the total number of predictions
float NeuralNetwork::evaluate(const vector<vector<float>> &inputs, const vector<int> &labels, const string &filename)
{
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        vector<float> output = predict(inputs[i]);
        int predicted = argmax(output);
        if (predicted == labels[i])
            correct++;
        else if (filename != "") // I don't really know if this works, never tested it :p
        {
            ofstream file(filename, ios::app);
            if (file.is_open())
            {
                file << "Predicted: " << predicted << ", Actual: " << labels[i] << "\n";
                testMNISTReader("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", i, filename);
                file.close();
            }
        }
    }
    return static_cast<float>(correct) / inputs.size();
}

// Train the network with given inputs and labels
// The training process consists of multiple epochs
// In each epoch, the network is trained on the entire dataset
// The training process uses the backpropagation algorithm to update the weights and biases
void NeuralNetwork::train(const vector<vector<float>> &inputs, const vector<int> &labels, int epochs, float learningRate)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            vector<float> one_hot = oneHot(labels[i]);
            // cout << "back" << i << " " << epoch << "\n";
            backward(inputs[i], one_hot, learningRate);
        }
        float acc = evaluate(inputs, labels, "");
        cout << "Epoch " << epoch + 1 << " - Accuracy: " << acc * 100 << "%" << endl;
    }
}

void NeuralNetwork::save(const string &filename)
{
    ofstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    // Save weights and biases
    for (const auto &layer : weights)
    {
        for (const auto &neuron : layer)
        {
            for (const auto &weight : neuron)
            {
                file.write(reinterpret_cast<const char *>(&weight), sizeof(weight));
            }
        }
    }

    for (const auto &layer : biases)
    {
        for (const auto &bias : layer)
        {
            file.write(reinterpret_cast<const char *>(&bias), sizeof(bias));
        }
    }

    file.close();
}

int NeuralNetwork::load(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return 0; // failure
    }

    // Load weights and biases
    for (auto &layer : weights)
    {
        for (auto &neuron : layer)
        {
            for (auto &weight : neuron)
            {
                file.read(reinterpret_cast<char *>(&weight), sizeof(weight));
            }
        }
    }

    for (auto &layer : biases)
    {
        for (auto &bias : layer)
        {
            file.read(reinterpret_cast<char *>(&bias), sizeof(bias));
        }
    }
    return 1; // success
    file.close();
}