#include <iostream>
#include <vector>
#include <cmath>
#include "activation.h"

using namespace std;

// sigmoid activation function
// The sigmoid function is defined as 1 / (1 + exp(-x))
float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

// sigmoid derivative
// The derivative of the sigmoid function is sigmoid * (1 - sigmoid)
float sigmoid_derivative(float x)
{
    float sig = sigmoid(x);
    return sig * (1.0f - sig);
}

// softmax activation function
// The softmax function is defined as exp(z_i) / sum(exp(z_j)) for all j
vector<float> softmax(const vector<float> &z)
{
    vector<float> result(z.size());
    float maxVal = *max_element(z.begin(), z.end());

    float sum = 0.0f;
    for (size_t i = 0; i < z.size(); ++i)
    {
        result[i] = exp(z[i] - maxVal); // for numerical stability
        sum += result[i];
    }

    for (size_t i = 0; i < result.size(); ++i)
    {
        result[i] /= sum;
    }

    return result;
}
