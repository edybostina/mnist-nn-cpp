#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

float sigmoid(float x);

float sigmoid_derivative(float x);

vector<float> softmax(const vector<float> &z);

#endif // ACTIVATION_H