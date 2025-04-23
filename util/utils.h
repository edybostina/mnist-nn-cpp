#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
using namespace std;

// Function to find the index of the maximum element in a vector
int argmax(const vector<float> &vec);

// Function to create a one-hot encoded vector for a given label
vector<float> oneHot(int label);

#endif // UTILS_H