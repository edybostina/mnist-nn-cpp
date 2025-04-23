#include <iostream>
#include <vector>
#include "util.h"

using namespace std;

int argmax(const vector<float> &vec)
{
    return max_element(vec.begin(), vec.end()) - vec.begin();
}

vector<float> oneHot(int label)
{
    vector<float> vec(10, 0.0f);
    vec[label] = 1.0f;
    return vec;
}