#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>

#include "mnist/mnist_reader.h"
#include "nn/network.h"
#include "util/utils.h"

using namespace std;

void mnist_data_set_nn()
{
    string imageFile = "data/train-images.idx3-ubyte";
    string labelFile = "data/train-labels.idx1-ubyte";
    vector<vector<float>> images;
    vector<int> labels;

    readMNISTImages(imageFile, images);
    readMNISTLabels(labelFile, labels);

    // Test the MNIST reader
    // int index = 34876; // Change this to test different images
    // testMNISTReader(imageFile, labelFile, index, "stdout");

    NeuralNetwork net({784, 128, 10}); // our neural network

    if (net.load("models/model.bin") == 0)
    {
        cout << "Could not load the model." << endl;
        cout << "Training a new model..." << endl;
        net.train(images, labels, 10, 0.01f); // Train the network
    }

    // Evaluate the model
    vector<vector<float>> testImages;
    vector<int> testLabels;
    readMNISTImages("data/t10k-images.idx3-ubyte", testImages);
    readMNISTLabels("data/t10k-labels.idx1-ubyte", testLabels);

    float accuracy = net.evaluate(testImages, testLabels, "");
    cout << "Test accuracy: " << accuracy * 100.0f << "%" << endl;

    // test 1 good image
    for (size_t i = 0; i < testImages.size(); i++)
    {
        int index = rand() % testImages.size();
        vector<float> output = net.predict(testImages[index]);
        int predicted = argmax(output);
        if (predicted != testLabels[index])
            continue;
        ofstream file("out/good.txt");
        file << "Predicted: " << predicted << ", Actual: " << testLabels[index] << "\n";
        testMNISTReader("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", index, "out/good.txt");
        break;
    }

    // test 1 bad image
    for (size_t i = 0; i < testImages.size(); i++)
    {
        int index = rand() % testImages.size();
        vector<float> output = net.predict(testImages[index]);
        int predicted = argmax(output);
        if (predicted == testLabels[index])
            continue;
        ofstream file("out/bad.txt");
        file << "Predicted: " << predicted << ", Actual: " << testLabels[index] << "\n";
        testMNISTReader("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", index, "out/bad.txt");
        break;
    }

    // Save the model
    net.save("models/model.bin");
}

// Ignore this function, it's just for testing
// This function normalizes the vector to the range [0, 1]
// It is not used in the main function
void normalize(vector<float> &vec)
{
    float minVal = *min_element(vec.begin(), vec.end());
    float maxVal = *max_element(vec.begin(), vec.end());

    for (auto &val : vec)
    {
        val = (val - minVal) / (maxVal - minVal);
    }
}

int main()
{
    mnist_data_set_nn();
    return 0;
}
