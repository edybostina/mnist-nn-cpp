#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "mnist_reader.h"

using namespace std;

// Function to read MNIST data from a file
void readMNISTImages(const string &filename, vector<vector<float>> &images)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    int magicNumber = 0, numImages = 0, rows = 0, cols = 0;
    file.read((char *)&magicNumber, 4);
    file.read((char *)&numImages, 4);
    file.read((char *)&rows, 4);
    file.read((char *)&cols, 4);

    // Convert from big-endian to little-endian
    // __builtin_bswap32 is a GCC built-in function that swaps the byte order of a 32-bit integer
    magicNumber = __builtin_bswap32(magicNumber);
    numImages = __builtin_bswap32(numImages);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    images.resize(numImages, vector<float>(rows * cols));

    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < rows * cols; ++j)
        {
            unsigned char pixel;
            file.read((char *)&pixel, 1);
            images[i][j] = (float)(pixel) / 255.0f; // Normalize pixel values to [0, 1]
        }
    }

    file.close();
}

void readMNISTLabels(const string &filename, vector<int> &labels)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    int magicNumber = 0, numLabels = 0;
    file.read((char *)&magicNumber, 4);
    file.read((char *)&numLabels, 4);

    // Convert from big-endian to little-endian
    magicNumber = __builtin_bswap32(magicNumber);
    numLabels = __builtin_bswap32(numLabels);

    labels.resize(numLabels);
    for (int i = 0; i < numLabels; ++i)
    {
        unsigned char label;
        file.read((char *)&label, 1);
        labels[i] = (int)label;
    }
}

void testMNISTReader(const string &imageFile, const string &labelFile, int index, const string &output_file)
{
    ofstream output(output_file);

    vector<char> gradient = {
        ' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'};
    vector<vector<float>> images;
    vector<int> labels;

    readMNISTImages(imageFile, images);
    readMNISTLabels(labelFile, labels);

    if (index < 0 || (unsigned long)index >= images.size())
    {
        cerr << "Index out of bounds: " << index << endl;
        return;
    }

    output << "Label: " << labels[index] << endl;
    output << "Image: " << endl;

    for (int i = 0; i < 28; ++i)
    {
        for (int j = 0; j < 28; ++j)
        {
            float pixel = images[index][i * 28 + j];
            int gradientIndex = (int)(pixel * (gradient.size() - 1));
            output << gradient[gradientIndex];
        }
        output << endl;
    }
    output << "End of image" << endl;
    output.close();
}

void csv_reader(const string &filename, vector<vector<float>> &features, vector<int> &labels)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string line;
    getline(file, line); // Skip the header line
    while (getline(file, line))
    {
        vector<float> featureRow;
        size_t pos = 0;
        while ((pos = line.find(',')) != string::npos)
        {
            string token = line.substr(0, pos);

            featureRow.push_back(stof(token));
            line.erase(0, pos + 1);
        }
        labels.push_back(stoi(line)); // Last element is the label
        features.push_back(featureRow);
    }

    file.close();
}