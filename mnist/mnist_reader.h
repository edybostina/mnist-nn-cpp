#ifndef MNIST_READER_H
#define MNIST_READER_H
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
using namespace std;

// Function to read MNIST data from a file
void readMNISTImages(const string &filename, vector<vector<float>> &images);

// Function to read MNIST labels from a file
void readMNISTLabels(const string &filename, vector<int> &labels);

// Function to test the MNIST reader
// Prints ASCII art of the digit read from the file
void testMNISTReader(const string &imageFile, const string &labelFile, int index, const string &output_file);

// Function to read CSV data
void csv_reader(const string &filename, vector<vector<float>> &features, vector<int> &labels);

#endif // MNIST_READER_H