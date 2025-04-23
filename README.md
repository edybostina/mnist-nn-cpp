# MNIST Neural Network in C++

This repository contains a C++ implementation of a simple neural network designed to classify handwritten digits from the MNIST dataset. The project demonstrates the use of basic machine learning concepts such as forward propagation, backpropagation, and gradient descent.

## Features

- **Neural Network Architecture**: Fully connected feedforward neural network with customizable layers.
- **Activation Functions**: Includes sigmoid and softmax activation functions.
- **Training and Evaluation**: Supports training on the MNIST dataset and evaluating the model's accuracy.
- **Model Persistence**: Save and load trained models for reuse.
- **MNIST Data Handling**: Includes utilities to read and visualize MNIST data.

# Train a new model

The project comes with a pre-trained model.
To train a new one, just delete the `model.bin` file and run the program.

## Project Structure

```
mnist-c++/
├── mnist/               # MNIST data reader
│   ├── mnist_reader.cpp
│   ├── mnist_reader.h
├── nn/                  # Neural network implementation
│   ├── network.cpp
│   ├── network.h
│   ├── activation.cpp
│   ├── activation.h
├── util/                # Utility functions
│   ├── utils.cpp
│   ├── utils.h
├── data/                # Dataset files (not included in the repo)
├── models/              # Directory to save trained models
├── out/                 # Output files (e.g., good.txt, bad.txt)
├── main.cpp             # Entry point of the application
├── Makefile             # Build instructions
└── README.md            # Project documentation
└── LICENSE              # Project license
```

## Prerequisites

- C++17 or later
- MNIST dataset files:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

## Build and Run

1. Clone the repository:

   ```bash
   git clone https://github.com/edybostina/mnist-nn-cpp.git
   cd mnist-cpp
   ```

2. Place the MNIST dataset files in the `data/` directory.

3. Build the project using the provided `Makefile`:

   ```bash
   make
   ```

4. Run the program:
   ```bash
   ./mnist_nn
   ```

## Output

- **Good Predictions**: Saved in `out/good.txt` with ASCII visualization of correctly classified digits.
- **Bad Predictions**: Saved in `out/bad.txt` with ASCII visualization of misclassified digits.
- **Model**: Saved in `models/model.bin` after training.

## Example

### Good Prediction

```
Predicted: 3, Actual: 3
<ASCII visualization of the digit>
```

### Bad Prediction

```
Predicted: 9, Actual: 8
<ASCII visualization of the digit>
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- ASCII visualization inspired by gradient-based rendering.

Feel free to contribute or raise issues for improvements!
