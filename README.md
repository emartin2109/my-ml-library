
# Basic Machine Learning Library

This project is a modular machine learning library written in C++ that enables users to configure, train, and use neural networks with ease. The library supports customizable layers, activation functions, training via backpropagation, and inference through a clean and intuitive API.

## Features

- Flexible neural network configuration.
- Support for common activation functions such as Linear, Sigmoid, and None.
- Training support with adjustable hyperparameters.
- Simple API for training, inference, and error computation.

## Example Usage

Below is an example of how to use the library to create, train, and evaluate a neural network:

```cpp
#include "configClasses/dataPoint.hpp"
#include "configClasses/hyperParameters.hpp"
#include "configClasses/neuralNetworkConfig.hpp"
#include "neuralNetwork/NeuralNetwork.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>

NeuralNetwork createNeuralNetwork() {
    NeuralNetworkConfig nnConfig;

    // Define the structure of the neural network
    nnConfig.addLayer({2, "None"});     // Input layer with 2 neurons
    nnConfig.addLayer({3, "Sigmoid"}); // Hidden layer with 3 neurons and Sigmoid activation
    nnConfig.addLayer({3, "Sigmoid"}); // Hidden layer with 3 neurons and Sigmoid activation
    nnConfig.addLayer({1, "Linear"});  // Output layer with 1 neuron and Linear activation

    // Set hyperparameters for training
    HyperParameters hyperParameters;
    hyperParameters.learningRate = 0.1;
    nnConfig.setHyperParameters(hyperParameters);

    return NeuralNetwork(nnConfig);
}

int main() {
    // Create the neural network
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(createNeuralNetwork());

    // Define the dataset for a simple OR problem
    std::vector<DataPoint> dataset {
        {{0, 0}, {0}}, // Input: {0, 0}, Expected Output: {0}
        {{1, 0}, {1}}, // Input: {1, 0}, Expected Output: {1}
        {{0, 1}, {1}}, // Input: {0, 1}, Expected Output: {1}
        {{1, 1}, {1}}  // Input: {1, 1}, Expected Output: {1}
    };

    unsigned int epoch = 10000;
    for (unsigned int i = 0; i < epoch; i++) {
        double avgError = 0.0;

        // Shuffle the dataset for each epoch
        std::shuffle(dataset.begin(), dataset.end(), std::default_random_engine{});

        // Train the neural network
        nn->learn(dataset);

        // Compute the average error
        for (DataPoint dp : dataset) {
            nn->computeOutput(dp.inputs);
            avgError += nn->computeError(dp.expectedOutputs);
        }
        
        // Log the average error for the epoch
        std::cout << "Average error at epoch " << i << " is " << avgError / dataset.size() << std::endl;
    }

    // Output final predictions
    std::cout << "\nFinal predictions:" << std::endl;
    for (DataPoint dp : dataset) {
        std::cout << "Inputs {" << dp.inputs[0] << ", " << dp.inputs[1] 
                  << "} Output: {" << nn->computeOutput(dp.inputs)[0] 
                  << "} Expected: {" << dp.expectedOutputs[0] << "}" << std::endl;
    }
}
```

## How It Works

### Neural Network Configuration

The neural network is configured by adding layers using the `NeuralNetworkConfig` class. Each layer specifies the number of neurons and the activation function.

Example configuration:

```cpp
nnConfig.addLayer({2, "None"});    // Input layer
nnConfig.addLayer({3, "Sigmoid"}); // Hidden layer
nnConfig.addLayer({1, "Linear"});  // Output layer
```

### Training with Backpropagation

The `learn` method trains the neural network on a dataset. Hyperparameters such as the learning rate can be adjusted using the `HyperParameters` class.

### Forward Pass and Error Computation

The `computeOutput` method calculates the network's output for given inputs, while `computeError` determines the error relative to the expected outputs.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/emartin2109/my-ml-library
   ```

2. Include the library headers in your C++ project.

## Requirements

- A C++ compiler with C++17 or later support.
- Standard Template Library (STL).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
