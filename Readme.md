# Basic Machine Learning Library

This project is a simple and modular machine learning library written in C++ that enables users to configure and use neural networks with ease. The library supports customizable layers, activation functions, and inference through a clean and intuitive API.

## Features

- Flexible neural network configuration using a vector of layers.
- Support for common activation functions such as Linear and Sigmoid.
- Simple forward pass for output computation.

## Example Usage

Below is an example of how to use the library to create and execute a neural network:

```cpp
#include <iostream>
#include <vector>
#include <tuple>
#include <map>

#include "neuralNetwork/NeuralNetworkFactory.hpp"

int main() {
    // Define the configuration for the neural network
    std::vector<std::tuple<int, std::map<std::string, std::string>>> neuralNetworkConfig = {
        {2, {{"Activation Function", "Linear"}}},
        {10000, {{"Activation Function", "Sigmoid"}}},
        {10000, {{"Activation Function", "Sigmoid"}}},
        {2, {{"Activation Function", "Sigmoid"}}}
    };

    // Create a NeuralNetworkFactory instance with the configuration
    NeuralNetworkFactory nnf(neuralNetworkConfig);

    // Generate the neural network
    std::unique_ptr<NeuralNetwork> nn = nnf.create();

    // Compute the output for the given input
    std::vector<float> outputs = nn->computeOutput({3, 4});
    std::cout << "Outputs: " << outputs[0] << ", " << outputs[1] << std::endl;

    return 0;
}
```

## How It Works

### Neural Network Configuration

The neural network is configured using a vector of tuples, where each tuple represents a layer. Each tuple contains:

1. The number of neurons in the layer.
2. A map of parameters, such as the activation function to use.

Example configuration:

```cpp
{
    {2, {{"Activation Function", "Linear"}}},
    {10000, {{"Activation Function", "Sigmoid"}}},
    {10000, {{"Activation Function", "Sigmoid"}}},
    {2, {{"Activation Function", "Sigmoid"}}}
}
```

### Factory Pattern

The `NeuralNetworkFactory` class is responsible for creating neural network instances based on the provided configuration. This approach ensures flexibility and separation of concerns.

### Forward Pass

Once the neural network is created, you can use the `computeOutput` method to calculate the output for a given input vector. The method applies the configured layers and activation functions to compute the final result.

## Installation

1. Clone the repository:

   ```bash
   git clone "future repository url here"
   ```

2. Include the library headers in your C++ project.

## Requirements

- A C++ compiler with C++17 or later support.
- Standard Template Library (STL).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
