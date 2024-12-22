#include "activationFunctions/ActivationFunctions.hpp"
#include <cmath>
#include <stdexcept>

double none(double x) {
    (void) x;
    throw std::runtime_error("Neuron with no activation function tried to make a prediction. please use linear activation functions if you want the neurons activation function to have no effect on the input instead");
}

double stepFunction(double x) {
    return x >= 0.0 ? 1.0 : 0.0;
}

double linear(double x) {
    return x;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
}
