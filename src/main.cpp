#include "configClasses/dataPoint.hpp"
#include "configClasses/hyperParameters.hpp"
#include "configClasses/neuralNetworkConfig.hpp"
#include "neuralNetwork/NeuralNetwork.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm> // For std::shuffle
#include <random>    // For std::default_random_engine


NeuralNetwork createNeuralNetwork() {
    NeuralNetworkConfig nnConfig;

    nnConfig.addLayer({2, "None"});
    nnConfig.addLayer({3, "Sigmoid"});
    nnConfig.addLayer({3, "Sigmoid"});
    nnConfig.addLayer({1, "Linear"});

    HyperParameters hyperParameters;
    hyperParameters.learningRate = 0.1;

    nnConfig.setHyperParameters(hyperParameters);

    return NeuralNetwork(nnConfig);
}

int main() {
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(createNeuralNetwork());

    std::vector<DataPoint> dataset {
        {{0, 0}, {0}},
        {{1, 0}, {1}},
        {{0, 1}, {1}},
        {{1, 1}, {1}}
    };

    unsigned int epoch = 10000;
    for (unsigned int i = 0; i < epoch; i++) {
        double avgError = 0.0;

        std::shuffle(dataset.begin(), dataset.end(), std::default_random_engine{});

        nn->learn(dataset);

        for (DataPoint dp : dataset) {
            nn->computeOutput(dp.inputs);
            avgError += nn->computeError(dp.expectedOutputs);
        }
        
        std::cout << "average error at epoch " << i << " is " << avgError / dataset.size() << std::endl;
    }

    std::cout << "\nFinal predictions:" << std::endl;
    for (DataPoint dp : dataset) {
        std::cout << "Inputs {" << dp.inputs[0] << ", " << dp.inputs[1] << "} Output: {" << nn->computeOutput(dp.inputs)[0] << "} Excpected: {" << dp.expectedOutputs[0] << "}" << std::endl;
    }
}
