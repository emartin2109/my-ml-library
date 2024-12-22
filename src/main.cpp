#include "configClasses/layerConfig.hpp"
#include "neuralNetwork/NeuralNetworkFactory.hpp"

#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>


int main() {
    std::vector<LayerConfig> neuralNetworkConfig = {
        LayerConfig({2, "None"}),
        LayerConfig({5, "Sigmoid"}),
        LayerConfig({1, "Linear"})
    };

    HyperParameters hyperParameters;
    hyperParameters.learningRate = 0.1;

    NeuralNetworkFactory nnf(neuralNetworkConfig, hyperParameters);

    std::unique_ptr<NeuralNetwork> nn = nnf.create();

    std::vector<std::vector<std::vector<double>>> datasets {
        {{0, 0}, {1, 0}, {0, 1}, {1, 1}},
        {{0}, {1}, {1}, {0}}
    };

    unsigned int epoch = 1000;
    for (unsigned int i = 0; i < epoch; i++) {
        double avgError = 0.0;

        for (size_t j = 0; j < datasets[0].size(); j++) {
            std::vector<double> output = nn->computeOutput(datasets[0][j]);
            avgError += nn->computeError(datasets[0][j], datasets[1][j]);
            nn->learn(datasets[0][j], datasets[1][j]);

            // show output
            std::cout << "Input: " << datasets[0][j][0] << ", " << datasets[0][j][1] << ", Output: " << output[0] << ", Expected: " << datasets[1][j][0] << std::endl;
        }
        
        std::cout << "Epoch: " << i << ", Average Error: " << avgError / datasets[0].size() << std::endl << std::endl;
    }

    return 0;
}
