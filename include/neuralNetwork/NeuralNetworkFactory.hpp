/*
** EPITECH PROJECT, 2024
** lobotomy-learning-v3
** File description:
** NeuralNetworkFactory
*/

#pragma once

#include "activationFunctions/ActivationFunctions.hpp"
#include "neuralNetwork/NeuralNetwork.hpp"
#include "configClasses/hyperParameters.hpp"
#include "configClasses/layerConfig.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class NeuralNetworkFactory {
    public:
        NeuralNetworkFactory(std::vector<LayerConfig> layerConfigs, HyperParameters hyperParameters) {
            for (size_t i = 0; i < layerConfigs.size(); i++) {
                const auto& layerConfig = layerConfigs[i];
                int layerSize = layerConfig.neuronsNbr;

                nn_shape.push_back(layerSize);

                if (ACTIVATION_FUNCTIONS_LIST.find(layerConfig.activationFunction) == ACTIVATION_FUNCTIONS_LIST.end()) {
                    throw std::invalid_argument("Invalid activation function '" + layerConfig.activationFunction + "' for layer " + std::to_string(i) + " in the Neural Network Factory");
                }

                actiavtionFunctions.push_back(ACTIVATION_FUNCTIONS_LIST.at(layerConfig.activationFunction)[0]);
                derivativeActiavtionFunctions.push_back(ACTIVATION_FUNCTIONS_LIST.at(layerConfig.activationFunction)[1]);
            }

            m_learningRate = hyperParameters.learningRate;
        }
        ~NeuralNetworkFactory() = default;
    
    public:
        std::unique_ptr<NeuralNetwork> create() {
            return std::make_unique<NeuralNetwork>(nn_shape, actiavtionFunctions, derivativeActiavtionFunctions, m_learningRate);
        };

    private:
        std::vector<int> nn_shape;
        std::vector<std::function<double(double)>> actiavtionFunctions;
        std::vector<std::function<double(double)>> derivativeActiavtionFunctions;
        double m_learningRate;
};
