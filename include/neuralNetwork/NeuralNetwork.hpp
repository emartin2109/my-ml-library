#pragma once

#include "../layer/concrete/FCL.hpp"
#include "../configClasses/neuralNetworkConfig.hpp"
#include "../activationFunctions/ActivationFunctions.hpp"
#include "../configClasses/dataPoint.hpp"

#include <cstddef>
#include <functional>
#include <vector>

class NeuralNetwork {
    public:
        NeuralNetwork(NeuralNetworkConfig nnConfig) {
            unsigned int prevLayerSize = 0;
            for (size_t i = 0; i < nnConfig.getSize(); i++) {
                const auto& lConfig = nnConfig.getLayers()[i];
                unsigned int layerSize = lConfig->neuronsNbr;

                if (ACTIVATION_FUNCTIONS_LIST.find(lConfig->activationFunction) == ACTIVATION_FUNCTIONS_LIST.end()) {
                    throw std::invalid_argument("Invalid activation function '" + lConfig->activationFunction + "' for layer " + std::to_string(i) + " in the Neural Network Factory");
                }

                std::function<double(double)> actiavtionFunction = ACTIVATION_FUNCTIONS_LIST.at(lConfig->activationFunction)[0];
                std::function<double(double)> derivativeActivationFunction = ACTIVATION_FUNCTIONS_LIST.at(lConfig->activationFunction)[1];


                m_layers.emplace_back(FullyConnectedLayer(layerSize,prevLayerSize,  actiavtionFunction, derivativeActivationFunction));
                prevLayerSize = layerSize;
            }

            m_learningRate = nnConfig.getHyperParameters()->learningRate;
        }

        ~NeuralNetwork() = default;
    
    public:
        std::vector<double> computeOutput(std::vector<double> inputs) {
            if (m_layers.size() == 0) return {};

            m_layers[0].setActivatedOutput(inputs);
            m_layers[0].setWeightedOutput(inputs);
            for (size_t i = 1; i < m_layers.size(); i++)
                m_layers[i].computeOutputs(m_layers[i - 1].getActivatedOutput(), m_layers[i - 1].getWeightedOutput());

            std::vector<double> output = m_layers[m_layers.size() - 1].getActivatedOutput();

            lastInput = inputs;
            lastOutput = output;

            return output;
        }

        double computeError(std::vector<double> expected) {
            return m_layers[m_layers.size() - 1].computeError(expected);
        }

        double computeErrorDerivative(std::vector<double> expected) {
            return m_layers[m_layers.size() - 1].computeErrorDerivative(expected);
        }

        void updateAllGradients(std::vector<double> inputs, std::vector<double> expectedOutputs) {
            computeOutput(inputs);

            // Backpropagation algorithm
            std::vector<double> nodeValues = m_layers[m_layers.size() - 1].computeNodeValues(expectedOutputs);
            m_layers[m_layers.size() - 1].updateGradients(nodeValues);

            for (int hiddenLayerI = static_cast<int>(m_layers.size()) - 2; hiddenLayerI >= 1; hiddenLayerI--) {
                nodeValues = m_layers[hiddenLayerI].calculateHiddenLayerNodeValues(m_layers[hiddenLayerI + 1], nodeValues);
                m_layers[hiddenLayerI].updateGradients(nodeValues);
            }
        }

        void applyAllGradients(double eta) {
            for (size_t i = 0; i < m_layers.size(); i++)
                m_layers[i].applyGradients(eta);
        }

        void clearAllGradients() {
            for (size_t i = 0; i < m_layers.size(); i++)
                m_layers[i].clearGradients();
        }

        void learn(std::vector<DataPoint> trainingBatch) {
            for (DataPoint dataPoint : trainingBatch)
                updateAllGradients(dataPoint.inputs, dataPoint.expectedOutputs);

            applyAllGradients(m_learningRate / trainingBatch.size());
            clearAllGradients();
        }

    private:
        double m_learningRate;
        std::vector<FullyConnectedLayer> m_layers;

        std::vector<double> lastInput;
        std::vector<double> lastOutput;
};
