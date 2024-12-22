#pragma once

#include "../layer/concrete/FCL.hpp"

#include <cstddef>
#include <functional>
#include <vector>

class NeuralNetwork {
    public:
        NeuralNetwork(std::vector<int> layers_sizes, std::vector<std::function<double(double)>> activationFunctions, 
        std::vector<std::function<double(double)>> derivateActivationFunctions, double learningRate) {
            int prevLayerSize = 0;
            for (size_t i = 0; i < layers_sizes.size(); i++) {
                m_layers.emplace_back(FullyConnectedLayer(layers_sizes[i], prevLayerSize, activationFunctions[i], derivateActivationFunctions[i]));
                prevLayerSize = layers_sizes[i];
            }

            m_learningRate = learningRate;
                
        }

        ~NeuralNetwork() = default;
    
    public:
        std::vector<double> computeOutput(std::vector<double> inputs) {
            if (m_layers.size() == 0) return {};

            m_layers[0].setOutput(inputs);
            for (size_t i = 1; i < m_layers.size(); i++)
                m_layers[i].computeOutputs(m_layers[i - 1].getOutput());

            return m_layers[m_layers.size() - 1].getOutput();
        }

        double computeError(std::vector<double> inputs, std::vector<double> expected) {
            // need to recompute the predicted output here
            computeOutput(inputs);
            return m_layers[m_layers.size() - 1].computeError(expected);
        }

        double computeErrorDerivative(std::vector<double> inputs, std::vector<double> expected) {
            // need to recompute the predicted output here
            computeOutput(inputs);
            return m_layers[m_layers.size() - 1].computeErrorDerivative(expected);
        }

        void updateAllGradients(std::vector<double> inputs, std::vector<double> expectedOutputs) {
            computeOutput(inputs);

            FullyConnectedLayer outputLayer = m_layers[m_layers.size() - 1];
            std::vector<double> nodeValues = outputLayer.computeNodeValues(expectedOutputs);
        }

        // for the time being learning is done through gradient descent (using finite-difference method)
        // replace that with derivative function for the future for better performance and lisibility
        void learn(std::vector<double> inputs, std::vector<double> excpected) {
            const double h = 0.0001;
            double originalCost = computeError(inputs, excpected);

            for (size_t i = 0; i < m_layers.size(); i++) {
                m_layers[i].costGradientBias.clear();
                m_layers[i].costGradientWeights.clear();
                for (size_t j = 0; j < m_layers[i].neurons.size(); j++) {
                    m_layers[i].costGradientWeights.emplace_back();
                    std::vector<double> originalWeights = m_layers[i].neurons[j].getWeights();
                    for (size_t k = 0; k < originalWeights.size(); k++) {
                        originalWeights[k] += h;
                        m_layers[i].neurons[j].setWeights(originalWeights);
                        double deltaCost = computeError(inputs, excpected) - originalCost;

                        originalWeights[k] -= h;
                        m_layers[i].neurons[j].setWeights(originalWeights);

                        m_layers[i].costGradientWeights[j].emplace_back(deltaCost / h);
                    }
                }
                for (size_t j = 0; j < m_layers[i].neurons.size(); j++) {
                    double originalBias = m_layers[i].neurons[j].getBias();
                    m_layers[i].neurons[j].setBias(originalBias + h);
                    double deltaCost = computeError(inputs, excpected) - originalCost;
                    m_layers[i].neurons[j].setBias(originalBias);

                    m_layers[i].costGradientBias.emplace_back(deltaCost / h);
                }
            }

            for (size_t i = 0; i < m_layers.size(); i++) {
                m_layers[i].applyGradients(m_learningRate);
            }
        }

    private:
        double m_learningRate;
        std::vector<FullyConnectedLayer> m_layers;
};
