#pragma once

#include "../virtual/ALayer.hpp"
#include "../../neurons/concrete/Pereptron.hpp"

#include <cstddef>
#include <functional>
#include <vector>

class FullyConnectedLayer : public ALayer {
    public:
        FullyConnectedLayer(int layerSize, int inputSize, std::function<double(double)> activationFunction, std::function<double(double)> derivateActivationFunction) {
            for (int i = 0; i < layerSize; ++i) {
                neurons.emplace_back(inputSize);
                m_outputs.emplace_back(0);
                m_inputs.emplace_back(0);
            }
            m_activationFunction = activationFunction;
            m_derivateActivationFunction = derivateActivationFunction;

            std::cout << "layer of size " << layerSize << " created" << std::endl;
        };
        ~FullyConnectedLayer() = default;
    
    public:
        std::vector<double> computeOutputs(std::vector<double> inputs) {
            m_inputs = inputs;

            for (size_t i = 0; i < neurons.size(); i++) {
                m_outputs[i] = m_activationFunction(neurons[i].computeOutput(inputs));
            }

            return m_outputs;
        }

        double computeError(std::vector<double> expected) {
            double error = 0;

            for (size_t i = 0; i < neurons.size(); i++)
                error += neurons[i].computeError(expected[i]);

            return error;
        }

        double computeErrorDerivative(std::vector<double> expected) {
            double sum = 0;

            for (size_t i = 0; i < neurons.size(); i++)
                sum += neurons[i].computeErrorDerivative(expected[i]);

            return sum;
        }

        std::vector<double> computeNodeValues(std::vector<double> expected) {
            // remake thisfdffdfef

            return {};
        }

        void applyGradients(double learningRate) {
            for (size_t i = 0; i < neurons.size(); i++) {
                Pereptron neuron = neurons[i];

                double currentBias = neuron.getBias();
                neuron.setBias(currentBias - costGradientBias[i] * learningRate);

                std::vector<double> currentWeights = neuron.getWeights();

                for (size_t j = 0; j < currentWeights.size(); j++) {
                    currentWeights[j] = currentWeights[j] - costGradientWeights[i][j] * learningRate;
                }


                neuron.setWeights(currentWeights);

                neurons[i] = neuron;
            }
        }

    public:
        void setOutput(std::vector<double> outputs) { m_outputs = outputs; }
        std::vector<double> getOutput() const { return m_outputs; }
    
    public:
        std::vector<std::vector<double>> costGradientWeights;
        std::vector<double> costGradientBias;
        std::vector<Pereptron> neurons;

    protected:
        std::string type = "FullyConnectedLayer";

    private:
        std::vector<double> m_outputs;
        std::vector<double> m_inputs;
        std::function<double(double)> m_activationFunction;
        std::function<double(double)> m_derivateActivationFunction;
};
