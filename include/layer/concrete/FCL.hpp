#pragma once

#include "../virtual/ALayer.hpp"
#include "../../neurons/concrete/Pereptron.hpp"

#include <cstddef>
#include <functional>
#include <ostream>
#include <vector>

class FullyConnectedLayer : public ALayer {
    public:
        FullyConnectedLayer(int layerSize, int inputSize, std::function<double(double)> activationFunction, std::function<double(double)> derivateActivationFunction) {
            costGradientWeights.resize(inputSize);
            costGradientBias.resize(layerSize);

            for (int i = 0; i < inputSize; i++)
                costGradientWeights[i].resize(layerSize);

            for (int i = 0; i < layerSize; i++) {
                neurons.emplace_back(inputSize);
                m_weightedOutputs.emplace_back(0);
                m_weightedInputs.emplace_back(0);
                m_activatedOutputs.emplace_back(0);
                m_activatedInputs.emplace_back(0);
            }

            m_activationFunction = activationFunction;
            m_derivateActivationFunction = derivateActivationFunction;

            std::cout << "layer of size " << layerSize << " created" << std::endl;
        };
        ~FullyConnectedLayer() = default;
    
    public:
        std::vector<double> computeOutputs(std::vector<double> activated_inputs, std::vector<double> weighted_inputs) {
            m_activatedInputs = activated_inputs;
            m_weightedInputs = weighted_inputs;

            for (size_t i = 0; i < neurons.size(); i++) {
                m_weightedOutputs[i] = neurons[i].computeOutput(m_activatedInputs);
                m_activatedOutputs[i] = m_activationFunction(m_weightedOutputs[i]);
            }

            return m_activatedOutputs;
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

        double neuronCost(double output, double expected) {
            double error = output - expected;
            return error * error;
        }

        double neuronCostDerivative(double outputs, double expected) {
            return 2 * (outputs - expected);
        }

        std::vector<double> computeNodeValues(std::vector<double> expected) {
            std::vector<double> nodeValues;

            for (size_t i = 0; i < neurons.size(); i++) {
                double costDerivative = neuronCostDerivative(m_activatedOutputs[i], expected[i]);
                double activationDeriavtive = m_derivateActivationFunction(m_weightedInputs[i]);
                nodeValues.push_back(costDerivative * activationDeriavtive);
            }

            return nodeValues;
        }

        std::vector<double> calculateHiddenLayerNodeValues(FullyConnectedLayer oldLayer, std::vector<double> oldNodeValues) {
            std::vector<double> newNodeValues(neurons.size());

            for (size_t newNodeI = 0; newNodeI < newNodeValues.size(); newNodeI++) {
                double newNodeValue = 0;
                for (size_t oldNodeI = 0; oldNodeI < oldNodeValues.size(); oldNodeI++) {
                    // maybe try inversing this idk
                    double weightedInputDerivative = oldLayer.getWeightAt(newNodeI, oldNodeI);
                    newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeI];
                }
                newNodeValue *= m_derivateActivationFunction(m_weightedInputs[newNodeI]);
                newNodeValues[newNodeI] = newNodeValue;
            }

            return newNodeValues;
        }

        void updateGradients(std::vector<double> nodeValues) {
            for (size_t nodeOut = 0; nodeOut < neurons.size(); nodeOut++) {
                double nodeValue = nodeValues[nodeOut];
                for (size_t nodeIn = 0; nodeIn < m_activatedInputs.size(); nodeIn++) {
                    // ERROR ON THE NEXT LINE
                    double derivativeCostWeight = m_activatedInputs[nodeIn] * nodeValue;
                    costGradientWeights[nodeIn][nodeOut] = derivativeCostWeight;
                }
                double derivativeCostBias = 1 * nodeValues[nodeOut];
                costGradientBias[nodeOut] += derivativeCostBias;
            }
        }

        void applyGradients(double learningRate) {
            for (size_t i = 0; i < neurons.size(); i++) {
                Pereptron neuron = neurons[i];

                double currentBias = neuron.getBias();
                neuron.setBias(currentBias - costGradientBias[i] * learningRate);

                std::vector<double> currentWeights = neuron.getWeights();

                for (size_t j = 0; j < currentWeights.size(); j++) {
                    currentWeights[j] -= costGradientWeights[j][i] * learningRate;
                }

                neuron.setWeights(currentWeights);

                neurons[i] = neuron;
            }
        }

        void clearGradients() {
            /*
            costGradientWeights.clear();
            costGradientBias.clear();
            */
            for (size_t i = 0; i < costGradientBias.size(); i++) {
                costGradientBias[i] = 0;
            }
            for (size_t i = 0; i < costGradientWeights.size(); i++) {
                for (size_t j = 0; j < costGradientWeights[i].size(); j++) {
                    costGradientWeights[i][j] = 0;
                }
            }
        }

        double getWeightAt(size_t nodeInIndex, size_t neuronIndex) {
            return neurons[neuronIndex].getWeights()[nodeInIndex];
        }


    public:
        void setActivatedOutput(std::vector<double> outputs) { m_activatedOutputs = outputs; }
        std::vector<double> getActivatedOutput() const { return m_activatedOutputs; }
        void setWeightedOutput(std::vector<double> outputs) { m_weightedOutputs = outputs; }
        std::vector<double> getWeightedOutput() const { return m_weightedOutputs; }
    
    public:
        std::vector<std::vector<double>> costGradientWeights;
        std::vector<double> costGradientBias;
        std::vector<Pereptron> neurons;

    protected:
        std::string type = "FullyConnectedLayer";

    private:
        std::vector<double> m_weightedOutputs;
        std::vector<double> m_weightedInputs;
        std::vector<double> m_activatedOutputs;
        std::vector<double> m_activatedInputs;
    
    private:
        std::function<double(double)> m_activationFunction;
        std::function<double(double)> m_derivateActivationFunction;
};
