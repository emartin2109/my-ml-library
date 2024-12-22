#pragma once

#include "../virtual/ANeuron.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

class Pereptron : public ANeuron {
    public:
        Pereptron(int inputLenght) {
            type = "Pereptron";

            for (int i = 0; i < inputLenght; i++) {
                m_weights.emplace_back(((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 2 - 1) / std::sqrt(inputLenght));
            }
    
            m_bias = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 2 - 1;  // random bias initialization between 0 and -1
        }
        ~Pereptron() = default;

    public:
        double computeOutput(const std::vector<double> inputs) {
            m_output = m_bias;

            if (inputs.size() != m_weights.size()) {
                throw std::invalid_argument("Input size does not match weights size for fcl. input = " + std::to_string(inputs.size()) + " weights = " + std::to_string(m_weights.size()));
            }

            for (size_t i = 0; i < inputs.size(); ++i) {
                m_output += inputs[i] * m_weights[i];
            }

            return m_output;
        }

        double computeError(double expected) {
            double error = m_output - expected;

            return error * error; // make the error positive and prioritize larger differences to correct in priority
        }

        double computeErrorDerivative(double expected) {
            return 2 * (m_output - expected);
        }   
    
    public:
        void setOutput(double output) { m_output = output; }
        double getOutput() const { return m_output; }

        double getBias() const { return m_bias; }
        std::vector<double> getWeights() const { return m_weights; }

        void setBias(double bias) { m_bias = bias; }
        void setWeights(const std::vector<double>& weights) { m_weights = weights; }
    
    private:
        std::vector<double> m_weights;
        double m_bias;

        double m_output = 0.0;
};
