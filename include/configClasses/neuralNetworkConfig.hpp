#pragma once

#include "layerConfig.hpp"
#include "hyperParameters.hpp"
#include <cstddef>
#include <memory>
#include <vector>

class NeuralNetworkConfig {
    public:
        NeuralNetworkConfig(unsigned int layerNbr = 0) { m_layers.resize(layerNbr); };

    public:
        void setLayer(unsigned int layerIndex, const LayerConfig& config) { m_layers[layerIndex] = std::make_shared<LayerConfig>(config); };
        const std::vector<std::shared_ptr<LayerConfig>>& getLayers() const { return m_layers; };
    
    public:
        void setHyperParameters(const HyperParameters hyperParameters) { m_hyperParameters = std::make_shared<HyperParameters>(hyperParameters); };
        const std::shared_ptr<HyperParameters>& getHyperParameters() const { return m_hyperParameters; };
    
    public:
        void addLayer(LayerConfig lConfig) { m_layers.push_back(std::make_shared<LayerConfig>(lConfig)); };
        size_t getSize() const { return m_layers.size(); };
        std::vector<size_t> getShape() const {
            std::vector<size_t> shape;
            for (const auto& layer : m_layers)
                shape.push_back(layer->neuronsNbr);
            return shape;
        };
    
    private:
        std::vector<std::shared_ptr<LayerConfig>> m_layers;
        std::shared_ptr<HyperParameters> m_hyperParameters;
};