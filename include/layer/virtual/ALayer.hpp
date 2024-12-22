#pragma once

#include "ILayer.hpp"

#include <iostream>
#include <string>

class ALayer : public ILayer {
    public:
        ALayer() = default;
        ~ALayer() = default;
    
    // Debug
    public:
        friend std::ostream& operator<<(std::ostream& os, const ALayer& layer) {
            os << "a neuron of type " << layer.type;
            return os;
        }
    
    protected:
        std::string type = "Default type";
};
