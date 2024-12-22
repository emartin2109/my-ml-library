#pragma once

#include "INeuron.hpp"

#include <iostream>
#include <string>

class ANeuron : public INeuron {
    public:
        ANeuron() = default;
        ~ANeuron() = default;
    
    // Debug
    public:
        friend std::ostream& operator<<(std::ostream& os, const ANeuron& neuron) {
            os << "a neuron of type " << neuron.type;
            return os;
        }
    
    protected:
        std::string type = "Default type";
};
