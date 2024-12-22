#include <array>
#include <functional>
#include <map>
#include <string>

double stepFunction(double x);
double linear(double x);
double sigmoid(double x);
double none(double x);

double sigmoidDerivative(double x);

const std::map<std::string, std::array<std::function<double(double)>, 2>> ACTIVATION_FUNCTIONS_LIST {
    {"None", {none, none}},
    {"Step Function", {stepFunction, none}},
    {"Linear", {linear, none}},
    {"Sigmoid", {sigmoid, sigmoidDerivative}},
};
