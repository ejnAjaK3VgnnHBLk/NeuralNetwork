#include "global_neuralnetwork.hpp"

#include <iostream>
#include <fstream>

using namespace std;

/*
 * Testing data should be in the form:
 * (aka xor operation)
 *
 * x1   |   x2    |  y
 * ---------------------
 *  0   |    0    |  0
 *  1   |    0    |  1
 *  0   |    1    |  1
 *  1   |    1    |  0
 *
 *  Neural network should learn this dataset ^
 */

int main() {
    NeuralNetwork net = NeuralNetwork({2, 4, 1});

    return 0;
    
}
