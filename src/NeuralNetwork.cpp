#include "NeuralNetwork.hpp"

/* std::vector<std::vector<double>> = [[0,0,1,1],   // x1
 *                                   [0,1,0,1],   // x2
 *                                   [0,1,1,0]];  // y
*/

NeuralNetwork::NeuralNetwork(std::vector<uint> topology, uint biasNeurons, float learningRate) {
    // Constructor for all forward prop elements

    if ((unsigned int)(topology.size()-1) < biasNeurons) 
        throw std::invalid_argument("number of bias neurons is larger than topology!");

    this->topology = topology;
    this->learningRate = learningRate;
    this->biasNeurons = biasNeurons;

    for (uint i = 0; i<topology.size(); i++) {

        if (i <= biasNeurons) {
            activationValues.push_back(new RowVector(topology[i] + 1));     // Add a bias neuron that we can adust for the first two layers
            weights.push_back(new Matrix(topology[i]+1, topology[i+1]+1));   // Little bit confusing, we are making a matrix 
                                                                            // that is NxM where N is the number of units in 
                                                                            // our current column in the network and M is the next number of units.
                                                                            // since we are adding a bias neuron, we neeed to ass an extra unit
            loss.push_back(new RowVector(topology[i] + 1));
            deltas.push_back(new RowVector(topology[i] + 1));
        } else {
            activationValues.push_back(new RowVector(topology[i]));         // No bias unit this time.
            weights.push_back(new Matrix(topology[i], topology[i+1]));
            loss.push_back(new RowVector(topology[i]));
            deltas.push_back(new RowVector(topology[i]));
        }
        activationValues[i]->setZero();
        weights[i]->setRandom();
        loss[i]->setZero();
        deltas[i]->setZero();

    }
}

float NeuralNetwork::activationFn(float x) { return x; }
float NeuralNetwork::activationFnDeriv(float x) { return (float) 1; }
float NeuralNetwork::lossFnDeriv(float z, float delta) { return z*delta; }

void NeuralNetwork::ForwdProp()
{
    /*
     * For this function, we want to start with the first layer of the network
     * and move our way forward, computing the hypothesis function and then running
     * that through the activation function to get the z value for the current layer.
     * That z value will be fed forward through the use of weights to create the next z
     * value, which will happen until the final layer. 
     */
    float hypothesis = 0;
    // BIG TODO: Implement checking for being in the first or second layer becuase we have a bias neuron
    
    for (uint activationIterator = 1; activationIterator<activationValues.size(); activationIterator++) {
        for (uint weightColIterator = 0; weightColIterator<weights[activationValues-1].cols(); weightColIterator++) {
            this->activationValues[activationIterator] = this->activationValues[activationIterator - 1].dot(this->weights[activationIterator -1].col(weightColIterator));
        }
    }
}
