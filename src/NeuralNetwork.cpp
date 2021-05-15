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
    for (uint activationIt = 1; activationIt<activationValues.size()-1; activationIt++) { // Iterate through each layer in the network
	uint activationColIt = 0;
	if (activationIt <= this->biasNeurons)  // Do we have a bias node?
	    activationColIt++;
	for ( ; activationColIt<activationValues[activationIt]->cols(); activationColIt++) // Iterate through each node in the layer.
	    this->activationValues[activationIt][activationColIt] = activationFn(this.activationValues[activationIt-1].dot(this->weights[activationColIt]));
    }
}
