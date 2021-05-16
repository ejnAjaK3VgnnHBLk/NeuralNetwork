#include "global_neuralnetwork.hpp"


void NeuralNetwork::getResults(vector<double> &resultValues) const {
	resultValues.clear();

	for(uint n = 0; n<p_layers.back().size() - 1; n++)
		resultValues.push_back(p_layers.back()[n].getOutputVal());
}

void NeuralNetwork::backProp(const vector<double> &targetVals) {
	// Calculate entire network's error (Root mean square error here)
	Layer &outputLayer = p_layers.back();
	p_error = 0.0;
	for (uint n = 0; n<outputLayer.size() - 1; n++) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		p_error += delta*delta;
	}
	p_error /= outputLayer.size() - 1;
	p_error = sqrt(p_error); // final rms calculation

	// Recent average measurement. Thanks to someone who I can't remember for suggesting
	// this. (NOT MY CODE)
	p_recentAvgError = (p_recentAvgError * p_recentAverageSmoothingFactor + p_error) / (p_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradient
	for (uint n = 0; n<outputLayer.size() - 1; n++) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}
	// Calculate hidden layer gradients
	for (uint layerNum = p_layers.size() -2; layerNum > 0; layerNum--) {
		Layer &hiddenLayer = p_layers[layerNum];
		Layer &nextLayer = p_layers[layerNum + 1];

		for (uint n = 0; n<hiddenLayer.size(); n++) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	
	// From output to hidden layers, update weights
	for (uint layerNum = p_layers.size() - 1; layerNum > 0; layerNum--) {
		Layer &layer = p_layers[layerNum];
		Layer &prevLayer = p_layers[layerNum - 1];

		for (uint n = 0; n < layer.size() - 1; n++) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void NeuralNetwork::forwardProp(const vector<double> &inputVals) {
	assert(inputVals.size() == p_layers[0].size() - 1);

	// Latch input values into input neurons
	for (uint i = 0; i<inputVals.size(); i++) {
		p_layers[0][i].setOutputValue(inputVals[i]);
	}

	// Do the forward prop
	for (uint layerNum = 1; layerNum<p_layers.size(); layerNum++) {
		Layer &prevLayer = p_layers[layerNum -1];
		for (uint n = 0; n<p_layers[layerNum].size() - 1; n++) {
			p_layers[layerNum][n].forwardProp(prevLayer);
		}
	}
}

NeuralNetwork::NeuralNetwork(const vector<uint> &topology) {
	uint numLayers = topology.size();
	for (uint layerNum = 0; layerNum<numLayers; layerNum++) {
		p_layers.push_back(Layer());
		
		uint numOutputs;
		if(layerNum == topology.size() -1) // We're in output layer
			numOutputs = 0;
		else
			numOutputs = topology[layerNum + 1];
		
		// Add neurons
		for (uint neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) { 	// Note that we're adding a bias
												// neuron thus the <=
			p_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "made a neuron!!" << endl;
		}
		p_layers.back().back().setOutputValue(1.0);
	}
}
