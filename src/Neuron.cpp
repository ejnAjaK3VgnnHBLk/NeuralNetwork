#include "global_neuralnetwork.hpp"

double Neuron::learningRate = 0.15; // Learning rate
double Neuron::alpha = 0.5; // Momentum rate

void Neuron::updateInputWeights(Layer &prevLayer) {
	for (uint n = 0; n<prevLayer.size(); n++) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaW = neuron.p_outputWeights[p_neuronIndex].deltaWeight;

		// This is an absolute mouthfull so let's break it down (so i can understand it as a write lol)
		double newDeltaW = 
			// The individual input is increased by the gradient and learning rate
			learningRate * neuron.getOutputVal() * p_gradient
			// Then there is momentum = fraction of previous delta (alpha is momentum rate)
			* alpha * oldDeltaW;
		neuron.p_outputWeights[p_neuronIndex].deltaWeight = newDeltaW;
		neuron.p_outputWeights[p_neuronIndex].weight += newDeltaW;
	}
}

double Neuron::sumDerivOfWeight(const Layer &nextLayer) const {
	double sum = 0.0;
	for(uint n = 0; n<nextLayer.size() -1; n++) 	// Essentially just sum contributions of errors of nodes that are 
							// fed by us.
		sum += p_outputWeights[n].weight * nextLayer[n].p_gradient;
	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
	double sumDerivOfWeight = Neuron::sumDerivOfWeight(nextLayer);
	p_gradient = sumDerivOfWeight * Neuron::activationFnDeriv(p_outputVal);
}

void Neuron::calcOutputGradients(double targetVal) {
	double delta = targetVal - p_outputVal;
	p_gradient = delta * Neuron::activationFnDeriv(p_outputVal);
}

double Neuron::activationFn(double x) { return tanh(x); }
double Neuron::activationFnDeriv(double x) { return 1 - x*x; } 	// Note that this isn't the actual derivative of
								// tanh, but it's close enough in our range

void Neuron::forwardProp(const Layer &prevLayer) {
	double sum;
	
	// Loop through all previous layer output and get weighted sum
	for (uint n = 0; n<prevLayer.size(); n++) {
		sum += prevLayer[n].getOutputVal() * prevLayer[n].p_outputWeights[p_neuronIndex].weight;
	}

	p_outputVal = Neuron::activationFn(sum);
}

Neuron::Neuron(uint numOutputs, uint neuronIndex) {
	for (uint c = 0; c<numOutputs; c++) {
		p_outputWeights.push_back(Connection());
		p_outputWeights.back().weight = randomWeight();
	}
	p_neuronIndex = neuronIndex;
}
