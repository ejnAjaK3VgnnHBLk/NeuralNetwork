#ifndef _global_nn_hpp
#define _global_nn_hpp

#include <vector>
#include <iostream> 
#include <cstdlib>
#include <cassert>
#include <cmath>

typedef unsigned int uint;
using namespace std;

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

class NeuralNetwork {
	public:
		NeuralNetwork(const vector<uint> &topology);
		void forwardProp(const vector<double> &inputVals);
		void backProp(const vector<double> &targetVals);
		void getResults(vector<double> &resultValues) const; // Doesn't modify the net so it's a constant function

	private:
		vector<Layer> p_layers; 	// p_layers[layer#][node#]
                                    // Layer is vector<Neuron> so vector<vector<Neuron>>

		double p_error;
		double p_recentAvgError;
		double p_recentAverageSmoothingFactor;
};

class Neuron {
	public:
		Neuron(uint numOutputs, uint neuronIndex);
		void setOutputValue(double val) { p_outputVal = val; }
		double getOutputVal(void) const { return p_outputVal; }
		void forwardProp(const Layer &prevLayer);
		void calcOutputGradients(double targetVal);
		void calcHiddenGradients(const Layer &nextLayer);
		void updateInputWeights(Layer &prevLayer);
	private:
		static double activationFn(double x);
		static double activationFnDeriv(double x);
		static double randomWeight(void) { return rand() / double(RAND_MAX); }
		double sumDerivOfWeight(const Layer &nextLayer) const;
		double p_outputVal;
		vector<Connection> p_outputWeights;
		uint p_neuronIndex;
		double p_gradient;
		static double learningRate;
		static double alpha;
};

#endif
