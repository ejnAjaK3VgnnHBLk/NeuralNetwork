#include "global_neuralnetwork.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>

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

class TrainingData {
    public:
        TrainingData(const string filename);
        bool isEof(void) { return p_file.eof(); }
        
        // Number of inputs read from the file
        uint getInputs(vector<double> &inputVals);
        uint getGroundTruth(vector<double> &targetOutputVals);
        
        void genData(const string filename);
    private:
        ifstream p_file;
};

TrainingData::TrainingData(const string filename) {
    p_file.open(filename.c_str());
}

uint TrainingData::getInputs(vector<double> &inputVals) {
    inputVals.clear();
    
    string line;
    getline(p_file, line);
    stringstream ss(line);
    
    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double value;
        while (ss >> value) 
            inputVals.push_back(value);
    }
    return inputVals.size();
}

uint TrainingData::getGroundTruth(vector<double> &targetOutputVals) {
    targetOutputVals.clear();
    string line;
    getline(p_file, line);
    stringstream ss(line);
    
    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double value;
        while (ss >> value)
            targetOutputVals.push_back(value);
    }
    return targetOutputVals.size();
}

void TrainingData::genData(const string filename) {
    srand((uint) time(NULL));
    ofstream outFile(filename);
    for (int i = 0; i<500; i++) {
        int first = (int)(2.0 * rand() / double(RAND_MAX));
        int second = (int)(2.0 * rand() / double(RAND_MAX));
        int third = first ^ second;
        outFile << "in: " << first << ".0 " << second << ".0 " << endl;
        outFile << "out: " << third << ".0 " << endl;
    }
}

void printWeights(string label, vector<double> &v) {
    cout << label << " ";
    for (uint i = 0; i<v.size(); i++) 
        cout << v[i] << " ";
    cout<<endl;
}

int main() {
    TrainingData train("xor.txt");
    train.genData("xor.txt");
    
    vector<uint> topology = {2, 4, 1};
    
    NeuralNetwork net(topology);
    
    vector<double> inputVals, targetVals, resultVals;
    int epoch = 0;
    
    while (!train.isEof()) {
        epoch++;
        cout << endl << "Pass " << epoch;
        
        if (train.getInputs(inputVals) != topology[0]) 
            break;
        // Forward prop the information we just got
        printWeights(": Inputs:", inputVals);
        net.forwardProp(inputVals);
        
        // Get the outputs
        net.getResults(resultVals);
        printWeights("Outputs:", resultVals);
        
        // Backprop on this
        train.getGroundTruth(targetVals);
        printWeights("Targets:", targetVals);
        assert(targetVals.size() == topology.back());
        net.backProp(targetVals);
        
        cout << "Net average error: " << net.getAvgError() << endl;
    }
    cout << endl << "done" << endl;

    return 0;
    
}
