#include "global_neuralnetwork.hpp"

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
