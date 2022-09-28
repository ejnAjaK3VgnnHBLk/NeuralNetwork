#include "global_neuralnetwork.hpp"

using namespace std;

void printWeights(string label, vector<double> &v) {
    cout << label << " ";
    for (uint i = 0; i<v.size(); i++) 
        cout << v[i] << " ";
    cout<<endl;
}


int main() {
    vector<uint> topology = {785, 196, 100, 25, 10};
    //vector<uint> topology = {784, 10};
    int numRows= 28, numCols = 28, numImages = 10;

    DataFileReader dfr("/users/n1le/Documents/NeuralNetwork/images", "/users/n1le/Documents/NeuralNetwork/labels");
    NeuralNetwork net(topology, true);

    vector<double> inputVals, targetVals, resultVals;
    dfr.getInputs(5, inputVals);
    dfr.getLabels(5, targetVals);

    int nImages = dfr.getnImages();
    int epoch = 0;

    vector<double> temp, asdf;
    for(int epoch = 0; epoch<10; epoch++) {
        for(int i = 0; i<dfr.getnImages()*28*28; i+=28*28) { // Each image
            temp.clear();
            for (int j = 0; j<28*28; j++) {
                temp.push_back(inputVals[i+j]);
            }
            net.forwardProp(temp);

            net.getResults(resultVals);

            dfr.getCurrentTruthArray(i%28*28, targetVals, asdf);
            assert(asdf.size() == topology.back());
            net.backProp(asdf);

           // cout << "net avg error: " << net.getAvgError() << endl;
        }
        cout << "net avg error: " << net.getAvgError() << endl;
    }
    cout << endl << "done" << endl; 

    return 0;
    
}