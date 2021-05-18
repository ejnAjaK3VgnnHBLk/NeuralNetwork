#include "global_neuralnetwork.hpp"
#include <cstring>

using namespace std;

void printWeights(string label, vector<double> &v) {
    cout << label << " ";
    for (uint i = 0; i<v.size(); i++) 
        cout << v[i] << " ";
    cout<<endl;
}

class DataFileReader {
    public:
        DataFileReader(string inname, string outname);
        void getInputs(uint numPictures, vector<double> &arr);
        void getLabels(uint numLabels, vector<double> &arr);
        void getCurrentTruthArray(int epoch, vector<double> &outputVector, vector<double> &currentTruth);
        int reverseInt(int i);
        int getnImages() { return nImages; }

    private:
        string inFileName;
        string outFileName;
        
        ifstream inFile;
        ifstream outFile;

        const uint inMagic = 2051;
        const uint groundMagic = 2049;

        const uint blackThreshold = 128;

        int nImages;
};

DataFileReader::DataFileReader(string inname, string outname) {
    this->inFileName = inname;
    this->outFileName = outname;
    this->nImages = 0;
    inFile.open(this->inFileName, ios::in|ios::binary|ios::ate );
    outFile.open(this->outFileName, ios::in|ios::binary|ios::ate );
}

int DataFileReader::reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void DataFileReader::getInputs(uint numPictures, vector<double> &arr) {
    int magicNumber = 0, numImages = 0, numRows = 0, numCols = 0;
    int size = 0;

    inFile.seekg(0, ios::end); // set pointer to end of file
    size = inFile.tellg(); // length of file
  //  cout << "Size of file: " << size << endl;
    inFile.seekg(0, ios::beg); // set pointer to beginning of file

    // Magic number
    inFile.read((char*)&magicNumber, sizeof(this->inMagic));
    magicNumber = reverseInt(magicNumber);
  //  cout << "Magic number: " << magicNumber << endl;

    // Number of images
    inFile.read((char*)&numImages, sizeof(this->inMagic));
    numImages = reverseInt(numImages);
    this->nImages = numImages;
  //  cout << "Number of iamges: " << numImages << endl;

    // Row & cols
    inFile.read((char*)&numRows, sizeof(this->inMagic));
    inFile.read((char*)&numCols, sizeof(this->inMagic));
    numRows = reverseInt(numRows);
    numCols = reverseInt(numCols);
 //   cout << "Number of rows: " << numRows << ", number of cols: " << numCols << endl;

    // Read all the bytes
    for (int i = 0; i<numRows*numCols*numImages; i++) {
        int temp = 0;
        inFile.read((char*)&temp, 1);
        temp = (temp >= this->blackThreshold) ? temp = 1 : temp = 0; // Threshold
        arr.push_back(temp);
    }
}

void DataFileReader::getLabels(uint numLabels, vector<double> &arr) {
    int magicNumber = 0; 
    numLabels = 0;
    int size = 0;

    outFile.seekg(0, ios::end); // set pointer to end of file
    size = outFile.tellg(); // length of file
  //  cout << "Size of file: " << size << endl;
    outFile.seekg(0, ios::beg); // set pointer to beginning of file

    // Magic number
    outFile.read((char*)&magicNumber, sizeof(this->inMagic));
    magicNumber = reverseInt(magicNumber);
 //   cout << "Magic Number: " << magicNumber << endl;

    // Number of labels
    outFile.read((char*)&numLabels, sizeof(this->inMagic));
    numLabels = reverseInt(numLabels);
 //   cout << "Num labels: " << numLabels << endl;
    
    for(int i = 0; i<numLabels; i++){
        int temp = 0;
        outFile.read((char*)&temp, 1);
        arr.push_back(temp);
    }
}

void DataFileReader::getCurrentTruthArray(int epoch, vector<double> &outputVector, vector<double> &currentTruth) {
    currentTruth.clear();

    int truth = outputVector[epoch];
    for(int i = 0; i<=9; i++) {
        if (i == truth) 
            currentTruth.push_back(1.0);
        else
            currentTruth.push_back(0.0);
        // cout << "Truth: " << truth << ", truth array: " << currentTruth[i] << endl;
    }
}

int main() {
    vector<uint> topology = {784, 196, 100, 25, 10};
    //vector<uint> topology = {784, 10};
    int numRows= 28, numCols = 28, numImages = 10;

    DataFileReader dfr("/home/n1le/Desktop/swag_nn/mnist/train-images-idx3-ubyte", "/home/n1le/Desktop/swag_nn/mnist/train-labels-idx1-ubyte");
    NeuralNetwork net(topology);

    vector<double> inputVals, targetVals, resultVals;
    dfr.getInputs(5000, inputVals);
    dfr.getLabels(5000, targetVals);

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

            dfr.getCurrentTruthArray(i%(28*28), targetVals, asdf);
            assert(asdf.size() == topology.back());
            net.backProp(asdf);
        }
        cout << "net avg error: " << net.getAvgError() << endl;
    }
    cout << endl << "done" << endl;

    return 0;
    
}
