#include "global_neuralnetwork.hpp"

DataFileReader::DataFileReader(string inname, string outname) {
    this->inFileName = inname;
    this->outFileName = outname;
    this->nImages = 0;
    inFile.open(this->inFileName, ios::in|ios::binary|ios::ate );
    if(inFile.fail())
        cout << "Error opening infile!" << endl;
    outFile.open(this->outFileName, ios::in|ios::binary|ios::ate );
    if(outFile.fail())
        cout << "Error opening outfile" << endl;
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
    cout << "Size of file: " << size << endl;
    inFile.seekg(0, ios::beg); // set pointer to beginning of file

    // Magic number
    inFile.read((char*)&magicNumber, sizeof(this->inMagic));
    magicNumber = reverseInt(magicNumber);
    cout << "Magic number: " << magicNumber << endl;

    // Number of images
    inFile.read((char*)&numImages, sizeof(this->inMagic));
    numImages = reverseInt(numImages);
    this->nImages = numImages;
    cout << "Number of images: " << numImages << endl;

    if(numPictures > numImages) { throw runtime_error("Number of requested images exceeds number of iamges in dataset!"); }

    // Row & cols
    inFile.read((char*)&numRows, sizeof(this->inMagic));
    inFile.read((char*)&numCols, sizeof(this->inMagic));
    numRows = reverseInt(numRows);
    numCols = reverseInt(numCols);
    cout << "Rows: " << numRows << " cols: " << numCols << endl;

    for (int i = 0; i<numRows*numCols*numPictures; i++) {
        int temp = 0;
        inFile.read((char*)&temp, 1);
        temp = (temp >= this->blackThreshold) ? temp = 1 : temp = 0; // Threshold
        arr.push_back(temp);
    }
}

void DataFileReader::getLabels(uint numLabels, vector<double> &arr) {
    int magicNumber = 0, readLabels = 0, size = 0; 

    outFile.seekg(0, ios::end); // set pointer to end of file
    size = outFile.tellg(); // length of file
    outFile.seekg(0, ios::beg); // set pointer to beginning of file

    // Magic number
    outFile.read((char*)&magicNumber, sizeof(this->inMagic));
    magicNumber = reverseInt(magicNumber);

    // Number of labels
    outFile.read((char*)&readLabels, sizeof(this->inMagic));
    readLabels = reverseInt(readLabels);

    if(numLabels > readLabels) { throw runtime_error("Number of requested labels exceeds number of labels in dataset!"); }

    
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
    }
}
