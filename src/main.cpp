#include "global_neuralnetwork.hpp"

using namespace std;

void printWeights(string label, vector<double> &v) {
    cout << label << " ";
    for (uint i = 0; i<v.size(); i++) 
        cout << v[i] << " ";
    cout<<endl;
}

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr)
{
    arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    ifstream file ("/home/n1le/Desktop/swag_nn/mnist/train-images-idx3-ubyte",ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    } else 
        throw runtime_error("Unable to open images database!");
    file.close();
}

void ReadLabels(int NumberOfImages, int DataOfAnImage, vector<double> &arr) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };
    
    arr.resize(NumberOfImages);
    ifstream file ("/home/n1le/Desktop/swag_nn/mnist/train-labels-idx3-ubyte",ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

       // if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&NumberOfImages, sizeof(NumberOfImages)), NumberOfImages = reverseInt(NumberOfImages);

        for(int i = 0; i < NumberOfImages; i++) {
            unsigned char temp = 0;
            file.read((char*)&temp, 1);
            arr[i] = (double)temp;
        }
        
    } else 
        throw runtime_error("Unable to open labels database!");
        
    file.close();
}

int main() {
    TrainingData train("xor.txt");
    train.genData("xor.txt");
                             // 28x28, 14x14
    vector<uint> topology = {784, 196, 100, 25, 10};
    
    NeuralNetwork net(topology);
    vector<vector<double>> trainingData;
    vector<double> inputVals, targetVals, resultVals;
    
    uint numberOfImages = 5000;
    uint DataOfAnImage = 784;
    int epoch = 0;
     
     
    // Get training data and ground truth 
   ReadMNIST(numberOfImages, DataOfAnImage, trainingData); 
    ReadLabels(numberOfImages, DataOfAnImage, targetVals);
    
    while ((long unsigned int) epoch < trainingData.size()) { // trainingData.size() is number of images.
        epoch++;
        cout << endl << "Epoch " << epoch << endl;
        
        inputVals = trainingData[epoch - 1];
        
        // Forward prop the information we just got
        // printWeights(": Inputs:", inputVals);
        net.forwardProp(inputVals);
        
        // Get our current ground truth into a 0-9 indexed array. Everything 
        // should be 0 except for the ground truth.
        vector<double> currentTruth;
        double groundTruth = targetVals[epoch - 1];
        cout << "Ground Truth (apparently is): " << groundTruth << endl;
        for (uint i = 0; i <= 9; i++) {
            if (i == groundTruth)
                currentTruth.push_back(1.0);
            else
                currentTruth.push_back(0.0);
        }
        
        // Get the outputs
        net.getResults(currentTruth);
        printWeights("Outputs:", currentTruth);
        
        // Backprop on this
        printWeights("Targets:", currentTruth);
        assert(currentTruth.size() == topology.back());
        net.backProp(currentTruth);
        
        cout << "Net average error: " << net.getAvgError() << endl;
    }
    cout << endl << "done" << endl;

    return 0;
    
}
