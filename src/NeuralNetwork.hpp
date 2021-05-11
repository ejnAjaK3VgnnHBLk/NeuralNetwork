#ifndef __nn_h_
#define __nn_h_

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Eigen>

typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef Eigen::MatrixXf Matrix;

class NeuralNetwork{
    public:
        NeuralNetwork(std::vector<uint> topology, uint biasNeurons, float learningRate);
        // Forward propagation datastructures
        std::vector<RowVector*> activationValues;
        std::vector<Matrix*> weights;

        //Backprop datastructs
        std::vector<RowVector*> loss;
        std::vector<RowVector*> deltas;

        float activationFn(float x);
        float activationFnDeriv(float x);
        float lossFnDeriv(float z, float delta);

        void ForwdProp();

        std::vector<uint> topology;
        float learningRate;
        uint biasNeurons;
};



#endif
