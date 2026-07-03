#include "../mnn.h"


int main() {
    MicroNeuralNetwork *mnn;
    int layer_sizes[] = {3, 5, 2, 1}; // Example:
    mnn = create_mnn(2, layer_sizes);

    printf("Micro Neural Network created with %d layers.\n", mnn->num_layers);
    float input[] = {1.0, 2.0, 3.0}; // Example input
    float output = inference(mnn, input);
    printf("Inference result: %f\n", output);
}