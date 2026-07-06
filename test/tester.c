#include "../mnn.h"


int main() {
    MicroNeuralNetwork *mnn;
    int layer_sizes[] = {3, 16, 16, 1}; // Example:
    mnn = create_mnn(3, layer_sizes); // Number of layers is all layers - 1 (input layer is not counted)

    printf("Micro Neural Network created with %d layers.\n", mnn->num_layers);
    float input[] = {1.0, 2.0, 3.0}; // Example input
    float* output = inference(mnn, input);
    printf("Inference result: %f\n", output[0]);

    for (int gens = 0; gens < 1000; gens++) {
        optimize(mnn, (float[]){1, 2, 3}, (float[]){6}, 0.01);
    }

    printf("After optimization:\n");
    output = inference(mnn, input);
    printf("Inference result: %f\n", output[0]);
    
}