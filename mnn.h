#include "stdio.h"
#include "stdlib.h"
#include "time.h"

typedef struct {
    int input_size;
    int output_size;
    float *weights;
    float *biases;
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
} MicroNeuralNetwork;

float relu(float x) {
    return x > 0 ? x : 0;
}

float random_normal() {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    return ((float)rand() / RAND_MAX) * 2 - 1; // Random float between -1 and 1
}

MicroNeuralNetwork* create_mnn(int num_layers, int *layer_sizes) {
    MicroNeuralNetwork *mnn = (MicroNeuralNetwork *)malloc(sizeof(MicroNeuralNetwork));
    mnn->num_layers = num_layers;
    mnn->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    
    for (int i = 0; i < num_layers; i++) {
        mnn->layers[i].input_size = layer_sizes[i];
        mnn->layers[i].output_size = layer_sizes[i + 1];
        mnn->layers[i].weights = (float *)malloc(layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));
        mnn->layers[i].biases = (float *)malloc(layer_sizes[i + 1] * sizeof(float));
        
        // Initialize weights and biases (for simplicity, using random values)
        for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
            mnn->layers[i].weights[j] = random_normal(); // Random weights between -1 and 1
        }
        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            mnn->layers[i].biases[j] = random_normal(); // Random biases between -1 and 1
        }
    }
    
    return mnn;
}

float* inference(MicroNeuralNetwork *mnn, float *input) {
    float *current_input = input;
    for (int i = 0; i < mnn->num_layers; i++) {
        Layer *layer = &mnn->layers[i];
        printf("Layer %d: input size = %d, output size = %d\n", i, layer->input_size, layer->output_size);
        float *output = (float *)malloc(layer->output_size * sizeof(float));
        for (int j = 0; j < layer->output_size; j++) {
            output[j] = layer->biases[j];
            for (int k = 0; k < layer->input_size; k++) {
                output[j] += current_input[k] * layer->weights[j * layer->input_size + k];
            }
            output[j] = relu(output[j]);
        }
        current_input = output;
    }
    return current_input; // Return the output of the last layer
}
