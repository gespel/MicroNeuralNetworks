#include "stdio.h"

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


