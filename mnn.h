#ifndef MNN_H
#define MNN_H

#include "stdio.h"
#include "time.h"

/* Maximum number of layers and neurons per layer.
 * Override before including this header if needed:
 *   #define MNN_MAX_LAYERS      16
 *   #define MNN_MAX_LAYER_SIZE  128
 */
#ifndef MNN_MAX_LAYERS
#define MNN_MAX_LAYERS 8
#endif

#ifndef MNN_MAX_LAYER_SIZE
#define MNN_MAX_LAYER_SIZE 64
#endif

typedef struct {
    int input_size;
    int output_size;
    float weights[MNN_MAX_LAYER_SIZE * MNN_MAX_LAYER_SIZE];
    float biases[MNN_MAX_LAYER_SIZE];
} Layer;

typedef struct {
    int num_layers;
    Layer layers[MNN_MAX_LAYERS];
} MicroNeuralNetwork;

static inline float relu(float x) {
    return x > 0 ? x : 0;
}

static inline void load_weights_from_file(MicroNeuralNetwork *mnn, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    for (int i = 0; i < mnn->num_layers; i++) {
        Layer *layer = &mnn->layers[i];
        fread(layer->weights, sizeof(float), layer->input_size * layer->output_size, file);
        fread(layer->biases, sizeof(float), layer->output_size, file);
    }

    fclose(file);
}

/* Returns a pointer into a static double-buffer; no heap allocation.
 * NOTE: the returned pointer is only valid until the next inference() call. */
static inline float *inference(MicroNeuralNetwork *mnn, float *input) {
    /* Two alternating static buffers avoid any dynamic allocation. */
    static float buf[2][MNN_MAX_LAYER_SIZE];
    int cur = 0;
    float *current_input = input;

    for (int i = 0; i < mnn->num_layers; i++) {
        Layer *layer  = &mnn->layers[i];
        float *output = buf[cur];

        for (int j = 0; j < layer->output_size; j++) {
            output[j] = layer->biases[j];
            for (int k = 0; k < layer->input_size; k++) {
                output[j] += current_input[k] * layer->weights[j * layer->input_size + k];
            }
            output[j] = relu(output[j]);
        }
        current_input = output;
        cur ^= 1;
    }
    return current_input;
}

static inline void optimize(MicroNeuralNetwork *mnn, float *input, float *target, float learning_rate) {
    for (int i = mnn->num_layers - 1; i >= 0; i--) {
        Layer *layer  = &mnn->layers[i];
        float *output = inference(mnn, input);

        /* Basic gradient descent */
        for (int j = 0; j < layer->output_size; j++) {
            float error = output[j] - target[j];
            for (int k = 0; k < layer->input_size; k++) {
                layer->weights[j * layer->input_size + k] -= learning_rate * error * input[k];
            }
            layer->biases[j] -= learning_rate * error;
        }
    }
}

static inline float get_model_memory_size(MicroNeuralNetwork *mnn) {
    float size = sizeof(MicroNeuralNetwork);
    for (int i = 0; i < mnn->num_layers; i++) {
        size += mnn->layers[i].input_size * mnn->layers[i].output_size * sizeof(float); // weights
        size += mnn->layers[i].output_size * sizeof(float); // biases
    }
    return size / (1024 * 1024); // Return size in MB
}

static inline float normalize_packet(float value, float min, float max) {
    if (max - min == 0) {
        return 0.0f; // Avoid division by zero
    }
    return (value - min) / (max - min);
}

#endif /* MNN_H */