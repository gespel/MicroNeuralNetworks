#include "../mnn.h"

int main() {
    // Layer 0: 2 Inputs → 4 Outputs  (weights: 2*4 = 8 Werte, flach)
    float w0[] = {0.1f, 0.2f, 0.3f, 0.4f,
                  0.5f, 0.6f, 0.7f, 0.8f};
    float b0[] = {0.1f, 0.2f, 0.3f, 0.4f};

    // Layer 1: 4 Inputs → 2 Outputs  (weights: 4*2 = 8 Werte, flach)
    float w1[] = {0.1f, 0.2f, 0.3f, 0.4f,
                  0.5f, 0.6f, 0.7f, 0.8f};
    float b1[] = {0.5f, 0.6f};

    Layer layers[] = {
        {.input_size = 2, .output_size = 4, .weights = w0, .biases = b0},
        {.input_size = 4, .output_size = 2, .weights = w1, .biases = b1},
    };

    MicroNeuralNetwork static_mnn = {
        .num_layers = 2,
        .layers = layers,
    };

    float input[] = {1.0f, 0.5f};
    float *out = inference(&static_mnn, input);

    printf("Output: %f, %f\n", out[0], out[1]);
    free(out);
}