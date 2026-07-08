#include <string.h>
#include "../mnn.h"

int main() {
    /* Layer 0: 2 Inputs → 4 Outputs  (weights: 2*4 = 8 Werte, flach) */
    float w0[] = {0.1f, 0.2f, 0.3f, 0.4f,
                  0.5f, 0.6f, 0.7f, 0.8f};
    float b0[] = {0.1f, 0.2f, 0.3f, 0.4f};

    /* Layer 1: 4 Inputs → 2 Outputs  (weights: 4*2 = 8 Werte, flach) */
    float w1[] = {0.1f, 0.2f, 0.3f, 0.4f,
                  0.5f, 0.6f, 0.7f, 0.8f};
    float b1[] = {0.5f, 0.6f};

    /* Layer/weights are now fixed arrays inside the struct – copy values in. */
    MicroNeuralNetwork mnn = {0};
    mnn.num_layers = 2;

    mnn.layers[0].input_size  = 2;
    mnn.layers[0].output_size = 4;
    memcpy(mnn.layers[0].weights, w0, sizeof(w0));
    memcpy(mnn.layers[0].biases,  b0, sizeof(b0));

    mnn.layers[1].input_size  = 4;
    mnn.layers[1].output_size = 2;
    memcpy(mnn.layers[1].weights, w1, sizeof(w1));
    memcpy(mnn.layers[1].biases,  b1, sizeof(b1));

    float input[] = {1.0f, 0.5f};
    float *out = inference(&mnn, input);

    printf("Output: %f, %f\n", out[0], out[1]);
    /* no free() – inference uses a static buffer */
}