#ifndef MNN_H
#define MNN_H

#include "stdio.h"
#include "time.h"
#include <stdint.h>

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

/* --------------------------------------------------------------------------
 * PCAP-style packet feature extraction
 *
 * These functions parse a raw Ethernet frame (packet content + length) and
 * return a single normalized value in [0, 1], matching the features produced
 * by model-generator/pcap-loader.py:
 *   packet_size, ttl, protocol, src_port, dst_port,
 *   tcp_flags, payload_size, tcp_window
 *
 * None of these functions are called anywhere yet; they are ready to be used
 * before feeding values into inference().
 * -------------------------------------------------------------------------- */

#define MNN_ETHERTYPE_IP 0x0800U
#define MNN_IP_PROTO_TCP 6U
#define MNN_IP_PROTO_UDP 17U

static inline uint16_t mnn_be16(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | (uint16_t)p[1];
}

typedef struct {
    int is_ip;
    int is_tcp;
    int is_udp;
    uint8_t ttl;
    uint8_t protocol;
    uint16_t total_length;
    uint16_t ip_header_len;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t tcp_flags;
    uint16_t tcp_window;
    uint16_t payload_size;
} mnn_packet_info_t;

static inline mnn_packet_info_t mnn_parse_packet(const uint8_t *packet, int len) {
    mnn_packet_info_t info = {0};

    if (!packet || len < 14) {
        return info;
    }

    uint16_t ether_type = mnn_be16(packet + 12);
    if (ether_type != MNN_ETHERTYPE_IP) {
        return info;
    }
    info.is_ip = 1;

    if (len < 14 + 20) {
        return info;
    }

    const uint8_t *ip = packet + 14;
    info.ip_header_len = (uint16_t)((ip[0] & 0x0FU) * 4U);
    info.total_length = mnn_be16(ip + 2);
    info.ttl = ip[8];
    info.protocol = ip[9];

    int transport_start = 14 + info.ip_header_len;
    if (transport_start >= len) {
        return info;
    }

    if (info.protocol == MNN_IP_PROTO_TCP && len >= transport_start + 20) {
        info.is_tcp = 1;
        const uint8_t *tcp = packet + transport_start;
        info.src_port = mnn_be16(tcp);
        info.dst_port = mnn_be16(tcp + 2);
        info.tcp_flags = tcp[13];
        info.tcp_window = mnn_be16(tcp + 14);
        uint16_t tcp_header_len = (uint16_t)(((tcp[12] >> 4) & 0x0FU) * 4U);
        int payload_start = transport_start + tcp_header_len;
        if (payload_start < len) {
            info.payload_size = (uint16_t)(len - payload_start);
        }
    } else if (info.protocol == MNN_IP_PROTO_UDP && len >= transport_start + 8) {
        info.is_udp = 1;
        const uint8_t *udp = packet + transport_start;
        info.src_port = mnn_be16(udp);
        info.dst_port = mnn_be16(udp + 2);
        uint16_t udp_len = mnn_be16(udp + 4);
        int payload_start = transport_start + 8;
        int payload_len = udp_len > 8 ? (int)udp_len - 8 : 0;
        int available = len - payload_start;
        info.payload_size = (uint16_t)(payload_len < available ? payload_len : available);
    }

    return info;
}

static inline float mnn_clamp_norm(float value, float min, float max) {
    if (max <= min) {
        return 0.0f;
    }
    if (value < min) {
        value = min;
    }
    if (value > max) {
        value = max;
    }
    return (value - min) / (max - min);
}

/* Einzelne normalisierte Features (Parameter: nur Paketcontent + Länge) */
static inline float normalize_packet_size(const uint8_t *packet, int len) {
    (void)packet;
    return mnn_clamp_norm((float)len, 0.0f, 1518.0f);
}

static inline float normalize_ttl(const uint8_t *packet, int len) {
    mnn_packet_info_t info = mnn_parse_packet(packet, len);
    if (!info.is_ip) {
        return 0.0f;
    }
    return mnn_clamp_norm((float)info.ttl, 0.0f, 255.0f);
}

static inline float normalize_protocol(const uint8_t *packet, int len) {
    mnn_packet_info_t info = mnn_parse_packet(packet, len);
    if (!info.is_ip) {
        return 0.0f;
    }
    return mnn_clamp_norm((float)info.protocol, 0.0f, 255.0f);
}

static inline float normalize_src_port(const uint8_t *packet, int len) {
    mnn_packet_info_t info = mnn_parse_packet(packet, len);
    if (!info.is_tcp && !info.is_udp) {
        return 0.0f;
    }
    return mnn_clamp_norm((float)info.src_port, 0.0f, 65535.0f);
}

static inline float normalize_dst_port(const uint8_t *packet, int len) {
    mnn_packet_info_t info = mnn_parse_packet(packet, len);
    if (!info.is_tcp && !info.is_udp) {
        return 0.0f;
    }
    return mnn_clamp_norm((float)info.dst_port, 0.0f, 65535.0f);
}

static inline float normalize_tcp_flags(const uint8_t *packet, int len) {
    mnn_packet_info_t info = mnn_parse_packet(packet, len);
    if (!info.is_tcp) {
        return 0.0f;
    }
    return mnn_clamp_norm((float)info.tcp_flags, 0.0f, 255.0f);
}

static inline float normalize_payload_size(const uint8_t *packet, int len) {
    mnn_packet_info_t info = mnn_parse_packet(packet, len);
    return mnn_clamp_norm((float)info.payload_size, 0.0f, 1500.0f);
}

static inline float normalize_tcp_window(const uint8_t *packet, int len) {
    mnn_packet_info_t info = mnn_parse_packet(packet, len);
    if (!info.is_tcp) {
        return 0.0f;
    }
    return mnn_clamp_norm((float)info.tcp_window, 0.0f, 65535.0f);
}

/* Hilfsfunktion: füllt ein 8-elementiges Array mit allen normalisierten Features */
static inline void extract_normalized_packet_features(const uint8_t *packet, int len, float features[8]) {
    features[0] = normalize_packet_size(packet, len);
    features[1] = normalize_ttl(packet, len);
    features[2] = normalize_protocol(packet, len);
    features[3] = normalize_src_port(packet, len);
    features[4] = normalize_dst_port(packet, len);
    features[5] = normalize_tcp_flags(packet, len);
    features[6] = normalize_payload_size(packet, len);
    features[7] = normalize_tcp_window(packet, len);
}

#endif /* MNN_H */