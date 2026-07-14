from random import randint

import tqdm

import torch
from models import *

def pytorch_model_to_c_mnn(model, packet_meta_data: dict = None):
    c_code = ""
    if packet_meta_data:
        c_code += f"#define MNN_MAX_PACKET_SIZE {packet_meta_data.get('max_packet_size', 1518)}\n"
        c_code += f"#define MNN_MIN_PACKET_SIZE {packet_meta_data.get('min_packet_size', 64)}\n"
        c_code += f"#define MNN_MAX_TTL {packet_meta_data.get('max_ttl', 255)}\n"
        c_code += f"#define MNN_MIN_TTL {packet_meta_data.get('min_ttl', 0)}\n"
        c_code += f"#define MNN_MAX_PROTOCOL {packet_meta_data.get('max_protocol', 255)}\n"
        c_code += f"#define MNN_MIN_PROTOCOL {packet_meta_data.get('min_protocol', 0)}\n"
        c_code += f"#define MNN_MAX_SRC_PORT {packet_meta_data.get('max_src_port', 65535)}\n"
        c_code += f"#define MNN_MIN_SRC_PORT {packet_meta_data.get('min_src_port', 0)}\n"
        c_code += f"#define MNN_MAX_DST_PORT {packet_meta_data.get('max_dst_port', 65535)}\n"
        c_code += f"#define MNN_MIN_DST_PORT {packet_meta_data.get('min_dst_port', 0)}\n"
        c_code += f"#define MNN_MAX_TCP_FLAGS {packet_meta_data.get('max_tcp_flags', 255)}\n"
        c_code += f"#define MNN_MIN_TCP_FLAGS {packet_meta_data.get('min_tcp_flags', 0)}\n"
        c_code += f"#define MNN_MAX_PAYLOAD_SIZE {packet_meta_data.get('max_payload_size', 1500)}\n"
        c_code += f"#define MNN_MIN_PAYLOAD_SIZE {packet_meta_data.get('min_payload_size', 0)}\n"
        c_code += f"#define MNN_MAX_TCP_WINDOW {packet_meta_data.get('max_tcp_window', 65535)}\n"
        c_code += f"#define MNN_MIN_TCP_WINDOW {packet_meta_data.get('min_tcp_window', 0)}\n"

    i = 0
    for layer in model.layer:
        if isinstance(layer, torch.nn.Linear):
            weights = layer.weight.detach().numpy().flatten()
            biases = layer.bias.detach().numpy()
            c_code += f"/* Layer {i}: {layer.in_features} Inputs → {layer.out_features} Outputs */\n"
            c_code += f"float w{i}[] = {{{', '.join(map(str, weights))}}};\n"
            c_code += f"float b{i}[] = {{{', '.join(map(str, biases))}}};\n\n"
            i += 1

    c_code += "static MicroNeuralNetwork mnn = {0};\n"
    c_code += f"mnn.num_layers = {i};\n\n"
    
    j = 0
    for layer in model.layer:
        if isinstance(layer, torch.nn.Linear):
            c_code += f"mnn.layers[{j}].input_size = {layer.in_features};\n"
            c_code += f"mnn.layers[{j}].output_size = {layer.out_features};\n"
            c_code += f"memcpy(mnn.layers[{j}].weights, w{j}, sizeof(w{j}));\n"
            c_code += f"memcpy(mnn.layers[{j}].biases, b{j}, sizeof(b{j}));\n\n"
            j += 1

    return c_code

def train_model(x, y, model, epochs=1000, learning_rate=0.01):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    for epoch in tqdm.tqdm(range(epochs), desc="Training", unit="epoch"):
        
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
                param.grad.zero_()


def main():
    m = AdditionNet()

    x = [[randint(0, 9), randint(0, 9)] for _ in range(100)]
    y = [[a[0] + a[1]] for a in x]
    
    train_model(x, y, m, epochs=10000, learning_rate=0.01)

    print("Test input: ", torch.tensor([[1.0, 5.0]]))
    print("Test output: ", m(torch.tensor([[1.0, 5.0]])))

    #print(pytorch_model_to_c_mnn(m))


if __name__ == "__main__":
    main()
