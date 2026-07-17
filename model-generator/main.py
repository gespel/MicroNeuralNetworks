from random import randint
import numpy as np
import pandas as pd
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
        c_code += f"#define MNN_MIN_TCP_WINDOW {packet_meta_data.get('min_tcp_window', 0)}\n\n"

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
    progress_bar = tqdm.tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in progress_bar:
        
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
                param.grad.zero_()
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")

def main():
    m = NetworkAnomalyDetectionNet()

    data_csv = pd.read_csv("features.csv")

    print(f"[*] Loaded {len(data_csv)} rows from features.csv")
    print(f"[*] DataFrame Shape: {data_csv.shape}")
    print(f"[*] DataFrame Columns: {data_csv.columns.tolist()}")

    # Autoencoder: Eingabe == Ziel (Rekonstruktion der Netzwerk-Features)
    x = data_csv.to_numpy(dtype=np.float32)
    y = x.copy()

    print(f"[*] Training model with {x.shape[0]} samples and {x.shape[1]} features...")

    train_model(x, y, m, epochs=10000, learning_rate=0.0001)

    # Optional: Konvertiere trainiertes Modell in C-Code für mnn.h
    packet_meta_data = {
        "max_packet_size": int(data_csv["packet_size"].max()),
        "min_packet_size": int(data_csv["packet_size"].min()),
        "max_ttl": int(data_csv["ttl"].max()),
        "min_ttl": int(data_csv["ttl"].min()),
        "max_protocol": int(data_csv["protocol"].max()),
        "min_protocol": int(data_csv["protocol"].min()),
        "max_src_port": int(data_csv["src_port"].max()),
        "min_src_port": int(data_csv["src_port"].min()),
        "max_dst_port": int(data_csv["dst_port"].max()),
        "min_dst_port": int(data_csv["dst_port"].min()),
        "max_tcp_flags": int(data_csv["flags"].max()),
        "min_tcp_flags": int(data_csv["flags"].min()),
        "max_payload_size": int(data_csv["payload_size"].max()),
        "min_payload_size": int(data_csv["payload_size"].min()),
        "max_tcp_window": int(data_csv["tcp_window"].max()),
        "min_tcp_window": int(data_csv["tcp_window"].min()),
    }
    c_code = pytorch_model_to_c_mnn(m, packet_meta_data)
    with open("model.c", "w") as f:
        f.write(c_code)
    print("[+] C-Modell in model.c gespeichert.")


if __name__ == "__main__":
    main()
