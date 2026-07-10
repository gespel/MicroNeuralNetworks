import torch
from models import *

def pytorch_model_to_c_mnn(model):
    c_code = ""
    i = 0
    for layer in model.layer:
        if isinstance(layer, torch.nn.Linear):
            weights = layer.weight.detach().numpy().flatten()
            biases = layer.bias.detach().numpy()
            c_code += f"/* Layer {i}: {layer.in_features} Inputs → {layer.out_features} Outputs */\n"
            c_code += f"float w{i}[] = {{{', '.join(map(str, weights))}}};\n"
            c_code += f"float b{i}[] = {{{', '.join(map(str, biases))}}};\n\n"
            i += 1
    return c_code

def main():
    m = AdditionNet()
    print(pytorch_model_to_c_mnn(m))


if __name__ == "__main__":
    main()
