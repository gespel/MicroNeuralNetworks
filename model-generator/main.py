import torch


class TestNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.layer(x)

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

    c_code += "MicroNeuroalNetwork mnn = {0};\n"
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

def main():
    m = TestNet()
    print(pytorch_model_to_c_mnn(m))


if __name__ == "__main__":
    main()
