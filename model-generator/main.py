from random import randint

import tqdm

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

    c_code += "MicroNeuralNetwork mnn = {0};\n"
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

    for epoch in tqdm.tqdm(range(100000)):
        x = [[randint(0, 9), randint(0, 9)] for _ in range(100)]
        y = [a[0] + a[1] for a in x]  # Simple sum of inputs as target
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        loss = torch.nn.functional.mse_loss(m(x), y)
        loss.backward()
        with torch.no_grad():
            for param in m.parameters():
                param -= 0.01 * param.grad
                param.grad.zero_()

    print("Test input: ", torch.tensor([[1.0, 5.0]]))
    print("Test output: ", m(torch.tensor([[1.0, 5.0]])))

    print(pytorch_model_to_c_mnn(m))


if __name__ == "__main__":
    main()
