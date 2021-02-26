import torch
import string

from model import RNN
from train import load_katakana


def char_tensor(string, all_characters):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor


def generate(model, initial_str="A", predict_char_len=100, temprature=0.85):
    hidden, cell = model.init_hidden(batch_size=self.batch_size)
    initial_input = self.char_tensor(initial_str)
    predicted = initial_str

    for p in range(len(initial_str) - 1):
        _, (hidden, cell) = model(initial_input[p].view(1).to(device), hidden, cell)

    last_char = initial_input[-1]

    for p in range(predict_char_len):
        output, (hidden, cell) = model(last_char.view(1).to(device), hidden, cell)
        output_dist = output.data.view(-1).div(temprature).exp()
        top_char = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_char]
        predicted += predicted_char
        last_char = self.char_tensor(predicted_char)

    return predicted


class Generator:
    def __init__(self):
        pass

    def char_tensor(self):
        pass

    def generate(self):
        pass


def main():
    all_katakana = load_katakana()
    all_characters = all_katakana + list(string.printable)

    n_characters = len(all_characters)
    hidden_size = 256
    num_layers = 2
    model = RNN(n_characters, hidden_size, num_layers, n_characters)
    model.load_state_dict(torch.load("../model/v1_state_dict.pth"))

    first_char = random.choice(all_characters)
    gen_names = generate(model, first_char)


if __name__ == "__main__":
    main()
