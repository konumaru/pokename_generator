import torch
import string
import random

from model import RNN
from train import load_katakana

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator:
    def __init__(self, all_characters):
        self.all_characters = all_characters
        self.batch_size = 1

        hidden_size = 256
        num_layers = 2
        n_characters = len(all_characters)
        self.rnn = RNN(n_characters, hidden_size, num_layers, n_characters)
        self.rnn.load_state_dict(torch.load("../model/v1_state_dict.pth"))

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c])
        return tensor

    def generate(self, initial_str="ア", predict_char_len=100, temprature=0.85):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(
                initial_input[p].view(1).to(device), hidden, cell
            )

        last_char = initial_input[-1]

        for p in range(predict_char_len):
            output, (hidden, cell) = self.rnn(
                last_char.view(1).to(device), hidden, cell
            )
            output_dist = output.data.view(-1).div(temprature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = self.all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        # Drop names of 2 characters or less.
        names = predicted.split("\n")
        names = [n for n in names if len(n) > 2]
        names = random.choices(names, k=5)
        names = "\n".join(names)
        return names


def main():
    all_katakana = load_katakana()

    generator = Generator(all_katakana)

    initial_str = random.choice(all_katakana)
    gen_names = generator.generate(initial_str)

    print("Genereated:")
    print(gen_names)


if __name__ == "__main__":
    main()
