import re
import sys
import random
import string

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pokemon_name():
    data = open("../data/PokemonStatus_Gen8.csv").readlines()
    names = []
    for i in range(1, len(data)):
        name = data[i].split(",")[1]
        name = re.search(r"(\w+)(\w+)", name).group(1)  # NOTE: カッコで囲まれた補足情報を除外
        names.append(name.strip())
    names = "\n".join(names)
    return names


def load_katakana():
    katakana = [chr(i) for i in range(ord("ァ"), ord("ヺ") + 1)]
    katakana += ["\n", "ー", "A"]
    return katakana


class NameGenerateTrainer:
    def __init__(self, all_characters, name_list):
        self.all_characters = all_characters
        self.n_characters = len(all_characters)
        self.name_list = name_list
        self.chunk_len = 250
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_every = 50
        self.lr = 0.003

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c])
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(self.name_list) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = self.name_list[start_idx:end_idx]

        if len(text_str) != self.chunk_len:
            diff = self.chunk_len - len(text_str)
            text_str += "\n" * diff

        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

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

        return predicted

    def train(self):
        hidden_size = 256
        num_layers = 2

        self.rnn = RNN(
            self.n_characters, hidden_size, num_layers, self.n_characters
        ).to(device)

        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(f"runs/names0")

        print("=> Starting training")

        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0:
                print(f"Epoch {epoch} Loss: {loss}")
                print(self.generate())

            writer.add_scalar("Training Loss", loss, global_step=epoch)


def main():
    all_katakana = load_katakana()
    all_names = load_pokemon_name()

    trainer = NameGenerateTrainer(all_characters=all_katakana, name_list=all_names)
    trainer.train()

    torch.save(trainer.rnn.state_dict(), "../model/v1_state_dict.pth")


if __name__ == "__main__":
    main()
