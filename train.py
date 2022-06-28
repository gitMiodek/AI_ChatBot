import json
import sys

import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from nltk_utils import tokenizer, stem, bag_of_wrds
from neural_net import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

tags = []
all_words = []
tags_n_words = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenizer(pattern)
        all_words.extend(w)
        tags_n_words.append((tag, w))

punctuation_marks = ['?', '!', ',', '.']
all_words = [stem(word) for word in all_words if word not in punctuation_marks]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for tag, pattern_sentence in tags_n_words:
    bag = bag_of_wrds(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


# CREATE Pytorch Dataset

class ChatSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples


# hyperparameters
batch_size = 8
input_layers = len(X_train[0])
hidden_layers = 8
num_class = len(tags)
num_epochs = 500
lr = 0.003

train_dataset = ChatSet()
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_layers=input_layers, hidden_layers=hidden_layers, num_classes=num_class).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_func = nn.CrossEntropyLoss()

example_sent, example_label = next(iter(train_loader))
print(example_sent)
print(example_label)

# Training loop
for epoch in range(num_epochs):

    for sentence, label in train_loader:
        sentence = sentence.to(device)
        label = label.to(device)

        prediction = model(sentence)
        loss = loss_func(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch }/{num_epochs}, loss: {loss.item():.4f}')
print(f'Final loss: {loss.item():4f}')

data = {
    'model_state': model.state_dict(),
    'input_size': input_layers,
    'output_size': num_class,
    'hidden_size': hidden_layers,
    'all_words': all_words,
    'tags': tags,


}
FILE = 'data.pth'
torch.save(data, FILE)
print(f'[INFO] Training done, saving to {FILE}')