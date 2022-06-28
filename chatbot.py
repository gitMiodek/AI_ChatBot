import torch
import json
import random
from neural_net import NeuralNet
from nltk_utils import tokenizer, bag_of_wrds

with open('intents.json', 'r') as f:
    intents = json.load(f)


class ChatBot():
    def __init__(self, data_file='data.pth'):
        self.data = torch.load(data_file)
        self.model_state = self.data['model_state']

        self.input_size = self.data["input_size"]
        self.hidden_size = self.data["hidden_size"]
        self.output_size = self.data["output_size"]
        self.all_words = self.data['all_words']
        self.tags = self.data['tags']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)

    def get_msg(self, message):
        self.model.load_state_dict(self.model_state)
        self.model.eval()
        msg = tokenizer(message)
        msg = bag_of_wrds(msg, self.all_words)
        msg = torch.from_numpy(msg).to(self.device)
        output = self.model(msg)

        x = output.tolist()
        idx = x.index(max(x))
        tag = self.tags[idx]


        for intent in intents['intents']:
            if tag == intent["tag"]:
                answer = random.choice(intent['responses'])
                return answer
