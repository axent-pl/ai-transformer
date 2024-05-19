import os
import torch
import torch.nn as nn

from gpt.model import GPTModel
from gpt.tokenizer import Tokenizer

BLOCK_SIZE = 256
EMBEDDING_DIM = 500

tokenizer = Tokenizer('../gobpe/bin', '../gobpe/params.json')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = GPTModel(vocabulary_size=tokenizer.vocabulary_size,
                 block_size=BLOCK_SIZE, embedding_dim=EMBEDDING_DIM).to(device)

if os.path.exists("gpt.pt"):
    checkpoint = torch.load("gpt.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

prompt = 'Co to będzie'
x = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
print(prompt, end='')
for token in model.generate(x.to(device), 100):
    print(tokenizer.decode(token.tolist()[0]), end='')