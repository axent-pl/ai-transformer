import os
import torch
import torch.nn as nn

from gpt.model import GPTModel
from gpt.tokenizer import Tokenizer
from gpt.data import transform_data, get_train_test_split,get_random_batch

DATA_FILE = 'data.txt'
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 64
BLOCK_SIZE = 256
EMBEDDING_DIM = 500
LEARNING_RATE = 2e-4

with open(DATA_FILE, 'r') as file:
    text = file.read()

tokenizer = Tokenizer('../gobpe/bin', '../gobpe/params.json')
data = transform_data(text, tokenizer)
train, test = get_train_test_split(data, TRAIN_TEST_SPLIT)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = GPTModel(vocabulary_size=tokenizer.vocabulary_size,
                 block_size=BLOCK_SIZE, embedding_dim=EMBEDDING_DIM).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

if os.path.exists("gpt.pt"):
    print('Loading from checkpoint')
    checkpoint = torch.load("gpt.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for i in range(500):
    inputs_, labels_ = get_random_batch(train, BATCH_SIZE, BLOCK_SIZE)
    inputs = inputs_.to(device)
    labels = labels_.to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)

    # reshape to (B*T,C)
    B, T, C = outputs.shape
    outputs_flat = outputs.view(B*T, C)
    labels_flat = labels.view(B*T)

    loss = loss_fn(outputs_flat, labels_flat)
    loss.backward()
    optimizer.step()

    if i%50 == 0:
        print(i, loss.item())

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "gpt.pt")