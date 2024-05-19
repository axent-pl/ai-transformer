import torch
from gpt.tokenizer import Tokenizer

def transform_data(text: str, tokenizer: Tokenizer) -> torch.Tensor:
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded, dtype=torch.long)

def get_train_test_split(data: torch.Tensor, test_size: float) -> (torch.Tensor, torch.Tensor):
    n = data.shape[0]
    n_test = int(n * test_size)
    n_train = n - n_test
    train = data[:n_train]
    test = data[n_train:]
    return train, test


def get_random_batch(data: torch.Tensor, batch_size: int, block_size: int) -> torch.Tensor:
    batch_ix = torch.randint(0, data.shape[0]-block_size, (batch_size,))
    x = [data[ix:ix + block_size] for ix in batch_ix]
    y = [data[ix+1:ix + block_size+1] for ix in batch_ix]
    return torch.stack(x), torch.stack(y)
