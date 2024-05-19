import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, block_size: int, input_size: int, head_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.query = nn.Linear(input_size, head_size, bias=False)
        self.key = nn.Linear(input_size, head_size, bias=False)
        self.value = nn.Linear(input_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        is_batched = (x.dim() == 3)

        if is_batched:
            _, T, C = x.shape
        else:
            T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        w = q @ k.transpose(-2, -1) * C**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        o = w @ v

        return o


class MultiHead(nn.Module):

    def __init__(self, block_size: int, embedding_dim: int, num_heads: int, head_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size=block_size, input_size=embedding_dim,
                                   head_size=head_size, dropout=dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        y = self.projection(y)
        y = self.dropout(y)
        return y


class FeedFoward(nn.Module):

    def __init__(self, embedding_dim, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):

    def __init__(self, block_size: int, embedding_dim: int, num_heads: int, dropout: float = 0.2) -> None:
        super().__init__()
        head_size = embedding_dim // num_heads
        self.multihead = MultiHead(block_size=block_size, embedding_dim=embedding_dim,
                                   num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.multihead_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedFoward(embedding_dim=embedding_dim)
        self.feedforward_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        y = x + self.multihead(self.multihead_norm(x))
        y = y + self.feedforward(self.feedforward_norm(y))
        return y


class GPTModel(nn.Module):

    def __init__(self, vocabulary_size: int, block_size: int, num_heads: int = 6, num_transformers: int = 6, embedding_dim: int = 12) -> None:
        super().__init__()

        self.block_size = block_size

        self.token_embdedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.position_embdedding = nn.Embedding(block_size, embedding_dim)
        self.transformers = nn.Sequential(
            *[Transformer(block_size=block_size, embedding_dim=embedding_dim, num_heads=num_heads) for _ in range(num_transformers)])
        self.ln = nn.LayerNorm(embedding_dim)
        self.ll = nn.Linear(embedding_dim, vocabulary_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_batched = (x.dim() == 2)

        if is_batched:
            B, T = x.shape
        else:
            T = x.shape

        xt = self.token_embdedding(x)
        xp = self.position_embdedding(torch.arange(T, device=x.device))

        xtp = xt + xp

        y = self.transformers(xtp)
        y = self.ln(y)
        y = self.ll(y)

        return y

    def generate(self, x: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        for i in range(horizon):
            y = self(x)
            probs = F.softmax(y[:, -1, :], dim=-1)
            c_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, c_next), dim=1)
            x = x[:, -min(self.block_size, x.shape[1]):]
            yield c_next
