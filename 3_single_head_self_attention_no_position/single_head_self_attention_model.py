import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

class Head(nn.Module):
    ''' one head of self-attention'''

    def __init__(self, emb_dim, head_size, block_size):
        super().__init__()
        self.key_layer = nn.Linear(emb_dim, head_size, bias=False)
        self.query_layer = nn.Linear(emb_dim, head_size, bias=False)
        self.value_layer = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
            # This is not a parameter. It is a buffer
            # Stores the lower triangular matrix

    def forward(self, x):
        B, T, C = x.shape   # C is head_size

        key = self.key_layer(x) # Shape: (B, T, C)
        query = self.query_layer(x) # Shape: (B, T, C)

        # Compute attention scores ("affinities")
        weight = query @ key.transpose(-2, -1) * C**-0.5    # Normalise by dividing using sqrt of head_size
            # Shape: (B, T, C) @ (B, C, T) -> (B, T, T)

        weight = weight.masked_fill(self.get_buffer("tril")[:T, :T] == 0, float('-inf'))    # Shape: (B, T, T)
        weight = F.softmax(weight, dim=-1)    # Shape: (B, T, T)

        # Perform weighted aggregation of values
        value = self.value_layer(x) # Shape: (B, T, C)
        output = weight @ value # Shape: (B, T, T) @ (B, T, C) -> (B, T, C)
        return output

class SingleHeadSelfAttentionModel(nn.Module):

    def __init__(self, vocab_size, n_emb, head_size, block_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.self_attention_head = Head(n_emb, head_size, block_size)
        self.lang_modelling_head = nn.Linear(n_emb, vocab_size)

    # B is Batch size
    # T is Time dimension, which is also the block size
    # C is Channel size
    def forward(self, idx, targets=None):
        # Shape of idx: (B, T)
        # Shape of targets: (B, T)
        B, T = idx.shape

        x = self.token_embedding_table(idx)    # Shape: (B, T, n_emb)
        x = self.self_attention_head(x)     # Shape: (B, T, n_emb)
        logits = self.lang_modelling_head(x)  # Shape: (B, T, vocab_size)

        if targets is None: # During inference
            loss = None

        else:   # During training
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    # Stretch out the tensor
                # Put C as the 2nd dimension before putting into F.cross_entropy() as 1st arg

            targets = targets.view(B*T) # Alt: targets.view(-1)
                # Make sure its dimension matches the 1st dimension of the 1st argument of F.cross_entropy()

            # Negative log likelihood
            loss = F.cross_entropy(logits, targets)

        return logits, loss
            # Loss is estimated to be around -ln(1/C), which is -ln(1/65) = 4.17439
            # If have a huge difference means the initial prediction is not super diffused and has some entropy.

    def generate(self, idx, num_max_new_tokens, block_size):
        # idx is (B, T) array of indices in the current context
        # the function of generate is
            # to turn idx from (B, T) to (B, T + num_max_new_tokens),
            # generate the next sequence one by one

        for _ in range(num_max_new_tokens):
            # Get the predictions
            logits, __ = self(idx[:, -block_size:])  # loss is ignored during inference
            # Focus on the logits of the last time step
            logits = logits[:, -1, :]   # Shape: (B, T, C) -> (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)   # Shape: (B, C)
            # Sample from the distribution to predict next token
            idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # Shape: (B, T+1)

        return idx  # Shape: (B, T+num_max_new_tokens)
        # For bigram model, the implementation of this generate() function is abit unnecessary
        # It does not need the tokens of the whole context, as bigram only needs the last token
        # However, this generate() function is kept generic to extend to the subsequent stages,
        # so that the prediction can take into account of the whole context later
