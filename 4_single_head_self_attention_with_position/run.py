import torch
from single_head_self_attention_model_with_position import SingleHeadSelfAttentionModelWithPosition

with open("data/tiny_shakespeare.txt", 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char2idx = { char:idx for idx, char in enumerate(chars) }
idx2char = { idx:char for idx, char in enumerate(chars) }

encode = lambda string : [ char2idx[c] for c in string ]
decode = lambda li : ''.join([idx2char[i] for i in li])

with open("data/tiny_shakespeare.txt", 'r', encoding='utf-8') as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)

train_valid_split_idx = int(0.9 * len(data))
train_data = data[:train_valid_split_idx]
valid_data = data[train_valid_split_idx:]

torch.manual_seed(1337)
block_size = 8
batch_size = 32
emb_dim = 32

def get_batch(training_flag):
    # generate a small batch of data of inputs x and targets y
    data = train_data if training_flag in ("train", "TRAIN", 1, True) else valid_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

@torch.no_grad()
# Tell pytorch we will never call loss.backward() inside this function (dont intent to do back propg)
# So that pytorch can be alot more efficient with its memory use
def estimate_loss(eval_iters):
    # Instead of getting the loss from a single batch
    # This function averages the losses from multiple batches
    # So that observing the change in loss will be less noisy and more accurate
    outputs = {}
    model.eval()    # Switch the model to evaluation phase
    for phase in ['train', 'validate']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(phase)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        outputs[phase] = losses.mean()
    model.train()   # Switch the model to training phase, turn back on the dropout & batchnorm layers
    return outputs

x_batch, y_batch = get_batch("train")

## TRAINING THE MODEL
model = SingleHeadSelfAttentionModelWithPosition(vocab_size, emb_dim, emb_dim, block_size)
    # Letting embedding size be head size also
logits, loss = model(x_batch, y_batch)

## MODEL TRAINING
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Optimizer takes the gradients and update the parameters using the gradients
    # Usually recommended lr is 1e-3 or 1e-4. But for smaller network like this, it can be higher like 1e-2

batch_size = 32
n_iters = 10000
eval_interval = 500
for iter in range(n_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss(eval_interval)
        print(f"step {iter}: Training loss {losses['train']:.4f}, Validation loss {losses['validate']:.4f}")

    # Sample a batch of data
    x_batch, y_batch = get_batch("train")

    # Evaluate the loss
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Current loss after training: {loss.item()}")

## MODEL PREDICION & INFERENCE AFTER TRAINING
first_token_aft_training = torch.zeros((1,1), dtype=torch.long)    # In our case, 0 is the newline character

assert first_token_aft_training < vocab_size

first_token = torch.zeros((1,1), dtype=torch.long)    # In our case, 0 is the newline character

generated_seq_aft_training = decode(model.generate(first_token, 1000, block_size)[0].tolist())
print(generated_seq_aft_training)