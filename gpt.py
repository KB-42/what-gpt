import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 64 # Independent sequences to process in parallel
block_size = 256 # what is the max context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

torch.manual_seed(1337)

# Read in the tiny shake
with open('input-tiny-shake.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters in the dataset
unique_characters = set(text)

# Get our characters in a sorted list format
chars = sorted(list(unique_characters))

# Number of unique characters
vocab_size = len(chars)

# Mapping of characters to integers and visa-versa
string_to_integer = { ch:i for i,ch in enumerate(chars) }
integer_to_string = { i:ch for i,ch in enumerate(chars) }

encode_text = lambda string: [string_to_integer[char] for char in string] # take a string (list of chars) and output a list of integers
decode_ints = lambda integers: ''.join([integer_to_string[integer] for integer in integers]) # take a list of integers and output the string

# train and split
data = torch.tensor(encode_text(text), dtype=torch.long)

n = int(0.9*len(data)) # 90% is the training data
train_data = data[:n] #1,003,854 tokens (characters)
validation_data = data[n:]

def get_batch(split):
    # generate a small batch of data of input x and target y
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()
    return out

class Head(nn.Module):
    '''one head of self-attention'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)  < -- What I have
        q = self.query(x) # (B,T,C) <-- What I am interested in
       
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) --> prevent time travel. Sentement does tho
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    ''' Multiple heads of self-attention in parallel '''

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    '''Transformer block: communication followed by computation'''
    
    def __init__(self, n_embd, n_head):
        #  n_embd: embedding demension, n_head: the number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    
    def get_token_embeddings(self):
        return self.token_embedding_table
    def get_pos_embeddings(self):
        return self.position_embedding_table

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) Batch, Block, Vocab_size
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # convert to 2D array
            targets = targets.view(B*T)
            # How well are we predicting the next token?
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:] #idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # Focus on last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx





model = BigramLanguageModel()
m = model.to(device)
new_mod = torch.load('test_sav', map_location=device)


print('---- ShakespeareGPT ----')

while True:
    prompt = input('Prompt: ')

    if (prompt == ''):
        break
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    new_context = torch.tensor([encode_text(prompt)], dtype=torch.long, device=device)
    print(decode_ints(new_mod.generate(new_context, max_new_tokens=500)[0].tolist()))




