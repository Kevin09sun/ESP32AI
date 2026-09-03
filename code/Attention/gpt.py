import os
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------- hyperparameters
n_embd        = 32
n_head        = 4
n_layer       = 2
dropout       = 0.0      # raise to 0.1 only if train/val gap exceeds ~0.15

batch_size    = 64
max_iters     = 15000
eval_interval = 500
learning_rate = 3e-3
min_lr        = 3e-4
warmup_iters  = 200
weight_decay  = 0.01

seed       = 1337
split_seed = 42          # MUST match your wavenet notebook's split, or the comparison is contaminated
device     = 'cuda' if torch.cuda.is_available() else 'cpu'

assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
head_size = n_embd // n_head
attn_scale = head_size ** -0.5

torch.manual_seed(seed)
random.seed(split_seed)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- data
with open(os.path.join(HERE, 'names.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))          # '\n' sorts first -> index 0, serves as start/end token
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
assert stoi['\n'] == 0, "expected newline at index 0 (it is the start/end token)"

names = [w for w in text.splitlines() if w]
max_word_len = max(len(w) for w in names)
block_size = max_word_len + 1            # every name fits whole; no sliding window ever
print(f"vocab_size={vocab_size}  names={len(names)}  max_word_len={max_word_len}  block_size={block_size}")

random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))
splits_words = {'train': names[:n1], 'val': names[n1:n2], 'test': names[n2:]}


def build_dataset(words):
    """One row per name. x = [start] + chars, y = chars + [end], padded with -1 targets."""
    X, Y = [], []
    for w in words:
        idx = encode(w)
        x = ([0] + idx)[:block_size]
        y = (idx + [0])[:block_size]
        pad = block_size - len(x)
        X.append(x + [0] * pad)
        Y.append(y + [-1] * pad)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


data = {k: build_dataset(v) for k, v in splits_words.items()}
for k, (X, Y) in data.items():
    print(f"  {k}: {X.shape[0]} names, {(Y >= 0).sum().item()} predicted characters")


def get_batch(split):
    X, Y = data[split]
    ix = torch.randint(X.shape[0], (batch_size,))
    return X[ix].to(device), Y[ix].to(device)


# ---------------------------------------------------------------- model
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * attn_scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
        # scaled init on residual projections (depth-stable; makemore/nanoGPT convention)
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('net.2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(
            torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T),
                                   ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate_name(self, max_new=None):
        """Generate one name, starting from the start token, stopping at end token."""
        max_new = max_new or (block_size - 1)
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = []
        for _ in range(max_new):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if nxt.item() == 0:
                break
            out.append(nxt.item())
            idx = torch.cat((idx, nxt), dim=1)
        return decode(out)


@torch.no_grad()
def exact_loss(split, chunk=512):
    """Full-pass loss over the whole split. No sampling noise."""
    model.eval()
    X, Y = data[split]
    total, count = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        xb, yb = X[i:i + chunk].to(device), Y[i:i + chunk].to(device)
        logits, _ = model(xb)
        mask = yb >= 0
        ls = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1),
                             ignore_index=-1, reduction='sum')
        total += ls.item()
        count += mask.sum().item()
    model.train()
    return total / count


def lr_at(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    frac = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * frac))


# ---------------------------------------------------------------- train
model = GPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params}  ({n_params * 4 / 1024:.1f} KB float32)")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                              weight_decay=weight_decay)

best_val = float('inf')
best_state = None
for it in range(max_iters):
    for g in optimizer.param_groups:
        g['lr'] = lr_at(it)

    if it % eval_interval == 0 or it == max_iters - 1:
        tr, va = exact_loss('train'), exact_loss('val')
        flag = ''
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = '  <- best'
        print(f"step {it:>6}: train {tr:.4f}  val {va:.4f}  lr {lr_at(it):.2e}{flag}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

model.load_state_dict(best_state)
final = {s: exact_loss(s) for s in ('train', 'val', 'test')}
print(f"\nbest checkpoint: train {final['train']:.4f}  val {final['val']:.4f}  "
      f"test {final['test']:.4f}")
print("samples:", ' '.join(model.generate_name() for _ in range(20)))


# ---------------------------------------------------------------- export
config = {
    'vocab_size': vocab_size, 'block_size': block_size, 'n_embd': n_embd,
    'n_head': n_head, 'n_layer': n_layer, 'head_size': head_size,
    'attn_scale': attn_scale, 'layernorm_eps': 1e-5,
    'n_params': n_params, 'bytes_float32': n_params * 4,
    'losses': final, 'seed': seed, 'split_seed': split_seed,
    'itos': {str(k): v for k, v in itos.items()},
}

torch.save({'state_dict': best_state, 'config': config}, os.path.join(HERE, 'transformer_ckpt.pt'))


def export_arrays():
    """Fuse per-head Q/K/V into single (n_embd, n_embd) matrices and transpose all
    weights to [in, out] so the C matmul is  acc[j] += x[k] * W[k][j]  as in the MLP sketch."""
    sd = best_state
    out = {}
    out['tok_emb'] = sd['token_embedding_table.weight'].numpy()          # [vocab, n_embd]
    out['pos_emb'] = sd['position_embedding_table.weight'].numpy()       # [block, n_embd]

    for L in range(n_layer):
        p = f'blocks.{L}.'
        out[f'l{L}_ln1_g'] = sd[p + 'ln1.weight'].numpy()
        out[f'l{L}_ln1_b'] = sd[p + 'ln1.bias'].numpy()
        for tag in ('key', 'query', 'value'):
            # concat heads in the same order forward() concatenates them
            fused = torch.cat([sd[f'{p}sa.heads.{h}.{tag}.weight'] for h in range(n_head)], dim=0)
            out[f'l{L}_{tag[0]}'] = fused.t().contiguous().numpy()       # [n_embd, n_embd]
        out[f'l{L}_o']    = sd[p + 'sa.proj.weight'].t().contiguous().numpy()
        out[f'l{L}_o_b']  = sd[p + 'sa.proj.bias'].numpy()
        out[f'l{L}_ln2_g'] = sd[p + 'ln2.weight'].numpy()
        out[f'l{L}_ln2_b'] = sd[p + 'ln2.bias'].numpy()
        out[f'l{L}_fc']   = sd[p + 'ffwd.net.0.weight'].t().contiguous().numpy()
        out[f'l{L}_fc_b'] = sd[p + 'ffwd.net.0.bias'].numpy()
        out[f'l{L}_pr']   = sd[p + 'ffwd.net.2.weight'].t().contiguous().numpy()
        out[f'l{L}_pr_b'] = sd[p + 'ffwd.net.2.bias'].numpy()

    out['lnf_g'] = sd['ln_f.weight'].numpy()
    out['lnf_b'] = sd['ln_f.bias'].numpy()
    out['head']   = sd['lm_head.weight'].t().contiguous().numpy()        # [n_embd, vocab]
    out['head_b'] = sd['lm_head.bias'].numpy()
    return {k: v.astype(np.float32) for k, v in out.items()}


arrays = export_arrays()
np.savez(os.path.join(HERE, 'transformer_weights.npz'), **arrays)
config['shapes'] = {k: list(v.shape) for k, v in arrays.items()}
config['export_bytes'] = int(sum(v.nbytes for v in arrays.values()))
with open(os.path.join(HERE, 'transformer_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"\nexported {len(arrays)} arrays, {config['export_bytes'] / 1024:.1f} KB float32")
for k, v in arrays.items():
    print(f"  {k:12s} {str(list(v.shape)):12s}")

# Numerical check: the fused/transposed export must reproduce the model exactly.
with torch.no_grad():
    ref, _ = model(data['test'][0][:8].to(device))
    q0 = torch.cat([best_state[f'blocks.0.sa.heads.{h}.query.weight'] for h in range(n_head)], dim=0)
    assert np.allclose(arrays['l0_q'], q0.t().numpy(), atol=0), "fused Q mismatch"
    print(f"\nfusion check passed. reference logits[0,0,:3] = {ref[0, 0, :3].cpu().numpy()}")
    print("Use these on the chip to verify the C port before trusting any timing.")