from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # k, q, v projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)   # output projection
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim (n_embd)
        # Split the combined projection into query, key, value tensors
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)

        # Reshape and transpose for multi-head attention
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_head, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # first linear layer
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x   

@dataclass
class GPTConfig:
    block_size: int = 1024   # maximum sequence length,  context length
    vocab_size: int = 50257  # num of tokens:  50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768   


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # positional embeddings
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # transformer blocks
                ln_f=nn.LayerNorm(config.n_embd),  # final layer norm
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # language modeling head

    def forward(self, idx, targets=None):
        B, T = idx.size()   # Batch dim, time dim
        assert T <= self.config.block_size, "Cannot forward sequence of length %d, block size is only %d" % (T, self.config.block_size)
        pos = torch.arange(T, device=idx.device, dtype=torch.long)  # positional indices
        pos_emb = self.transformer.wpe(pos)  # positional embeddings of shape (T, n_embd), pos independent of batch
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb  # combine token and positional embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab size) output logits
        return logits

    @classmethod
    def from_pretrained(cls, model_type):   
        """ loads pretrained gpt2 weights from HuggingFace transformers """
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], "Invalid model type"
        from transformers import GPT2Config, GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # num of layers, heads, embd are determined by model type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),            # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),    # 350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),     # 774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),        # 1558M
        }[model_type]
        config_args['block_size'] = 1024
        config_args['vocab_size'] = 50257
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()   # original 161, our model 149
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # transpose Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                # copy the rest paramters
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

if __name__ == "__main__":
    torch.manual_seed(42)  # for reproducibility
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)  # for reproducibility on GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)
    num_return_sequences = 5
    max_length = 30

    model = GPT.from_pretrained('gpt2')
    model.eval()    # not training
    model.to(device)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model")
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  #(8,) -> (1, 8)
    tokens = tokens.repeat(num_return_sequences, 1)  # repeat for num_return_sequences, (5, 8)
    x = tokens.to(device)

    # B = 5, T = 8, vocab_size = 50257
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)   # (B, T, vocab_size) ()
            logits = logits[:, -1, :]  # (B, vocab_size) only last token
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # (B, 50)
            next_tokens = torch.multinomial(topk_probs, num_samples=1)  # (B, 1) sample from topk probabilities
            next_idx = torch.gather(topk_indices, dim=-1, index=next_tokens)    # (B, 1) get the indices of the sampled tokens
            x = torch.cat((x, next_idx), dim=1)  # append next token to the sequence
    
    for i in range(num_return_sequences):
        output_tokens = x[i, :max_length].tolist()
        output_text = enc.decode(output_tokens)
        print(">", output_text)
        # print(f"Generated text {i+1}: {output_text}")
        # print(f"Tokens: {output_tokens}")
        # print(f"Token count: {len(output_tokens)}")
        