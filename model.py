from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken, math, time, inspect, os
import argparse
from pathlib import Path

import torch._dynamo
torch._dynamo.config.suppress_errors = True

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
        self.c_proj.scale_init = 1
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

        # att = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_head, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Flash Attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # first linear layer
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.scale_init = 1
    
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
        self.transformer.wte.weight = self.lm_head.weight  # tie weights between token embeddings and output layer

        self.apply(self._init_weights)  # initialize weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'scale_init'):
                 std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
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
        loss = None
        logits = self.lm_head(x)  # (B, T, vocab size) output logits
        if targets is not None:
            # compute loss if targets are provided
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss

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
    
    def configure_optimizers(self, weight_decay, lr, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]   # wieghts in MatMul, Embedding, etc.
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # bias, layer norm, etc.
        assert len(decay_params) + len(nodecay_params) == len(param_dict)," deacy + non-deacy = total"
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_available and device.type == 'cuda' and torch.cuda.is_available()
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameter")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameter")
            print(f"Using fused AdamW: {used_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, fused=used_fused, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


class DataLoaderLite: 
    def __init__(self, file_path: str, B: int, T: int, device, rank: int, num_processes: int):
        self.B = B
        self.T = T
        self.device = device
        self.rank, self.num_processes = rank, num_processes
        with open(file_path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, device=device)
        self.current_position = (self.B * self.T * self.rank) % len(self.tokens)  # start position inside buffer
        if master_process:
            print(f"Loaded {len(self.tokens)} tokens from {file_path}")
            print(f"1 epoch = {len(self.tokens) // (B * T)} batches of size {B} and sequence length {T}")

    def next_batch(self):
        """Return the next (x, y) batch for this rank, wrapping around the
        token buffer if necessary so that the slice is always contiguous and
        of the expected length. This avoids shape errors in multi-GPU runs
        when the dataset size is not an exact multiple of (B*T*num_processes).
        """
        B, T = self.B, self.T
        needed = B * T + 1

        if self.current_position + needed > len(self.tokens):
            # wrap around the end of the buffer
            first_part = self.tokens[self.current_position:]
            remaining = needed - first_part.size(0)
            second_part = self.tokens[:remaining]
            buf = torch.cat((first_part, second_part), dim=0)
        else:
            buf = self.tokens[self.current_position:self.current_position + needed]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # advance the read pointer for the next call (each rank jumps by the
        # full global batch so that all processes walk the dataset together
        # without overlap)
        self.current_position = (self.current_position + B * T * self.num_processes) % len(self.tokens)
        return x, y

if __name__ == "__main__":

    ddp = int(os.environ.get("WORLD_SIZE", "-1")) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA to be available"   
        # ddp_rank = int(os.environ.get("RANK", "0"))
        # ddp_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # ddp_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        # master_process = ddp_rank == 0
        # init_process_group(backend='nccl', rank=ddp_rank, world_size= ddp_world_size)
        init_process_group(backend='nccl')  # initialize DDP
        ddp_rank = int(os.environ["RANK"])  # rank of the current process
        ddp_local_rank = int(os.environ["LOCAL_RANK"])  # local rank of the current process
        ddp_world_size = int(os.environ["WORLD_SIZE"])  # total
        device = torch.device(f"cuda:{ddp_local_rank}")  # set device for this process
        torch.cuda.set_device(device)  # set the current device for this process
        master_process = (ddp_rank == 0)  # only the master process will
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.manual_seed(42)  # for reproducibility on GPU
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

    torch.manual_seed(42)  # for reproducibility    

    # ---------------- Argument parsing -----------------
    parser = argparse.ArgumentParser(description="GPT training script (single-GPU or DDP)")
    parser.add_argument("--data", type=str, default="combined.txt", help="Path to the training token file (raw text)")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if master_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Hyper-parameters -----------------
    # Following GPT-3 scaling: keep lr ~6e-4 for 0.5M token batch
    # We run with ~0.52M tokens/step (B=64, T=1024, 8 GPUs, grad_accum=1)
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 2000
    max_steps = 288000

    def get_lr(step):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        if step > max_steps:
            return min_lr
        decay = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay <= 1, "Decay should be in the range [0, 1]"
        # cosine learning rate schedule
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(decay * math.pi))

    B, T = 64, 1024  # per-GPU micro-batch (tokens per device per fwd pass)
    gradient_accumulation_steps = 1  # gives ~0.5M tokens/opt step on 8 GPUs
    total_batch_size = B * T * ddp_world_size * gradient_accumulation_steps
    if master_process:
        print(f"Using DDP with {ddp_world_size} processes, each with batch size {B}, sequence length {T}, and gradient accumulation steps {gradient_accumulation_steps}")
        print(f"Total batch size: {total_batch_size}, per process batch size: {B * T}, per process gradient accumulation steps: {gradient_accumulation_steps}")
    train_loader = DataLoaderLite(file_path=args.data, B=B, T=T, device=device, rank=ddp_rank, num_processes=ddp_world_size)

    # torch.set_float32_matmul_precision('high')  # set float32 matmul precision for better performance
    # torch.backends.cudnn.benchmark = False
 
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig(vocab_size=50304))    # not pretrained, so gibberish
    model.to(device)
    model = torch.compile(model)  # compile the model for better performance
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])  # wrap model in DDP
    raw_model = model.module if ddp else model  # get the raw model for optimizer configuration

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8) # betas, gpt3
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, lr=1e-3, device=device)  # configure optimizer with weight decay

    # ---------------- Optional resume -----------------
    start_step = 0
    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.is_file():
            map_location = {f"cuda:{ddp_local_rank}": device} if device.type == "cuda" else "cpu"
            checkpoint = torch.load(ckpt_path, map_location=map_location)
            raw_model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint.get("step", 0)
            train_loader.current_position = checkpoint.get("dataloader_pos", train_loader.current_position)
            if master_process:
                print(f"Resumed from {ckpt_path} at step {start_step}")
        else:
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

    # ---------------- Training loop -----------------
    for step in range(start_step, max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_acum = 0.0
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        model.train()  # set model to training mode
        for micro_step in range(gradient_accumulation_steps):
            x, y = train_loader.next_batch()  # get next batch of data
            x, y = x.to(device), y.to(device)  # move to device
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps  # scale loss for gradient accumulation
            loss_acum += loss.detach()
            if ddp:
                  model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)  # sync gradients only on the last micro step
            loss.backward()
        if ddp:
            dist.all_reduce(loss_acum, op=dist.ReduceOp.AVG)  # reduce loss across all processes
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping, gpt3
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time() 
        dt = (t1 - t0) * 1000 # time in milliseconds
        tokens_per_sec = (x.size(0) * x.size(1) * gradient_accumulation_steps * ddp_world_size) / (t1- t0)  # tokens per second
        if master_process:
            print(f"Step {step:4d} | loss: {loss_acum.item():.6f} | norm: {norm:.4f} | dt: {dt:.2f} ms | tokens/sec: {tokens_per_sec:.2f}")

        # --------------- Checkpointing ---------------
        if master_process and ((step + 1) % args.save_interval == 0 or step == max_steps - 1):
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "dataloader_pos": train_loader.current_position,
            }
            ckpt_file = output_dir / f"checkpoint_step_{step+1:07d}.pt"
            torch.save(ckpt, ckpt_file)
            print(f"Saved checkpoint to {ckpt_file}")

    if ddp:
        destroy_process_group()

    # torchrun --nproc_per_node=8 model.py --data combined.txt --output_dir ckpts
    # torchrun --nproc_per_node=8 model.py --resume_from ckpts/checkpoint_step_000100.pt
