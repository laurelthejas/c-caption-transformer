"""The model from c_caption_transformer.ipynb, for use outside the notebook.

The three model classes are copies of the notebook's, so a checkpoint trained there loads here. Generation follows
the notebook's generate(), with one addition: a key-value cache, which keeps the attention keys and values of the
characters already read, so each new character costs one step instead of re-reading the whole text.
"""
import os
import shutil
import subprocess
import tempfile

import torch
import torch.nn as nn
from torch.nn import functional as F

END = "// END"   # every document in the dataset ends with this line

# The notebook's model settings. load_model() replaces them with the checkpoint's own values.
BLOCK_SIZE = 1024
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2    # only used in training; the model is always in evaluation mode here


class CausalSelfAttention(nn.Module):
    # Multi-head self-attention (MHSA): every character looks back at the earlier characters
    # and pulls in information from the ones that matter for predicting what comes next.
    # All heads at once; scaled_dot_product_attention uses fast fused kernels on the GPU.
    def __init__(self):
        super().__init__()
        # One layer makes the query, key and value vectors for all heads together:
        #   query = "what am I looking for?"  key = "what do I contain?"  value = "what I pass on"
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD)   # mixes the heads' outputs back together
        self.resid_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                       # each (B, T, C)
        # Split C into N_HEAD heads of size C // N_HEAD: (B, T, C) -> (B, N_HEAD, T, head_size).
        # Each head can learn to attend to something different (matching braces, variable names, ...).
        q, k, v = (t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in (q, k, v))
        # softmax(q·k / sqrt(head_size)) · v  for every head.
        # is_causal=True hides future characters: position t may only look at positions 0..t.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=DROPOUT if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # glue the heads back together
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    # One transformer block: attention (characters talk to each other),
    # then a small feed-forward network (each position "thinks" on its own).
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)   # LayerNorm keeps the numbers in a stable range
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
                                 nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT))

    def forward(self, x):
        # "x + ..." is a residual connection: each layer ADDS to what it received instead of
        # replacing it. That keeps information flowing and makes deep networks trainable.
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)   # learned vector for each character ("what")
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)   # learned vector for each position ("where")
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])   # N_LAYER blocks in a row
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)   # vector -> one score per possible next character
        self.apply(self._init)

    @staticmethod
    def _init(m):
        # Start with small random weights: the model knows nothing at step 0.
        # Everything it will "know" about C is learned later by adjusting these numbers during training.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        # idx: (B, T) character ids  ->  logits: (B, T, vocab_size) scores for the next character
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        logits = self.head(self.ln_f(self.blocks(x)))
        # Cross-entropy loss: how surprised the model was by the real next character
        # (lower = better guesses). Training is all about pushing this number down.
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class CCaptionModel:
    """A trained checkpoint with its vocabulary, ready to write, explain and answer."""

    def __init__(self, checkpoint_path, device):
        global BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER
        # weights_only refuses anything in the file except tensors and plain values.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.stoi, self.itos = checkpoint["stoi"], checkpoint["itos"]
        config = checkpoint.get("config", {})   # a checkpoint from the feedback cell has no config: notebook defaults
        BLOCK_SIZE = config.get("BLOCK_SIZE", 1024)
        N_EMBD, N_HEAD, N_LAYER = config.get("N_EMBD", 384), config.get("N_HEAD", 6), config.get("N_LAYER", 6)
        self.model = TransformerLanguageModel(len(self.stoi)).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.device = device
        self.use_amp = device == "cuda"
        self.step = checkpoint.get("iter")
        self.best_val = checkpoint.get("best_val")

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]   # unknown characters are skipped, as in the notebook

    @torch.no_grad()
    def _forward_cached(self, idx, cache, start):
        # Runs new characters through the model. With a cache, earlier characters' keys and values are reused,
        # and the new characters attend to them. Same layers and weights as TransformerLanguageModel.forward.
        m = self.model
        B, T = idx.shape
        x = m.tok_emb(idx) + m.pos_emb(torch.arange(start, start + T, device=idx.device))
        new_cache = []
        for layer, block in enumerate(m.blocks):
            h = block.ln1(x)
            C = h.shape[-1]
            q, k, v = block.attn.qkv(h).split(C, dim=2)
            q, k, v = (t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in (q, k, v))
            if cache is not None:
                k = torch.cat([cache[layer][0], k], dim=2)
                v = torch.cat([cache[layer][1], v], dim=2)
            new_cache.append((k, v))
            # Without a cache, every character sees only earlier ones. With one, the single new character
            # sees everything before it, which is the same rule.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
            x = x + block.attn.proj(y.transpose(1, 2).contiguous().view(B, T, C))
            x = x + block.mlp(block.ln2(x))
        return m.head(m.ln_f(x)), new_cache

    @torch.no_grad()
    def generate(self, prompt, max_new_chars=1500, temperature=0.7, top_k=20, seed=1337, on_text=None):
        # Continues the prompt one character at a time until the model writes the END marker, as the notebook does.
        # on_text(text) is called now and then with the text so far.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        ids = self.encode(prompt)
        text = prompt
        window = ids[-BLOCK_SIZE:]
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
            logits, cache = self._forward_cached(torch.tensor([window], device=self.device), None, 0)
        position = len(window)
        for i in range(max_new_chars):
            scores = logits[:, -1, :].float() / temperature        # scores for the NEXT character only
            if top_k:
                kth = torch.topk(scores, min(top_k, scores.size(-1))).values[:, [-1]]
                scores[scores < kth] = -float("inf")              # remove everything outside the top k
            nxt = torch.multinomial(F.softmax(scores, dim=-1), num_samples=1, generator=generator)
            ids.append(nxt.item())
            text += self.itos[nxt.item()]
            if text.endswith(END):
                break
            if on_text and i % 25 == 0:
                on_text(text)
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                if position < BLOCK_SIZE:
                    logits, cache = self._forward_cached(nxt.view(1, 1), cache, position)
                    position += 1
                else:
                    # The model can only see BLOCK_SIZE characters. Like the notebook, keep the last BLOCK_SIZE
                    # and read them again from position 0, as the cached positions no longer apply.
                    logits, cache = self._forward_cached(torch.tensor([ids[-BLOCK_SIZE:]], device=self.device),
                                                         None, 0)
        return text

    def write_program(self, caption, **settings):
        # caption -> plan, C program and English explanation
        return self.generate(f"/* Caption: {caption} */\n", **settings)

    def explain_code(self, code, **settings):
        # C code -> English explanation
        code = code.split("/* Explanation:")[0].split(END)[0].rstrip()
        return self.generate(code + "\n/* Explanation:", max_new_chars=700, **settings)[len(code) + 1:]

    def ask(self, question, **settings):
        # question -> short answer
        return self.generate(f"/* Question: {question} */\n/* Answer:", max_new_chars=700, **settings)


def find_compiler():
    return shutil.which("gcc") or shutil.which("cc")


def compiles(code, compiler):
    # Syntax and type check only, as in the notebook: nothing is built or run. Returns (ok, first error line).
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "prog.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code.replace(END, ""))
        result = subprocess.run([compiler, "-std=c11", "-fsyntax-only", src], capture_output=True, text=True,
                                timeout=30)
    errors = [line for line in result.stderr.splitlines() if "error" in line]
    return result.returncode == 0, (errors[0] if errors else "")


def split_program(output):
    # Splits a written program into (caption line and headers, plan, code, explanation), as the notebook's 💬 cell does.
    code, _, explanation = output.partition("/* Explanation:")
    head, _, tail = code.partition("/* Plan:")
    plan, _, body = tail.partition("*/")
    return head.rstrip(), plan.strip(), body.strip(), clean(explanation)


def clean(text):
    # Removes the comment markers and the END marker, keeping plain sentences.
    return " ".join(text.replace("/*", "").replace("*/", "").replace(END, "").split())
