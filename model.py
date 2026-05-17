import torch
import torch.nn as nn
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex # Intel optimization for custom models

# ==========================================
# 1. THE DATA (Malay Corpus)
# ==========================================
# In a real scenario, you would load a massive dataset. 
# Here, we use a small paragraph so the model can learn it in seconds.
malay_text = """
Bahasa Melayu ialah bahasa kebangsaan dan bahasa rasmi di Malaysia. 
Bahasa ini digunakan oleh pelbagai kaum di negara kita untuk berkomunikasi.
Memelihara bahasa Melayu adalah tanggungjawab semua rakyat Malaysia.
Kesusasteraan Melayu tradisional sangat kaya dengan pantun, sajak, dan gurindam.
"""

# Simple Character-level tokenizer (Real LLMs use Byte-Pair Encoding)
chars = sorted(list(set(malay_text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode the text into integers
data = torch.tensor([char_to_idx[ch] for ch in malay_text], dtype=torch.long)

# ==========================================
# 2. THE ARCHITECTURE (From Scratch)
# ==========================================
class MiniMalayGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        
        # Transformer Decoder Blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, block_size) 
            for _ in range(num_layers)
        ])
        
        # Final Layer Norm and Output Head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        # Get token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Pass through transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Predict next character
        logits = self.lm_head(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.sa = CausalSelfAttention(embed_dim, num_heads, block_size)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connection + Attention
        x = x + self.ffwd(self.ln2(x)) # Residual connection + Feed Forward
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # Causal mask to prevent looking into the future
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Calculate attention scores
        wei = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # Apply attention to values
        out = wei @ v
        return out

# ==========================================
# 3. INSTANTIATE MODEL & OPTIMIZER
# ==========================================
hyperparams = {
    "vocab_size": vocab_size,
    "embed_dim": 64,      # Very small for demonstration
    "block_size": 32,     # Context window (how many chars it looks at once)
    "num_heads": 4,
    "num_layers": 2       # A real LLM has 32-80+ layers
}

model = MiniMalayGPT(**hyperparams)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# ==========================================
# 4. APPLY INTEL IPEX OPTIMIZATION
# ==========================================
# IPEX-LLM's quantization doesn't work on raw PyTorch models, but the base 
# IPEX library optimizes the graph, fused layers, and CPU/XPU instructions.
model, optimizer = ipex.optimize(model, optimizer=optimizer)
print("Model optimized with Intel Extension for PyTorch (IPEX).")

# ==========================================
# 5. TRAINING LOOP
# ==========================================
model.train()
epochs = 200
batch_size = 16

print("\n--- Training on Malay Text ---")
for epoch in range(epochs):
    # Create random mini-batches from the text
    ix = torch.randint(0, len(data) - hyperparams["block_size"], (batch_size,))
    x = torch.stack([data[i:i+hyperparams["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+1+hyperparams["block_size"]] for i in ix])
    
    # Forward pass
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ==========================================
# 6. GENERATION (Inference)
# ==========================================
model.eval()
print("\n--- Generating Malay Text ---")

# Start with a prompt
prompt = "Bahasa Melayu"
context_indices = torch.tensor([[char_to_idx[ch] for ch in prompt]], dtype=torch.long)

# Autoregressively generate 100 characters
generated_chars = list(prompt)
for _ in range(100):
    # Crop to block_size if needed
    context_crop = context_indices[:, -hyperparams["block_size"]:]
    with torch.no_grad():
        logits = model(context_crop)
    # Get probabilities of the very last character
    probs = F.softmax(logits[:, -1, :], dim=-1)
    # Sample the next character
    idx_next = torch.multinomial(probs, num_samples=1)
    context_indices = torch.cat((context_indices, idx_next), dim=1)
    generated_chars.append(idx_to_char[idx_next.item()])

print("".join(generated_chars))
