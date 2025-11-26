import torch
import torch.nn as nn
import tiktoken



GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size - The number of subwords after BPE
    "context_length": 1024, # Context length - The max num of words used to predict the next word
    "emb_dim": 768,         # Embedding dimension - Every token of the (50257) is represented in 768 dim vector
    "n_heads": 12,          # Number of attention heads - we will have 12 attention heads which is part of the attn mechanism
                            # which (the attn mechanism) is part of a transformer block (note that we can have many layers of
                            # transformer blocks stacked to each others)
    "n_layers": 12,         # Number of layers - we will have 12 transformer blocks stacked to each others.
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # 3. Step 3: Initialize Trainable Weights (Wq, Wk, Wv):
        self.W_query = nn.Linear(d_in, d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Optional projection layer to combine head outputs
        # note that out_proj is d_out, d_out


        # Add the dropout layer
        self.dropout = nn.Dropout(dropout)

        # Add the mask => fixed mask matrix will be used later on
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Step4 : Calculate Queries, Keys, and Values.
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        # Shape X is (1,3,6) * Wq,Wk,Wv (6,6) => (1,3,6) which is batch, num_tokens, d_out

        # Step 5: Unroll last dimension of Q, K, and Kv to include num_heads, and head_dim
        # d_out = num_heads * head_dim
        # (b, num_tokens, d_out) => (b, num_tokens, num_heads, head_dim)
        # (1,3,6) => (1,3,2,3)
        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)

        Q = Q.view(b, num_tokens, self.num_heads, self.head_dim)
        K = K.view(b, num_tokens, self.num_heads, self.head_dim)
        V = V.view(b, num_tokens, self.num_heads, self.head_dim)

        # Step 6. Group the matrices by numb_heads not by num_tokens
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        Q = Q.transpose(1,2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Step 7: Compute Attn Scores (Causal Att)
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # result (1,2,3,3) X (1,2,3,3) -> (1,2,3,3) which is (batch, num_heads, num_tokens, num_tokens)
        # att scores represent token to token attn needed.
        attn_scores = Q @ K.transpose(2, 3)  # Dot product for each head

        # Step 8: Find the attention weights
        # take the attn scores matrix, and replace all elements above diagonal with -inf
        # apply softmax, -inf will be changed to zero

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        d = self.head_dim ** .5
        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / d ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 9 : Compute the contex vectors:
        # att weights (b, num_heads, num_tokens, num_tokens) X V (b, num_heads, num_tokens, head_dim)
        # result = (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ V

        # Step 10: Reformat the context vect.
        # Since the output shape should be 3X6 (num_tokens X d_out)
        # so we need to swap num_head and num_tokens to make num_heads near head_dim and then
        # merge num_head and head_dim again to be d_out. since d_out = head_dim * num_heads
        # move from (1,2,3,3) -> to (1,3,2,3) -> to (1,3,6)
        context_vec = context_vec.transpose(1,2) # move from (1,2,3,3) -> to (1,3,2,3)
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # contiguous() makes sure when we reshape matrices they are in the same block of memory.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)


        # Step 11: Merge output heads horizontally together
        # combine information across heads
        # final result is 3*6 matrix  3 num_tokens  and 6 is 3 and 3 each is out_dim from each head.
        context_vec = self.out_proj(context_vec)  # optional projection
        # print(context_vec)
        return context_vec



class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits



def generate_text_simple(model, batch, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        batch_cond = batch[:, -context_size:]

        # Get the predictions
        # with torch.no_grad(): “Don’t compute gradients for any operations inside this block.”
        with torch.no_grad():
            logits = model(batch_cond)


        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]


        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)


        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        batch = torch.cat((batch, idx_next), dim=1)  # (batch, n_tokens+1)

    return batch