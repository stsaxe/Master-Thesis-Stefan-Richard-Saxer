import math
import random
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Build a causal self-attention mask that also masks padding tokens.

    Args:
        tokens: LongTensor of shape [B, T]
        pad_id: token id used for padding

    Returns:
        mask: BoolTensor of shape [B, 1, T, T]
              True  -> attention allowed
              False -> attention blocked

    Semantics:
      - causal: position i may only attend to positions <= i
      - padding keys are masked out, so no token attends to PAD positions
      - padding queries are also masked out, so PAD rows become all False
    """
    B, T = tokens.shape
    device = tokens.device

    # [T, T], lower triangular
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    # valid token positions: [B, T]
    valid = tokens != pad_id

    # key mask: can attend only to valid key positions
    key_mask = valid[:, None, None, :]   # [B, 1, 1, T]

    # query mask: only valid queries should produce attention rows
    query_mask = valid[:, None, :, None] # [B, 1, T, 1]

    # broadcast causal to [1, 1, T, T]
    causal = causal[None, None, :, :]

    mask = causal & key_mask & query_mask
    return mask


class HydraLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, p_in: int, p_out: Optional[int] = None) -> torch.Tensor:
        if p_out is None:
            w = self.weight[:, :p_in]
            b = self.bias
        else:
            w = self.weight[:p_out, :p_in]
            b = self.bias[:p_out] if self.bias is not None else None
        return F.linear(x[..., :p_in], w, b)


class HydraLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor, p: int) -> torch.Tensor:
        return F.layer_norm(
            x[..., :p],
            normalized_shape=(p,),
            weight=self.weight[:p],
            bias=self.bias[:p],
            eps=self.eps,
        )


class HydraAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_heads: int,
        head_dim: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.max_heads = max_heads
        self.head_dim = head_dim
        self.d_model = d_model

        self.qkv = HydraLinear(d_model, 3 * d_model, bias=True)
        self.proj = HydraLinear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, p: int, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = p // self.head_dim
        B, T, _ = x.shape

        qkv = self.qkv(x, p_in=p, p_out=3 * p)
        qkv = qkv.view(B, T, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, h, T, head_dim]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, h, T, T]

        if attn_mask is not None:
            # attn_mask: [B, 1, T, T], True = allowed
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)

        # Safety for rows that are entirely masked, e.g. PAD queries.
        # softmax([-inf, -inf, ...]) can become NaN on some setups.
        if attn_mask is not None:
            fully_masked = ~attn_mask.any(dim=-1, keepdim=True)  # [B, 1, T, 1]
            attn = torch.where(fully_masked, torch.zeros_like(attn), attn)

        attn = self.attn_drop(attn)

        out = attn @ v  # [B, h, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, p)
        out = self.proj(out, p_in=p, p_out=p)
        return self.proj_drop(out)


class HydraMlp(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = HydraLinear(d_model, hidden, bias=True)
        self.fc2 = HydraLinear(hidden, d_model, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, p: int) -> torch.Tensor:
        x = self.fc1(x, p_in=p, p_out=4 * p)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x, p_in=4 * p, p_out=p)
        x = self.drop(x)
        return x


class HydraEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = HydraLayerNorm(d_model)
        self.attn = HydraAttention(d_model, max_heads, head_dim, proj_drop=drop)
        self.norm2 = HydraLayerNorm(d_model)
        self.mlp = HydraMlp(d_model, mlp_ratio, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        p: int,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x[..., :p] + self.attn(self.norm1(x, p), p, attn_mask)
        x = x[..., :p] + self.mlp(self.norm2(x, p), p)
        return x


class HydraBLETransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_id: int,
        max_heads: int = 8,
        head_dim: int = 64,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        max_len: int = 512,
        subnet_heads: Sequence[int] = (2, 4, 6, 8),
        separate_cls_heads: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.head_dim = head_dim
        self.max_heads = max_heads
        self.d_model = max_heads * head_dim
        self.subnet_heads = list(subnet_heads)
        self.p = self.d_model

        self.token_embed = nn.Embedding(vocab_size, self.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, self.d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        self.blocks = nn.ModuleList([
            HydraEncoderBlock(self.d_model, max_heads, head_dim, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = HydraLayerNorm(self.d_model)

        # Pretraining LM head: keep bias unless you explicitly want otherwise
        self.lm_head = HydraLinear(self.d_model, vocab_size, bias=True)

        # Classification-only final FC: NO bias
        self.separate_cls_heads = separate_cls_heads
        if separate_cls_heads:
            self.cls_heads = nn.ModuleDict({
                str(h): HydraLinear(self.d_model, num_classes, bias=False)
                for h in self.subnet_heads
            })
        else:
            self.cls_head = HydraLinear(self.d_model, num_classes, bias=False)

    def set_active_heads(self, h: int) -> None:
        assert h in self.subnet_heads
        self.p = h * self.head_dim

    def sample_active_heads(self, weights=None) -> int:
        if weights is None:
            h = random.choice(self.subnet_heads)
        else:
            h = random.choices(self.subnet_heads, weights=weights, k=1)[0]
        self.set_active_heads(h)
        return h

    def forward_backbone(self, tokens: torch.Tensor, causal: bool = False, use_cls_token: bool = True) -> torch.Tensor:
        """
        Args:
            tokens: [B, T]
            causal: whether to apply causal masking
            use_cls_token: classification uses CLS, pretraining usually does not

        Returns:
            x: [B, T(+1), p]
        """
        B, T = tokens.shape

        x = self.token_embed(tokens)[:, :, :self.p]

        if use_cls_token:
            cls = self.cls_token[:, :, :self.p].expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embed[:, :T + 1, :self.p]

            if causal:
                # Pretraining usually should not use CLS.
                # This branch is supported, but typically avoided.
                cls_tokens = torch.full(
                    (B, 1),
                    fill_value=-1,  # definitely not equal to pad_id
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                mask_tokens = torch.cat([cls_tokens, tokens], dim=1)
                attn_mask = build_causal_mask(mask_tokens, pad_id=self.pad_id)
            else:
                attn_mask = None
        else:
            x = x + self.pos_embed[:, :T, :self.p]
            attn_mask = build_causal_mask(tokens, pad_id=self.pad_id) if causal else None

        for blk in self.blocks:
            x = blk(x, p=self.p, attn_mask=attn_mask)

        x = self.norm(x, p=self.p)
        return x

    def forward_pretrain(self, tokens: torch.Tensor):
        """
        Next-token prediction.

        Input tokens:  [B, T]
        Predict token i+1 from tokens <= i

        Returns:
            logits:  [B, T-1, vocab_size]
            targets: [B, T-1]
        """
        x = self.forward_backbone(tokens[:, :-1], causal=True, use_cls_token=False)
        logits = self.lm_head(x, p_in=self.p, p_out=None)
        targets = tokens[:, 1:]
        return logits, targets

    def forward_classify(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.forward_backbone(tokens, causal=False, use_cls_token=True)
        cls = x[:, 0, :]
        h = self.p // self.head_dim

        if self.separate_cls_heads:
            logits = self.cls_heads[str(h)](cls, p_in=self.p, p_out=None)
        else:
            logits = self.cls_head(cls, p_in=self.p, p_out=None)
        return logits

    def forward(self, tokens: torch.Tensor, pretrain: bool = False):
        """
        Standard forward entry point.

        Args:
            tokens: [B, T]
            pretrain:
                True  -> returns (logits, targets) for next-token training
                False -> returns classification logits

        Returns:
            pretrain=False: logits [B, num_classes]
            pretrain=True:  (logits [B, T-1, vocab_size], targets [B, T-1])
        """
        if pretrain:
            return self.forward_pretrain(tokens)
        return self.forward_classify(tokens)


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HydraBLETransformer(
        vocab_size=2048,
        num_classes=16,
        pad_id=0,
        max_heads=8,
        head_dim=64,
        depth=4,
        max_len=2048,
        subnet_heads=(1, 2, 4, 8),
        separate_cls_heads=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = 50
    seq_len = 32*32

    # Random token batch on device
    tokens = torch.randint(
        low=1,  # avoid PAD for the random content
        high=1000,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    # Optional: add some padding to simulate variable-length sequences
    #tokens[:, -8:] = 0

    # Choose one active Hydra subnetwork
    model.set_active_heads(8)

    model.train()
    optimizer.zero_grad()

    # Pretraining forward pass
    lm_logits, targets = model(tokens, pretrain=True)

    # Next-token loss, ignoring padding targets
    loss = F.cross_entropy(
        lm_logits.reshape(-1, lm_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=model.pad_id,
    )

    print("loss:", loss.item())


    loss.backward()
    optimizer.step()

    # Pretraining forward pass
    lm_logits, targets = model(tokens, pretrain=True)

    # Next-token loss, ignoring padding targets
    loss = F.cross_entropy(
        lm_logits.reshape(-1, lm_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=model.pad_id,
    )

    print("new loss:", loss.item())





    print("device:", device)
    print("tokens shape:", tokens.shape)
    print("lm_logits shape:", lm_logits.shape)
    print("targets shape:", targets.shape)
    print("loss:", loss.item())




    def count_parameters(model, trainable_only: bool = False) -> int:

        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")

    torch.cuda.empty_cache()