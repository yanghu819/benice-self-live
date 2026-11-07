"""Self-live: a minimal, self-contained reference.

BeNice! (Prairie et al., 2026) post-trains a looped LM with
    L = CE(y, p^T) + gamma * CE(y, p^(1<-T)),
where p^(1<-T) runs the core block once with first-recurrence queries/residuals
and final-recurrence keys/values. With a plain causal mask, query t also reads
its *own* final-recurrence K/V, an entry a shared-cache decoder never has
(token t's later recurrences have not run yet). Self-live keeps the
final-recurrence history (keys j < t) and gives the diagonal the live K/V.

Run `python self_live.py` to check, on a tiny random looped transformer, that
the self-live auxiliary pass (all T recurrences unrolled, history iterated to
its fixed point) reproduces token-by-token shared-cache decoding exactly, while
the causal (leaked-diagonal) pass does not.
"""
import torch
from torch import nn
from torch.nn import functional as F


def self_live_mask(n, device=None):
    """Boolean mask over keys [final-recurrence history (n) | live self (n)]:
    query t reads history j < t and its own live entry j == t."""
    history = torch.ones(n, n, dtype=torch.bool, device=device).tril(-1)
    return torch.cat([history, torch.eye(n, dtype=torch.bool, device=device)], dim=1)


def aux_attention(q, k_live, v_live, k_final, v_final, self_live=True):
    """Auxiliary attention for one layer; all tensors are (batch, heads, n, head_dim).

    q, k_live, v_live: projections of the current (live) recurrence.
    k_final, v_final:  projections of the final recurrence (detached inputs).
    """
    if not self_live:  # BeNice!: query t also reads k_final[t] (the leaked diagonal)
        return F.scaled_dot_product_attention(q, k_final, v_final, is_causal=True)
    mask = self_live_mask(q.shape[-2], q.device)
    return F.scaled_dot_product_attention(
        q, torch.cat([k_final, k_live], -2), torch.cat([v_final, v_live], -2), attn_mask=mask)


class Layer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def project(self, x):
        b, n, d = x.shape
        return [t.view(b, n, self.heads, d // self.heads).transpose(1, 2) for t in self.qkv(x).chunk(3, -1)]

    def finish(self, h, y):
        b, _, n, _ = y.shape
        h = h + self.out(y.transpose(1, 2).reshape(b, n, -1))
        return h + self.mlp(self.norm2(h))


class LoopedLM(nn.Module):
    """A toy looped LM: one block of `layers` layers applied for T recurrences."""

    def __init__(self, vocab=97, d=64, heads=4, layers=2, T=4, max_len=64):
        super().__init__()
        self.T = T
        self.embed, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.layers = nn.ModuleList(Layer(d, heads) for _ in range(layers))
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def inputs(self, ids, start=0):
        return self.embed(ids) + self.pos(torch.arange(start, start + ids.shape[1], device=ids.device))

    def forward(self, ids):
        """Standard (unshared) forward. Also returns each layer's final-recurrence
        attention input, from which the auxiliary pass recomputes K_T/V_T."""
        h, final_inputs = self.inputs(ids), []
        for r in range(self.T):
            for layer in self.layers:
                x = layer.norm1(h)
                if r == self.T - 1:
                    final_inputs.append(x)
                q, k, v = layer.project(x)
                h = layer.finish(h, F.scaled_dot_product_attention(q, k, v, is_causal=True))
        return self.head(self.norm(h)), final_inputs

    def aux_forward(self, ids, history, rounds=1, self_live=True, capture=False):
        """Auxiliary pass: live queries/residuals, keys/values from `history`
        (per-layer final-recurrence attention inputs). rounds=1 is BeNice!'s p^(1<-T)."""
        h, captured = self.inputs(ids), []
        for r in range(rounds):
            for layer, x_final in zip(self.layers, history):
                x = layer.norm1(h)
                if capture and r == rounds - 1:
                    captured.append(x)
                q, k, v = layer.project(x)
                _, k_final, v_final = layer.project(x_final)
                h = layer.finish(h, aux_attention(q, k, v, k_final, v_final, self_live))
        logits = self.head(self.norm(h))
        return (logits, captured) if capture else logits

    @torch.no_grad()
    def decode_shared(self, ids):
        """Strict token-by-token decoding with ONE K/V slot per (layer, position).
        Every recurrence of token t reads the earlier tokens' final-recurrence K/V
        plus its own live K/V; the last write (recurrence T) is what later tokens see."""
        cache_k, cache_v = [[] for _ in self.layers], [[] for _ in self.layers]
        logits = []
        for t in range(ids.shape[1]):
            h = self.inputs(ids[:, t:t + 1], start=t)
            for r in range(self.T):
                for i, layer in enumerate(self.layers):
                    q, k, v = layer.project(layer.norm1(h))
                    keys = torch.cat(cache_k[i] + [k], -2)
                    values = torch.cat(cache_v[i] + [v], -2)
                    h = layer.finish(h, F.scaled_dot_product_attention(q, keys, values))
                    if r == self.T - 1:
                        cache_k[i].append(k)
                        cache_v[i].append(v)
            logits.append(self.head(self.norm(h)))
        return torch.cat(logits, 1)


def benice_loss(model, ids, gamma=0.01, self_live=True):
    """CE(p^T) + gamma * CE(p^(1<-T)); self_live=False is BeNice!'s causal mask."""
    logits, final_inputs = model(ids)
    aux_logits = model.aux_forward(ids, [x.detach() for x in final_inputs], rounds=1, self_live=self_live)
    ce = lambda z: F.cross_entropy(z[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
    return ce(logits) + gamma * ce(aux_logits)


@torch.no_grad()
def fixed_point_aux(model, ids, self_live):
    """Unroll all T recurrences and iterate the history to its fixed point. Position t's
    history depends only on positions < t, so n sweeps make every position exact."""
    _, history = model(ids)
    for _ in range(ids.shape[1]):
        _, history = model.aux_forward(ids, history, model.T, self_live, capture=True)
    return model.aux_forward(ids, history, model.T, self_live)


if __name__ == '__main__':
    torch.manual_seed(0)
    model = LoopedLM().double().eval()
    ids = torch.randint(0, 97, (2, 24))
    shared = model.decode_shared(ids)
    unshared, _ = model(ids)
    print(f'shared vs unshared decoding, max |logit diff|: {(shared - unshared).abs().max():.3e}')
    for self_live in (True, False):
        aux = fixed_point_aux(model, ids, self_live)
        name = 'self-live  ' if self_live else 'causal mask'
        print(f'{name} aux pass vs shared decoding: max |logit diff| {(aux - shared).abs().max():.3e}, '
              f'top-1 agreement {(aux.argmax(-1) == shared.argmax(-1)).double().mean():.0%}')
    model.train()
    print(f'training loss, self-live: {benice_loss(model, ids, self_live=True).item():.4f}')
