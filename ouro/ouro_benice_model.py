"""Ouro adaptation of BeNice!'s cache-swap cross-entropy objective, with self-live.

Usage: wrap a loaded Ouro model (`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`):
    objective = OuroBeNiceObjective(model, gamma=0.1, aux_self_live=True, aux_rounds=4, aux_history_iterations=1)
    total, ce, aux = objective(input_ids)
`aux_self_live=False, aux_rounds=1, aux_history_iterations=0` is our port of BeNice!.

The main path uses four complete recurrent rounds.  The auxiliary path starts
from detached initial embeddings and makes one pass whose queries/residuals
are live, while each layer's K/V projections consume its detached final-round
attention input.  This is a training objective, not a shared-cache decoder.
"""
from __future__ import annotations

from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class OuroBeNiceObjective(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.01,
        gradient_checkpointing: bool = True,
        aux_self_live: bool = False,
        aux_rounds: int = 1,
        aux_history_iterations: int = 0,
    ) -> None:
        super().__init__()
        self.model = model.float()
        self.model.requires_grad_(True)
        self.model.model.early_exit_gate.requires_grad_(False)
        self.gamma = float(gamma)
        # Shared-cache decoding has no final-round K/V for the current token in
        # an early round; its own diagonal entry uses the live K/V instead.
        self.aux_self_live = aux_self_live
        self.aux_rounds = aux_rounds
        self.aux_history_iterations = aux_history_iterations
        self.gradient_checkpointing = gradient_checkpointing
        self.num_steps = 4
        official = import_module(type(model).__module__)
        self._apply_rotary = official.apply_rotary_pos_emb
        self._create_causal_mask = official.create_causal_mask
        self._attention_interface = official.ALL_ATTENTION_FUNCTIONS[
            model.config._attn_implementation
        ]

    @staticmethod
    def _autocast(input_ids: torch.Tensor):
        # Keep the trainable master weights FP32; only CUDA matmuls use BF16.
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=input_ids.is_cuda
        )

    @property
    def _checkpoint_enabled(self) -> bool:
        return self.gradient_checkpointing and self.training and torch.is_grad_enabled()

    @staticmethod
    def _ce(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )

    @staticmethod
    def _full_layer(
        hidden: torch.Tensor,
        layer: nn.Module,
        current_ut: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        capture: bool,
    ):
        attention_input = layer.input_layernorm(hidden)
        attention_output, _ = layer.self_attn(
            hidden_states=attention_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=None,
            current_ut=current_ut,
        )
        hidden = hidden + layer.input_layernorm_2(attention_output)
        hidden = hidden + layer.post_attention_layernorm_2(
            layer.mlp(layer.post_attention_layernorm(hidden))
        )
        if capture:
            # Detach at capture time: neither the list nor the auxiliary loss
            # retains the producer graph through these teacher activations.
            return hidden, attention_input.detach()
        return hidden

    def _full_pass(self, input_ids: torch.Tensor, *, capture: bool):
        decoder = self.model.model
        hidden = decoder.embed_tokens(input_ids)
        initial = hidden.detach() if capture else None
        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        # Match the official mask builder as well as its causal semantics: an
        # explicit mask and is_causal-only can select different BF16 kernels.
        attention_mask = self._create_causal_mask(
            config=self.model.config,
            input_embeds=hidden,
            attention_mask=None,
            cache_position=position_ids[0],
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = decoder.rotary_emb(hidden, position_ids)
        final_inputs = []
        for current_ut in range(self.num_steps):
            capture_round = capture and current_ut == self.num_steps - 1
            for layer in decoder.layers:
                # Defaults bind this particular layer/round for recomputation.
                def run_layer(
                    x, layer=layer, current_ut=current_ut, capture_round=capture_round
                ):
                    return self._full_layer(
                        x, layer, current_ut, position_embeddings, attention_mask, capture_round
                    )

                if self._checkpoint_enabled:
                    output = checkpoint(run_layer, hidden, use_reentrant=False)
                else:
                    output = run_layer(hidden)
                if capture_round:
                    hidden, final_input = output
                    final_inputs.append(final_input)
                else:
                    hidden = output
            hidden = decoder.norm(hidden)
        return hidden, initial, final_inputs, position_embeddings, attention_mask

    def _aux_layer(
        self,
        hidden: torch.Tensor,
        final_input: torch.Tensor,
        layer: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attention = layer.self_attn
        query_input = layer.input_layernorm(hidden)
        batch, sequence = hidden.shape[:2]
        head_shape = (batch, sequence, -1, attention.head_dim)
        query = attention.q_proj(query_input).view(head_shape).transpose(1, 2)
        # The input is detached, but the shared projection weights remain live.
        key = attention.k_proj(final_input).view(head_shape).transpose(1, 2)
        value = attention.v_proj(final_input).view(head_shape).transpose(1, 2)
        if self.aux_self_live:
            live_key = attention.k_proj(query_input).view(head_shape).transpose(1, 2)
            live_value = attention.v_proj(query_input).view(head_shape).transpose(1, 2)
            query, key = self._apply_rotary(query, key, *position_embeddings)
            _, live_key = self._apply_rotary(query, live_key, *position_embeddings)
            # Keys [teacher history | live self]: query t reads teacher j < t and live j == t.
            mask = torch.ones(sequence, sequence, device=hidden.device, dtype=torch.bool)
            mask = torch.cat((mask.tril(-1), torch.eye(sequence, device=hidden.device, dtype=torch.bool)), dim=1)
            output = F.scaled_dot_product_attention(
                query,
                torch.cat((key, live_key), dim=2),
                torch.cat((value, live_value), dim=2),
                attn_mask=mask,
                scale=attention.scaling,
            ).transpose(1, 2)
        else:
            query, key = self._apply_rotary(query, key, *position_embeddings)
            output, _ = self._attention_interface(
                attention,
                query,
                key,
                value,
                attention_mask,
                dropout=attention.attention_dropout if attention.training else 0.0,
                scaling=attention.scaling,
                sliding_window=attention.sliding_window,
            )
        output = output.reshape(batch, sequence, -1).contiguous()
        hidden = hidden + layer.input_layernorm_2(attention.o_proj(output))
        hidden = hidden + layer.post_attention_layernorm_2(
            layer.mlp(layer.post_attention_layernorm(hidden))
        )
        return hidden

    def aux_hidden(self, hidden, final_inputs, position_embeddings, attention_mask, capture=False):
        # Every aux round reads the same final-round teacher history, as the
        # last-write shared cache does for all recurrences of a later token.
        captured = []
        for round_index in range(self.aux_rounds):
            for layer, final_input in zip(self.model.model.layers, final_inputs):
                if capture and round_index == self.aux_rounds - 1:
                    captured.append(layer.input_layernorm(hidden).detach())
                def run_aux(x, source, layer=layer):
                    return self._aux_layer(
                        x, source, layer, position_embeddings, attention_mask
                    )

                if self._checkpoint_enabled:
                    hidden = checkpoint(
                        run_aux, hidden, final_input, use_reentrant=False
                    )
                else:
                    hidden = run_aux(hidden, final_input)
            hidden = self.model.model.norm(hidden)
        return (hidden, captured) if capture else hidden

    def full_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Fixed-four-round logits for comparison with the official model."""
        with self._autocast(input_ids):
            hidden, _, _, _, _ = self._full_pass(input_ids, capture=False)
            return self.model.lm_head(hidden)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._autocast(input_ids):
            hidden, initial, final_inputs, position_embeddings, attention_mask = self._full_pass(
                input_ids, capture=True
            )
            full_logits = self.model.lm_head(hidden)
            ce = self._ce(full_logits, input_ids)
            # Replace teacher history by the final-round inputs of a no-grad
            # shared pass: one Jacobi step toward self-generated decode history.
            for _ in range(self.aux_history_iterations):
                with torch.no_grad():
                    _, final_inputs = self.aux_hidden(
                        initial, final_inputs, position_embeddings, attention_mask, capture=True
                    )
            hidden = self.aux_hidden(initial, final_inputs, position_embeddings, attention_mask)
            aux_logits = self.model.lm_head(hidden)
            aux = self._ce(aux_logits, input_ids)
            total = ce + self.gamma * aux
        return total, ce, aux
