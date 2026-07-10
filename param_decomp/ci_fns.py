"""Causal-importance function configs, CI-fn modules, and wrappers."""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Self, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt, model_validator
from torch import Tensor, nn

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.ci_nn_blocks import Linear, ParallelLinear, TransformerBlock
from param_decomp.components import Components, EmbeddingComponents, get_module_input_dim

LayerwiseCiFnType = Literal["mlp", "vector_mlp", "shared_mlp"]
GlobalCiFnType = Literal["global_shared_mlp", "global_shared_transformer"]


class LayerwiseCiConfig(BaseConfig):
    """Layerwise CI fns — one independent CI fn per decomposition target."""

    mode: Literal["layerwise"] = "layerwise"
    fn_type: LayerwiseCiFnType = Field(
        ..., description="Type of layerwise CI function: mlp, vector_mlp, or shared_mlp"
    )
    hidden_dims: list[PositiveInt] = Field(
        ..., description="Hidden dimensions for the CI function MLP"
    )

    @model_validator(mode="after")
    def validate_hidden_dims(self) -> Self:
        if self.fn_type in ("mlp", "vector_mlp") and not self.hidden_dims:
            raise ValueError(f"hidden_dims must be non-empty for fn_type={self.fn_type!r}")
        return self


class AttnConfig(BaseConfig):
    """Self-attention config for the transformer CI fn. Uses RoPE for length generalization."""

    n_heads: PositiveInt = Field(
        ...,
        description="Number of attention heads. Must divide the input dimension.",
    )
    max_len: PositiveInt = Field(
        default=2048,
        description="Maximum sequence length for RoPE embeddings.",
    )
    rope_base: float = Field(
        default=10000.0,
        description="Base for RoPE frequency computation.",
    )


class GlobalSharedTransformerCiConfig(BaseConfig):
    """Config for the global transformer CI fn.

    `d_model` must be divisible by `attn_config.n_heads` and the resulting per-head dim
    must be even (RoPE). `mlp_hidden_dim` defaults to `[4 * d_model]`.
    """

    d_model: PositiveInt
    n_blocks: PositiveInt
    mlp_hidden_dim: list[PositiveInt] | None = Field(
        default=None,
        description="Hidden dimension for transformer MLP blocks. "
        "If None, defaults to [4 * d_model].",
    )
    attn_config: AttnConfig

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        assert self.d_model % self.attn_config.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"attn_config.n_heads ({self.attn_config.n_heads})"
        )
        d_head = self.d_model // self.attn_config.n_heads
        assert d_head % 2 == 0, (
            f"d_head ({d_head}) must be even for RoPE. "
            f"d_model={self.d_model}, "
            f"n_heads={self.attn_config.n_heads}"
        )
        return self


class GlobalCiConfig(BaseConfig):
    """A single global CI fn that maps all layers jointly."""

    mode: Literal["global"] = "global"
    fn_type: GlobalCiFnType = Field(
        ...,
        description="Type of global CI function: global_shared_mlp or global_shared_transformer",
    )
    hidden_dims: list[PositiveInt] | None = Field(
        default=None,
        description="Hidden dimensions for global_shared_mlp CI function.",
    )
    simple_transformer_ci_cfg: GlobalSharedTransformerCiConfig | None = None

    @model_validator(mode="after")
    def validate_ci_config(self) -> Self:
        if self.fn_type == "global_shared_mlp":
            assert self.hidden_dims is not None, (
                "hidden_dims must be specified when fn_type='global_shared_mlp'"
            )
        elif self.fn_type == "global_shared_transformer":
            assert self.simple_transformer_ci_cfg is not None, (
                "simple_transformer_ci_cfg must be specified when fn_type='global_shared_transformer'"
            )
            assert self.hidden_dims is None, (
                "hidden_dims is only used for fn_type='global_shared_mlp'"
            )
        return self


class SpikeGatedCiConfig(BaseConfig):
    """Stage-1 spike-gated probabilistic CI bottleneck (gate-only, `D=0`).

    A shared encoder maps the concatenated decomposition-target inputs to `n_mechanisms`
    gate logits; the gate maps logits to `z ∈ [0,1]^K`; a linear decoder `B ∈ R^{M×K}` maps
    `z` to per-component pre-sigmoid CI logits, split back per layer. Two gate types:
    `hard_concrete` (stochastic spike, deterministic at eval; `docs/slpd_stage1_spike_design.md`)
    and `deterministic` (`z = sigmoid(logits) = π`, identical in train/eval, the minimal
    first-pass gate; `docs/slpd_stage1_minimal_ci.md`).
    """

    mode: Literal["spike_gated"] = "spike_gated"
    encoder_hidden_dims: list[PositiveInt] = Field(
        ...,
        description="Hidden dims of the shared encoder trunk (empty list ⇒ a linear encoder).",
    )
    n_mechanisms: PositiveInt = Field(
        ..., description="Number of latent mechanisms K (overcomplete vs. expected count)."
    )
    gate_type: Literal["hard_concrete", "deterministic", "straight_through"] = Field(
        default="hard_concrete",
        description="`hard_concrete` = stochastic spike; `deterministic` = z=sigmoid(logits); "
        "`straight_through` = Bernoulli(σ(ℓ)) sample with a σ'(ℓ) straight-through backward "
        "(no temperature/stretch; the minimal-bottleneck gate).",
    )
    hard_concrete_temp: PositiveFloat = Field(
        default=0.5,
        description="Binary/hard-concrete temperature τ (the start value when annealing).",
    )
    hard_concrete_temp_final: PositiveFloat | None = Field(
        default=None,
        description="Final τ for linear annealing from `hard_concrete_temp`; None ⇒ constant τ. "
        "Lower τ sharpens the gate toward {0,1}; anneal from soft (low-variance) to sharp.",
    )
    temp_anneal_start_frac: Probability = Field(
        default=0.0, description="Fraction of training at which τ annealing begins."
    )
    temp_anneal_end_frac: Probability = Field(
        default=1.0,
        description="Fraction of training at which τ reaches `hard_concrete_temp_final`.",
    )
    hard_concrete_stretch: NonNegativeFloat = Field(
        default=0.1, description="Hard-concrete stretch s; the interval is (γ,ζ)=(-s, 1+s)."
    )
    slab_sigma0: NonNegativeFloat = Field(
        default=0.0,
        description="Multiplicative slab jitter std (0 ⇒ z=γ, a clean binary gate).",
    )
    decoder_nonneg: bool = Field(
        default=False,
        description="If True, B is projected to ≥0 after each optimizer step (projected gradient) "
        "so mechanisms only turn components on and off-wirings reach exactly 0.",
    )
    decoder_init_std: NonNegativeFloat = Field(
        default=0.1,
        description="Std of the N(0, std) initialization of the decoder B (fixed, not fan-scaled).",
    )
    encoder_head_init_scale: PositiveFloat = Field(
        default=1.0,
        description="Multiplier applied to the final encoder Linear (logit-head) weight after "
        "init; scales the initial logit spread hence the initial per-mechanism π diversity. "
        "1.0 ⇒ unchanged.",
    )
    center_logits: bool = Field(
        default=False,
        description="If True, center the encoder logits per datapoint (subtract the mean over "
        "mechanisms) before the gate, making the gate shift-invariant. Kills the common-mode "
        "'raise all logits' direction (the mildest form of top-K's rank invariance).",
    )
    center_logits_until_frac: Probability = Field(
        default=1.0,
        description="Fraction of training over which `center_logits` stays active; centering "
        "switches off once `current_frac >= center_logits_until_frac`. 1.0 ⇒ whole run.",
    )


# Discriminated union (by `mode`) of every CI-fn config the trainer accepts. Pydantic
# picks the right branch from the YAML `pd.ci_config.mode` literal.
CiConfig = LayerwiseCiConfig | GlobalCiConfig | SpikeGatedCiConfig


class MLPCiFn(nn.Module):
    """Per-component scalar-input MLP CI fn.

    Each of `C` components gets its own MLP mapping a scalar component activation to a
    scalar CI value; built from `ParallelLinear` layers operating on a singleton last dim.
    """

    def __init__(self, C: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = 1 if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        x = einops.rearrange(x, "... C -> ... C 1")
        x = self.layers(x)
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class VectorMLPCiFn(nn.Module):
    """Per-component vector-input MLP CI fn.

    Each of `C` components gets its own MLP consuming the full `[..., d_in]` layer input;
    built from `ParallelLinear` so all `C` networks run in one batched einsum.
    """

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())

        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        x = self.layers(einops.rearrange(x, "... d_in -> ... 1 d_in"))
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class VectorSharedMLPCiFn(nn.Module):
    """Shared MLP `[..., d_in] -> [..., C]`.

    All components share every hidden layer; only the final projection splits
    per-component.
    """

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(Linear(in_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        final_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        self.layers.append(Linear(final_dim, C, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        return self.layers(x)


class GlobalSharedMLPCiFn(nn.Module):
    """Global MLP over all layers.

    Concatenates all decomposition-target inputs along the feature dim, runs one shared
    MLP, then splits the output back into per-layer `[..., C]` slices. Layer order is
    fixed by sorted layer name so concatenation is deterministic.
    """

    def __init__(
        self,
        layer_configs: dict[str, tuple[int, int]],  # layer_name -> (input_dim, C)
        hidden_dims: list[int],
    ):
        super().__init__()

        self.layer_order = sorted(layer_configs.keys())
        self.layer_configs = layer_configs
        self.split_sizes = [layer_configs[name][1] for name in self.layer_order]

        total_input_dim = sum(input_dim for input_dim, _ in layer_configs.values())
        total_C = sum(C for _, C in layer_configs.values())

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            in_dim = total_input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(Linear(in_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        final_dim = hidden_dims[-1] if len(hidden_dims) > 0 else total_input_dim
        self.layers.append(Linear(final_dim, total_C, nonlinearity="linear"))

    @override
    def forward(
        self,
        input_acts: dict[str, Float[Tensor, "... d_in"]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        inputs_list = [input_acts[name] for name in self.layer_order]
        concatenated = torch.cat(inputs_list, dim=-1)
        output = self.layers(concatenated)
        split_outputs = torch.split(output, self.split_sizes, dim=-1)
        return {name: split_outputs[i] for i, name in enumerate(self.layer_order)}


class SpikeGatedCiFn(nn.Module):
    """Spike-gated probabilistic CI bottleneck (Stage-1: gate-only, `D=0`).

    Same dict-in/dict-out contract as `GlobalSharedMLPCiFn` so it sits behind
    `GlobalCiFnWrapper` unchanged. A shared encoder produces `K` gate logits from the
    concatenated layer inputs; the gate maps logits to `z ∈ [0,1]^K` — either `hard_concrete`
    (stochastic spike when training, the median gate `sigmoid(logits)·(ζ-γ)+γ` clamped at eval)
    or `deterministic` (`z = sigmoid(logits) = π`, train/eval identical); a linear decoder
    `B ∈ R^{M×K}` maps `z` to per-component pre-sigmoid logits, split back per layer.

    Side effects consumed by the Stage-1 losses: every forward caches the per-mechanism gate
    probabilities `π = sigmoid(logits)` on `self._pi` (grad-attached) for `SpikeGateKLLoss`,
    and `self.B` is read directly by `DecoderColumnMassLoss`. Both rely on the CI fn forward
    running exactly once per training step (inside `calc_causal_importances`).
    """

    def __init__(
        self,
        layer_configs: dict[str, tuple[int, int]],  # layer_name -> (input_dim, C)
        encoder_hidden_dims: list[int],
        n_mechanisms: int,
        gate_type: str,
        hard_concrete_temp: float,
        hard_concrete_temp_final: float | None,
        temp_anneal_start_frac: float,
        temp_anneal_end_frac: float,
        hard_concrete_stretch: float,
        slab_sigma0: float,
        decoder_nonneg: bool,
        decoder_init_std: float,
        encoder_head_init_scale: float,
        center_logits: bool,
        center_logits_until_frac: float,
    ):
        super().__init__()
        self.layer_order = sorted(layer_configs.keys())
        self.split_sizes = [layer_configs[name][1] for name in self.layer_order]
        total_input_dim = sum(input_dim for input_dim, _ in layer_configs.values())
        self.M = sum(C for _, C in layer_configs.values())
        self.K = n_mechanisms
        self.gate_type = gate_type
        # `temp` is mutated per-step by `anneal_temperature`; `_temp_start` is the fixed start.
        self.temp = hard_concrete_temp
        self._temp_start = hard_concrete_temp
        self._temp_final = hard_concrete_temp_final
        self._temp_anneal_start_frac = temp_anneal_start_frac
        self._temp_anneal_end_frac = temp_anneal_end_frac
        self.stretch = hard_concrete_stretch
        self.slab_sigma0 = slab_sigma0
        self.decoder_nonneg = decoder_nonneg
        self.decoder_init_std = decoder_init_std
        # `_center_active` is refreshed per-step from `current_frac` by `anneal_temperature`.
        self.center_logits = center_logits
        self._center_logits_until_frac = center_logits_until_frac
        self._center_active = True

        self.encoder = nn.Sequential()
        for i in range(len(encoder_hidden_dims)):
            in_dim = total_input_dim if i == 0 else encoder_hidden_dims[i - 1]
            self.encoder.append(Linear(in_dim, encoder_hidden_dims[i], nonlinearity="relu"))
            self.encoder.append(nn.GELU())
        final_dim = encoder_hidden_dims[-1] if encoder_hidden_dims else total_input_dim
        self.encoder.append(Linear(final_dim, n_mechanisms, nonlinearity="linear"))
        if encoder_head_init_scale != 1.0:
            with torch.no_grad():
                self.encoder[-1].W.mul_(encoder_head_init_scale)

        self.B = nn.Parameter(torch.empty(self.M, n_mechanisms))
        nn.init.normal_(self.B, std=self.decoder_init_std)
        if self.decoder_nonneg:
            # Start every wiring positive ("on"); the penalty + recon prune toward 0 under the
            # per-step projection (`project_nonneg`). Same magnitude as the signed init.
            self.B.data.abs_()

        # Cached per forward for SpikeGateKLLoss; None until the first forward. `_logits` backs
        # the hard-concrete open-probability `gate_open_prob()`; `_pi = sigmoid(_logits)`.
        self._pi: Float[Tensor, "... K"] | None = None
        self._logits: Float[Tensor, "... K"] | None = None

        # Opt-in diagnostics: when true, `forward` retains grad on the sampled gate `z` and the
        # decoder pre-sigmoid `η`, exposing `∂L/∂z_k` and `∂L/∂η_c`. Both hold the last forward's
        # tensor (a step runs several forwards); see `slpd/collapse_diagnostics.py`.
        self._capture_grads: bool = False
        self._captured_gate: Tensor | None = None
        self._captured_pre_sigmoid: Tensor | None = None

        # When true, the hard-concrete gate uses its noise-off (median) branch even in
        # training mode. Set transiently via `force_deterministic_gate` so the adversarial
        # recon term attacks the deterministic gate `z̄` (the deployed network).
        self._force_deterministic_gate: bool = False

        # Optional oracle-gate override: when set, `forward` uses `self._oracle_z_fn(input_acts)`
        # as the gate instead of the encoder-driven sample (the encoder still runs so `_pi`/`_logits`
        # stay populated for any KL term). Used by the oracle-freeze validation experiments.
        self._oracle_z_fn: Callable[[dict[str, Tensor]], Tensor] | None = None

    @contextmanager
    def force_deterministic_gate(self) -> Iterator[None]:
        """Force the noise-off (median) gate for the duration of the block."""
        prev = self._force_deterministic_gate
        self._force_deterministic_gate = True
        try:
            yield
        finally:
            self._force_deterministic_gate = prev

    def anneal_temperature(self, current_frac: float) -> None:
        """Set `self.temp` by linearly annealing `_temp_start → _temp_final` over the configured
        fraction window. No-op when `_temp_final is None`. Called once per training step by the
        trainer (mirrors the importance-minimality p-anneal); recomputed from scratch so it is
        resume-safe. Only the stochastic (training) hard-concrete branch reads `self.temp`.
        Also refreshes `_center_active` (logit-centering window) from `current_frac`."""
        self._center_active = (
            self._center_logits_until_frac >= 1.0 or current_frac < self._center_logits_until_frac
        )
        if self._temp_final is None:
            return
        start, end = self._temp_anneal_start_frac, self._temp_anneal_end_frac
        if current_frac <= start:
            self.temp = self._temp_start
        elif current_frac >= end:
            self.temp = self._temp_final
        else:
            progress = (current_frac - start) / (end - start)
            self.temp = self._temp_start + (self._temp_final - self._temp_start) * progress

    def project_nonneg(self) -> None:
        """Clamp the decoder `B` to ≥0 in-place (projected-gradient non-negativity). No-op when
        signed. Called by the trainer after each CI-fn optimizer step, so `B` is the effective
        non-negative decoder everywhere (forward, penalty, inspection) and off-wirings reach
        exactly 0; an entry at 0 can still revive if a later gradient pushes it positive."""
        if self.decoder_nonneg:
            with torch.no_grad():
                self.B.clamp_(min=0.0)

    def _sample_gate(self, logits: Float[Tensor, "... K"]) -> Float[Tensor, "... K"]:
        """Hard-concrete gate (stochastic train / median eval), or deterministic z=sigmoid(logits)."""
        if self.gate_type == "deterministic":
            return torch.sigmoid(logits)
        if self.gate_type == "straight_through":
            pi = torch.sigmoid(logits)
            hard = (
                torch.bernoulli(pi)
                if (self.training and not self._force_deterministic_gate)
                else (pi > 0.5).to(pi.dtype)
            )
            return hard + (pi - pi.detach())  # forward = hard binary; backward ∂z/∂ℓ = σ'(ℓ)
        gamma, zeta = -self.stretch, 1.0 + self.stretch
        if self.training and not self._force_deterministic_gate:
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + logits) / self.temp)
        else:
            s = torch.sigmoid(logits)
        return (s * (zeta - gamma) + gamma).clamp(0.0, 1.0)

    def gate_open_prob(self) -> Float[Tensor, "... K"]:
        """`P(z_k > 0)` for the hard-concrete gate at the current temperature (Louizos L0):
        `sigmoid(logits - τ·log(-γ/ζ))`, the probability the stretched-and-clamped gate is open.
        Differs from `π = sigmoid(logits)` while `τ > 0`; → π as τ → 0. Deterministic gate ⇒ π."""
        assert self._logits is not None, "forward must run before gate_open_prob"
        if self.gate_type in ("deterministic", "straight_through"):
            return torch.sigmoid(
                self._logits
            )  # π is the true firing prob; no temp/stretch correction
        assert self.stretch > 0.0, "open-prob requires hard_concrete_stretch > 0"
        gamma, zeta = -self.stretch, 1.0 + self.stretch
        return torch.sigmoid(self._logits - self.temp * math.log(-gamma / zeta))

    @override
    def forward(
        self,
        input_acts: dict[str, Float[Tensor, "... d_in"]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        concatenated = torch.cat([input_acts[name] for name in self.layer_order], dim=-1)
        logits = self.encoder(concatenated)
        if self.center_logits and self._center_active:
            logits = logits - logits.mean(dim=-1, keepdim=True)
        self._logits = logits
        self._pi = torch.sigmoid(logits)
        gate = (
            self._oracle_z_fn(input_acts)
            if self._oracle_z_fn is not None
            else self._sample_gate(logits)
        )
        if self.slab_sigma0 > 0.0:
            gate = gate * (1.0 + self.slab_sigma0 * torch.randn_like(gate))
        # B is the effective decoder; non-negativity (when enabled) is enforced by `project_nonneg`
        # after each optimizer step, so off-wirings reach exactly 0 (vs. softplus, which floors at
        # ~0.69 and disagrees with the raw-B penalty).
        pre_sigmoid = einops.einsum(gate, self.B, "... K, M K -> ... M")
        if self._capture_grads:
            if gate.requires_grad:
                gate.retain_grad()
                self._captured_gate = gate
            if pre_sigmoid.requires_grad:
                pre_sigmoid.retain_grad()
                self._captured_pre_sigmoid = pre_sigmoid
        split_outputs = torch.split(pre_sigmoid, self.split_sizes, dim=-1)
        return {name: split_outputs[i] for i, name in enumerate(self.layer_order)}


@dataclass
class TargetLayerConfig:
    """Per-target metadata consumed by `GlobalSharedTransformerCiFn`."""

    input_dim: int
    C: int


class GlobalSharedTransformerCiFn(nn.Module):
    """Global transformer attending over sequence to produce per-component CI.

    Per-layer inputs are RMS-normed, concatenated along the feature dim, projected to
    `d_model`, and run through `n_layers` `TransformerBlock`s with bidirectional
    self-attention. A final linear projection produces logits which are split back into
    per-layer `[..., C]` slices in sorted-name order. For 2D inputs (e.g. TMS, resid_mlp
    — no sequence axis) a singleton sequence dim is added before the transformer and
    squeezed out after.
    """

    def __init__(
        self,
        target_model_layer_configs: dict[str, TargetLayerConfig],
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_len: int,
        mlp_hidden_dims: list[int] | None = None,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.layer_order = sorted(target_model_layer_configs.keys())
        self.target_model_layer_configs = target_model_layer_configs
        self.split_sizes = [target_model_layer_configs[name].C for name in self.layer_order]
        self.d_model = d_model
        self.n_transformer_layers = n_layers
        self.n_heads = n_heads

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [4 * d_model]

        total_input_dim = sum(config.input_dim for config in target_model_layer_configs.values())
        total_c = sum(config.C for config in target_model_layer_configs.values())

        self._input_projector = Linear(total_input_dim, d_model, nonlinearity="relu")
        self._output_head = Linear(d_model, total_c, nonlinearity="linear")

        self._blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_hidden_dims=mlp_hidden_dims,
                    max_len=max_len,
                    rope_base=rope_base,
                )
                for _ in range(n_layers)
            ]
        )

    @override
    def forward(
        self,
        input_acts: dict[str, Float[Tensor, "... d_in"]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        inputs_list = [
            F.rms_norm(input_acts[name], (input_acts[name].shape[-1],)) for name in self.layer_order
        ]
        concatenated = torch.cat(inputs_list, dim=-1)
        projected: Tensor = self._input_projector(concatenated)

        # The transformer blocks expect a sequence dimension, so we add an extra dimension to our
        # activations if we only have 2D acts (e.g. in TMS and resid_mlp).
        added_seq_dim = False
        if projected.ndim < 3:
            projected = projected.unsqueeze(-2)
            added_seq_dim = True

        x = projected
        for block in self._blocks:
            x = block(x)

        output = self._output_head(x)

        if added_seq_dim:
            output = output.squeeze(-2)

        split_outputs = torch.split(output, self.split_sizes, dim=-1)
        outputs = {name: split_outputs[i] for i, name in enumerate(self.layer_order)}

        return outputs


class LayerwiseCiFnWrapper(nn.Module):
    """Bundle a dict of per-layer CI fns behind a single dict-in/dict-out interface.

    Runs each layer's CI fn on its own input. For `ci_fn_type == "mlp"` the per-component
    scalar activations are obtained via `Components.get_component_acts` first; the other
    variants receive the raw layer input. Layer names are stored under `ModuleDict` with
    `.` replaced by `-` so state-dict keys are well-formed.
    """

    def __init__(
        self,
        ci_fns: dict[str, nn.Module],
        components: dict[str, Components],
        ci_fn_type: LayerwiseCiFnType,
    ):
        super().__init__()
        self.layer_names = sorted(ci_fns.keys())
        self.components = components
        self.ci_fn_type = ci_fn_type

        # Store as ModuleDict with "." replaced by "-" for state dict compatibility
        self._ci_fns = nn.ModuleDict(
            {name.replace(".", "-"): ci_fns[name] for name in self.layer_names}
        )

    @override
    def forward(
        self,
        layer_acts: dict[str, Float[Tensor, "..."]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        outputs: dict[str, Float[Tensor, "... C"]] = {}

        for layer_name in self.layer_names:
            ci_fn = self._ci_fns[layer_name.replace(".", "-")]
            input_acts = layer_acts[layer_name]

            # MLPCiFn expects component activations, others take raw input
            if self.ci_fn_type == "mlp":
                ci_fn_input = self.components[layer_name].get_component_acts(input_acts)
            else:
                ci_fn_input = input_acts

            outputs[layer_name] = ci_fn(ci_fn_input)

        return outputs


class GlobalCiFnWrapper(nn.Module):
    """Gives the global CI fns the same dict-in/dict-out interface as the layerwise wrapper.

    For `EmbeddingComponents` the raw input is a tensor of token ids; this wrapper
    converts them to component activations via `EmbeddingComponents.get_component_acts`
    so the global CI fn always sees floating-point activations.
    """

    def __init__(
        self,
        global_ci_fn: GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn | SpikeGatedCiFn,
        components: dict[str, Components],
    ):
        super().__init__()
        self._global_ci_fn = global_ci_fn
        self.components = components

    @override
    def forward(
        self,
        layer_acts: dict[str, Float[Tensor, "..."]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        transformed: dict[str, Float[Tensor, ...]] = {}

        for layer_name, acts in layer_acts.items():
            component = self.components[layer_name]
            if isinstance(component, EmbeddingComponents):
                # Embeddings pass token IDs; convert to component activations
                transformed[layer_name] = component.get_component_acts(acts)
            else:
                transformed[layer_name] = acts

        return self._global_ci_fn(transformed)


def _make_layerwise_ci_fn(
    target_module: nn.Module,
    C: int,
    ci_fn_type: LayerwiseCiFnType,
    ci_fn_hidden_dims: list[int],
) -> nn.Module:
    if isinstance(target_module, nn.Embedding):
        assert ci_fn_type == "mlp", "Embedding modules only supported for ci_fn_type='mlp'"

    if ci_fn_type == "mlp":
        return MLPCiFn(C=C, hidden_dims=ci_fn_hidden_dims)

    input_dim = get_module_input_dim(target_module)
    match ci_fn_type:
        case "vector_mlp":
            return VectorMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)
        case "shared_mlp":
            return VectorSharedMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)


def _make_global_ci_fn(
    target_model: nn.Module,
    module_to_c: dict[str, int],
    components: dict[str, Components],
    ci_config: GlobalCiConfig,
) -> GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn:
    ci_fn_type = ci_config.fn_type
    ci_fn_hidden_dims = ci_config.hidden_dims

    layer_configs: dict[str, tuple[int, int]] = {}
    for path, module_c in module_to_c.items():
        target_module = target_model.get_submodule(path)
        component = components[path]
        if isinstance(target_module, nn.Embedding):
            assert isinstance(component, EmbeddingComponents)
            input_dim = component.C
        else:
            input_dim = get_module_input_dim(target_module)
        layer_configs[path] = (input_dim, module_c)

    match ci_fn_type:
        case "global_shared_mlp":
            assert ci_fn_hidden_dims is not None
            return GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=ci_fn_hidden_dims)
        case "global_shared_transformer":
            transformer_cfg = ci_config.simple_transformer_ci_cfg
            assert transformer_cfg is not None
            return GlobalSharedTransformerCiFn(
                target_model_layer_configs={
                    path: TargetLayerConfig(input_dim=input_dim, C=C)
                    for path, (input_dim, C) in layer_configs.items()
                },
                d_model=transformer_cfg.d_model,
                n_layers=transformer_cfg.n_blocks,
                n_heads=transformer_cfg.attn_config.n_heads,
                mlp_hidden_dims=transformer_cfg.mlp_hidden_dim,
                max_len=transformer_cfg.attn_config.max_len,
                rope_base=transformer_cfg.attn_config.rope_base,
            )


def _make_spike_gated_ci_fn(
    target_model: nn.Module,
    module_to_c: dict[str, int],
    components: dict[str, Components],
    ci_config: SpikeGatedCiConfig,
) -> SpikeGatedCiFn:
    layer_configs: dict[str, tuple[int, int]] = {}
    for path, module_c in module_to_c.items():
        target_module = target_model.get_submodule(path)
        component = components[path]
        if isinstance(target_module, nn.Embedding):
            assert isinstance(component, EmbeddingComponents)
            input_dim = component.C
        else:
            input_dim = get_module_input_dim(target_module)
        layer_configs[path] = (input_dim, module_c)
    return SpikeGatedCiFn(
        layer_configs=layer_configs,
        encoder_hidden_dims=ci_config.encoder_hidden_dims,
        n_mechanisms=ci_config.n_mechanisms,
        gate_type=ci_config.gate_type,
        hard_concrete_temp=ci_config.hard_concrete_temp,
        hard_concrete_temp_final=ci_config.hard_concrete_temp_final,
        temp_anneal_start_frac=ci_config.temp_anneal_start_frac,
        temp_anneal_end_frac=ci_config.temp_anneal_end_frac,
        hard_concrete_stretch=ci_config.hard_concrete_stretch,
        slab_sigma0=ci_config.slab_sigma0,
        decoder_nonneg=ci_config.decoder_nonneg,
        decoder_init_std=ci_config.decoder_init_std,
        encoder_head_init_scale=ci_config.encoder_head_init_scale,
        center_logits=ci_config.center_logits,
        center_logits_until_frac=ci_config.center_logits_until_frac,
    )


def get_spike_gated_ci_fn(ci_fn: nn.Module) -> SpikeGatedCiFn:
    """Return the inner `SpikeGatedCiFn` from a `GlobalCiFnWrapper` (asserts the type).

    Used by the Stage-1 losses to reach the cached gate probabilities / decoder weights.
    """
    inner = getattr(ci_fn, "_global_ci_fn", None)
    assert isinstance(inner, SpikeGatedCiFn), (
        "expected a spike-gated CI fn (set ci_config.mode='spike_gated')"
    )
    return inner


def _maybe_spike_gated_ci_fn(ci_fn: nn.Module) -> "SpikeGatedCiFn | None":
    """The inner `SpikeGatedCiFn` if `ci_fn` wraps one, else None (non-asserting)."""
    inner = getattr(ci_fn, "_global_ci_fn", None)
    return inner if isinstance(inner, SpikeGatedCiFn) else None


@contextmanager
def maybe_force_deterministic_gate(ci_fn: nn.Module, enabled: bool) -> Iterator[None]:
    """Force the spike gate's deterministic branch when `enabled` and `ci_fn` is spike-gated.

    No-op for non-gated CI fns (already deterministic) or when `enabled` is False.
    """
    spike_fn = _maybe_spike_gated_ci_fn(ci_fn) if enabled else None
    if spike_fn is None:
        yield
    else:
        with spike_fn.force_deterministic_gate():
            yield


def make_ci_fn_wrapper(
    target_model: nn.Module,
    module_to_c: dict[str, int],
    components: dict[str, Components],
    ci_config: CiConfig,
) -> LayerwiseCiFnWrapper | GlobalCiFnWrapper:
    """Build the CI-fn wrapper selected by `ci_config`.

    `LayerwiseCiConfig` → one inner CI fn per `module_to_c` entry inside a
    `LayerwiseCiFnWrapper`; `GlobalCiConfig` → a single global CI fn inside a
    `GlobalCiFnWrapper`.

    Args:
        target_model: Frozen target model; used to look up each decomposition target's
            input dimensionality.
        module_to_c: Map from decomposition-target submodule path to component count.
        components: Map from decomposition-target submodule path to its `Components`
            instance (used by `MLPCiFn` and embedding-target dispatch).
        ci_config: Discriminated CI-fn config; runtime type selects the wrapper.
    """
    match ci_config:
        case LayerwiseCiConfig():
            raw_ci_fns = {
                path: _make_layerwise_ci_fn(
                    target_module=target_model.get_submodule(path),
                    C=C,
                    ci_fn_type=ci_config.fn_type,
                    ci_fn_hidden_dims=ci_config.hidden_dims,
                )
                for path, C in module_to_c.items()
            }
            return LayerwiseCiFnWrapper(
                ci_fns=raw_ci_fns,
                components=components,
                ci_fn_type=ci_config.fn_type,
            )
        case GlobalCiConfig():
            raw_global = _make_global_ci_fn(
                target_model=target_model,
                module_to_c=module_to_c,
                components=components,
                ci_config=ci_config,
            )
            return GlobalCiFnWrapper(global_ci_fn=raw_global, components=components)
        case SpikeGatedCiConfig():
            raw_spike = _make_spike_gated_ci_fn(
                target_model=target_model,
                module_to_c=module_to_c,
                components=components,
                ci_config=ci_config,
            )
            return GlobalCiFnWrapper(global_ci_fn=raw_spike, components=components)
