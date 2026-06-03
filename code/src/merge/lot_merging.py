from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LotPaperSolverConfig:
    """Solver config for the canonical LOT variant from Sun et al., NeurIPS 2025.

    Paper formulation: no ridge regularization, Moore-Penrose pseudoinverse for matrix
    multiplication parameters (Eq. 9), per-dimension feature-weighted average for
    normalization scale parameters (Eq. 12). Element-wise additions (biases) handled
    by mean_delta at the dispatcher level.

    Numerical-stability additions (not in the paper but needed for Qwen3-8B where
    intermediate_size=12288 > typical calibration row counts):
      - pinv_rcond: truncated SVD threshold; None reproduces the paper exactly.
      - max_delta_norm_ratio: when LOT solution norm exceeds N times the mean_delta
        norm, treat the solution as numerically unstable and fall back to mean_delta.
        None disables the check (paper default).
    """
    output_scale: float = 1.0
    rms_eps: float = 1.0e-6      # epsilon used when normalizing features for RMSNorm-equivalent inputs
    div_eps: float = 1.0e-10     # epsilon protecting Eq. 12 denominator from zero
    pinv_rcond: float | None = None  # rcond for torch.linalg.pinv; None uses the library default
    fallback_on_nonfinite: bool = True
    max_delta_norm_ratio: float | None = None  # fallback to mean_delta if delta_norm/mean_delta_norm > this


@dataclass(frozen=True)
class TensorMergeResult:
    tensor: torch.Tensor
    strategy: str
    fallback: bool
    reason: str | None
    delta_norm: float | None
    mean_delta_norm: float | None
    correction_norm: float | None


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def mean_task_delta(base: torch.Tensor, source_tensors: list[torch.Tensor]) -> torch.Tensor:
    if not source_tensors:
        raise ValueError("source_tensors must be non-empty")
    base_f = base.float()
    acc = torch.zeros_like(base_f)
    for source in source_tensors:
        if source.shape != base.shape:
            raise ValueError(f"shape mismatch: source {tuple(source.shape)} vs base {tuple(base.shape)}")
        acc.add_(source.float() - base_f)
    return acc.div(float(len(source_tensors)))


def apply_mean_delta(
    base: torch.Tensor,
    source_tensors: list[torch.Tensor],
    *,
    output_scale: float,
) -> tuple[torch.Tensor, float]:
    mean_delta = mean_task_delta(base, source_tensors)
    out = base.float().add(mean_delta, alpha=float(output_scale))
    return out.to(dtype=base.dtype), float(torch.linalg.vector_norm(mean_delta).detach().cpu())


def _validate_features(
    *,
    tensor_name: str,
    base: torch.Tensor,
    features_by_source: list[torch.Tensor],
    source_count: int,
) -> str | None:
    if base.ndim != 2:
        return f"LOT solver requires a 2D weight tensor, got ndim={base.ndim}"
    if len(features_by_source) != source_count:
        return f"feature source count mismatch: got {len(features_by_source)}, expected {source_count}"
    in_dim = int(base.shape[1])
    for idx, features in enumerate(features_by_source):
        if features.ndim != 2:
            return f"features[{idx}] for {tensor_name} must be 2D, got ndim={features.ndim}"
        if int(features.shape[1]) != in_dim:
            return (
                f"features[{idx}] dim mismatch for {tensor_name}: "
                f"{int(features.shape[1])} != tensor in_dim {in_dim}"
            )
        if int(features.shape[0]) <= 0:
            return f"features[{idx}] for {tensor_name} has no rows"
    return None


def lot_paper_linear_delta(
    *,
    tensor_name: str,
    base: torch.Tensor,
    source_tensors: list[torch.Tensor],
    features_by_source: list[torch.Tensor],
    cfg: LotPaperSolverConfig,
) -> TensorMergeResult:
    """Canonical LOT closed-form solution for matrix multiplication parameters (Eq. 9).

    Solves T* = pinv(sum_k X_k^T X_k) @ (sum_k X_k^T X_k T_k), where:
        - X_k is the matrix of inputs to this module observed for specialist k,
        - T_k is the task vector W_k - W_0 in paper notation (d_in, d_out).
    Stored tensors follow the PyTorch convention (d_out, d_in), so transposes wrap
    the inputs and the output appropriately.
    """
    if not source_tensors:
        raise ValueError("source_tensors must be non-empty")
    if any(source.shape != base.shape for source in source_tensors):
        raise ValueError(f"source tensor shape mismatch for {tensor_name}")

    mean_delta = mean_task_delta(base, source_tensors)
    mean_delta_norm = float(torch.linalg.vector_norm(mean_delta).detach().cpu())

    feature_error = _validate_features(
        tensor_name=tensor_name,
        base=base,
        features_by_source=features_by_source,
        source_count=len(source_tensors),
    )
    if feature_error:
        out = base.float().add(mean_delta, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
        return TensorMergeResult(
            tensor=out,
            strategy="mean_delta",
            fallback=True,
            reason=feature_error,
            delta_norm=mean_delta_norm,
            mean_delta_norm=mean_delta_norm,
            correction_norm=0.0,
        )

    base_f = base.float()
    d_out, d_in = int(base.shape[0]), int(base.shape[1])

    # Task vectors in paper notation (d_in, d_out): T_k_paper = (W_k - W_0)^T in PyTorch storage.
    task_deltas_paper = [(source.float() - base_f).T.contiguous() for source in source_tensors]
    x_parts = [features.float() for features in features_by_source]

    try:
        sum_XTX = torch.zeros((d_in, d_in), dtype=torch.float32, device=base.device)
        sum_XTT = torch.zeros((d_in, d_out), dtype=torch.float32, device=base.device)
        for x_i, t_i in zip(x_parts, task_deltas_paper):
            xtx = x_i.T @ x_i
            sum_XTX = sum_XTX + xtx
            sum_XTT = sum_XTT + xtx @ t_i
        if cfg.pinv_rcond is None:
            pinv = torch.linalg.pinv(sum_XTX)
        else:
            pinv = torch.linalg.pinv(sum_XTX, rcond=float(cfg.pinv_rcond))
        T_paper = pinv @ sum_XTT  # (d_in, d_out)
        delta_torch = T_paper.T.contiguous()  # back to PyTorch convention (d_out, d_in)
    except Exception as exc:
        out = base_f.add(mean_delta, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
        return TensorMergeResult(
            tensor=out,
            strategy="mean_delta",
            fallback=True,
            reason=f"LOT paper solve failed: {exc!r}",
            delta_norm=mean_delta_norm,
            mean_delta_norm=mean_delta_norm,
            correction_norm=None,
        )

    delta_norm = float(torch.linalg.vector_norm(delta_torch).detach().cpu())
    if cfg.fallback_on_nonfinite and not torch.isfinite(delta_torch).all().item():
        out = base_f.add(mean_delta, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
        return TensorMergeResult(
            tensor=out,
            strategy="mean_delta",
            fallback=True,
            reason="LOT paper delta contains non-finite values",
            delta_norm=mean_delta_norm,
            mean_delta_norm=mean_delta_norm,
            correction_norm=delta_norm,
        )

    if (
        cfg.max_delta_norm_ratio is not None
        and mean_delta_norm > 1.0e-8
        and delta_norm > float(cfg.max_delta_norm_ratio) * mean_delta_norm
    ):
        out = base_f.add(mean_delta, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
        return TensorMergeResult(
            tensor=out,
            strategy="mean_delta",
            fallback=True,
            reason=(
                f"LOT paper delta_norm/mean_delta_norm = "
                f"{delta_norm / mean_delta_norm:.3f} > max_delta_norm_ratio "
                f"{float(cfg.max_delta_norm_ratio):.3f}"
            ),
            delta_norm=mean_delta_norm,
            mean_delta_norm=mean_delta_norm,
            correction_norm=delta_norm,
        )

    out = base_f.add(delta_torch, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
    return TensorMergeResult(
        tensor=out,
        strategy="lot_paper",
        fallback=False,
        reason=None,
        delta_norm=delta_norm,
        mean_delta_norm=mean_delta_norm,
        correction_norm=delta_norm,
    )


def lot_paper_rmsnorm_delta(
    *,
    tensor_name: str,
    base: torch.Tensor,
    source_tensors: list[torch.Tensor],
    features_by_source: list[torch.Tensor],
    cfg: LotPaperSolverConfig,
) -> TensorMergeResult:
    """Canonical LOT closed-form solution for normalization scale parameters (Eq. 12).

    Computes T*[d] = sum_k ||X_k[:, d]||^2 T_k[d] / sum_k ||X_k[:, d]||^2 per dimension,
    where X_k are the normalized features feeding the scale multiplication for
    specialist k (i.e., RMSNorm output divided by the scale, equivalent to applying
    RMS normalization to the raw input).
    """
    if not source_tensors:
        raise ValueError("source_tensors must be non-empty")
    if any(source.shape != base.shape for source in source_tensors):
        raise ValueError(f"source tensor shape mismatch for {tensor_name}")
    if base.ndim != 1:
        feature_error = f"LOT paper RMSNorm solver requires a 1D weight, got ndim={base.ndim}"
        out = base.float().add(mean_task_delta(base, source_tensors), alpha=float(cfg.output_scale)).to(dtype=base.dtype)
        return TensorMergeResult(
            tensor=out,
            strategy="mean_delta",
            fallback=True,
            reason=feature_error,
            delta_norm=None,
            mean_delta_norm=None,
            correction_norm=None,
        )

    mean_delta = mean_task_delta(base, source_tensors)
    mean_delta_norm = float(torch.linalg.vector_norm(mean_delta).detach().cpu())

    feature_error = _validate_features(
        tensor_name=tensor_name,
        base=base.unsqueeze(0),
        features_by_source=features_by_source,
        source_count=len(source_tensors),
    )
    if feature_error:
        out = base.float().add(mean_delta, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
        return TensorMergeResult(
            tensor=out,
            strategy="mean_delta",
            fallback=True,
            reason=feature_error,
            delta_norm=mean_delta_norm,
            mean_delta_norm=mean_delta_norm,
            correction_norm=0.0,
        )

    try:
        base_f = base.float()
        task_deltas = [source.float() - base_f for source in source_tensors]
        numerator = torch.zeros_like(base_f)
        denominator = torch.zeros_like(base_f)
        for X_k, T_k in zip(features_by_source, task_deltas):
            sq = (X_k.float() ** 2).sum(dim=0)
            numerator = numerator + sq * T_k
            denominator = denominator + sq
        eps_t = torch.full_like(denominator, float(cfg.div_eps))
        denominator = torch.where(denominator == 0, eps_t, denominator)
        delta = numerator / denominator
    except Exception as exc:
        out = base.float().add(mean_delta, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
        return TensorMergeResult(
            tensor=out,
            strategy="mean_delta",
            fallback=True,
            reason=f"LOT paper RMSNorm solve failed: {exc!r}",
            delta_norm=mean_delta_norm,
            mean_delta_norm=mean_delta_norm,
            correction_norm=None,
        )

    delta_norm = float(torch.linalg.vector_norm(delta).detach().cpu())
    if cfg.fallback_on_nonfinite and not torch.isfinite(delta).all().item():
        out = base.float().add(mean_delta, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
        return TensorMergeResult(
            tensor=out,
            strategy="mean_delta",
            fallback=True,
            reason="LOT paper RMSNorm delta contains non-finite values",
            delta_norm=mean_delta_norm,
            mean_delta_norm=mean_delta_norm,
            correction_norm=delta_norm,
        )

    if (
        cfg.max_delta_norm_ratio is not None
        and mean_delta_norm > 1.0e-8
        and delta_norm > float(cfg.max_delta_norm_ratio) * mean_delta_norm
    ):
        out = base.float().add(mean_delta, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
        return TensorMergeResult(
            tensor=out,
            strategy="mean_delta",
            fallback=True,
            reason=(
                f"LOT paper RMSNorm delta_norm/mean_delta_norm = "
                f"{delta_norm / mean_delta_norm:.3f} > max_delta_norm_ratio "
                f"{float(cfg.max_delta_norm_ratio):.3f}"
            ),
            delta_norm=mean_delta_norm,
            mean_delta_norm=mean_delta_norm,
            correction_norm=delta_norm,
        )

    out = base.float().add(delta, alpha=float(cfg.output_scale)).to(dtype=base.dtype)
    return TensorMergeResult(
        tensor=out,
        strategy="lot_paper_norm",
        fallback=False,
        reason=None,
        delta_norm=delta_norm,
        mean_delta_norm=mean_delta_norm,
        correction_norm=delta_norm,
    )


def merge_tensor(
    *,
    tensor_name: str,
    base: torch.Tensor,
    source_tensors: list[torch.Tensor],
    features_by_source: list[torch.Tensor] | None,
    strategy: str,
    cfg: LotPaperSolverConfig,
) -> TensorMergeResult:
    if not base.is_floating_point():
        return TensorMergeResult(
            tensor=base.clone(),
            strategy="copy_base",
            fallback=False,
            reason="non-floating tensor",
            delta_norm=None,
            mean_delta_norm=None,
            correction_norm=None,
        )
    if strategy == "copy_base":
        return TensorMergeResult(
            tensor=base.clone(),
            strategy="copy_base",
            fallback=False,
            reason=None,
            delta_norm=None,
            mean_delta_norm=None,
            correction_norm=None,
        )
    if strategy == "mean_delta":
        tensor, mean_norm = apply_mean_delta(base, source_tensors, output_scale=float(cfg.output_scale))
        return TensorMergeResult(
            tensor=tensor,
            strategy="mean_delta",
            fallback=False,
            reason=None,
            delta_norm=mean_norm,
            mean_delta_norm=mean_norm,
            correction_norm=0.0,
        )
    if strategy == "lot_paper":
        if not isinstance(cfg, LotPaperSolverConfig):
            raise ValueError(f"strategy 'lot_paper' requires LotPaperSolverConfig, got {type(cfg).__name__}")
        return lot_paper_linear_delta(
            tensor_name=tensor_name,
            base=base,
            source_tensors=source_tensors,
            features_by_source=features_by_source or [],
            cfg=cfg,
        )
    if strategy == "lot_paper_norm":
        if not isinstance(cfg, LotPaperSolverConfig):
            raise ValueError(f"strategy 'lot_paper_norm' requires LotPaperSolverConfig, got {type(cfg).__name__}")
        return lot_paper_rmsnorm_delta(
            tensor_name=tensor_name,
            base=base,
            source_tensors=source_tensors,
            features_by_source=features_by_source or [],
            cfg=cfg,
        )
    raise ValueError(f"unknown tensor merge strategy for {tensor_name}: {strategy}")


def self_test() -> None:
    torch.manual_seed(0)
    base = torch.zeros(4, 3, dtype=torch.float32)
    delta = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 100.0
    source_a = base + delta
    source_b = base + delta
    features = [torch.randn(8, 3), torch.randn(8, 3)]

    # --- Canonical paper-faithful LOT (Eq. 9) ---
    paper_cfg = LotPaperSolverConfig()
    same_paper = lot_paper_linear_delta(
        tensor_name="toy.weight",
        base=base,
        source_tensors=[source_a, source_b],
        features_by_source=features,
        cfg=paper_cfg,
    )
    expected = base + delta
    if not torch.allclose(same_paper.tensor.float(), expected.float(), atol=1.0e-4):
        raise AssertionError("identical-source LOT-paper should equal the shared task delta")

    bad_paper = lot_paper_linear_delta(
        tensor_name="toy.weight",
        base=base,
        source_tensors=[source_a, source_b],
        features_by_source=[torch.randn(8, 2), torch.randn(8, 2)],
        cfg=paper_cfg,
    )
    if not bad_paper.fallback or bad_paper.strategy != "mean_delta":
        raise AssertionError("feature-dimension mismatch should fallback to mean_delta")

    # Identical features + opposing deltas: sum_XTX*(T_A + T_B) = 0, so T* = 0.
    shared_feat = torch.randn(8, 3)
    opposite_paper = lot_paper_linear_delta(
        tensor_name="toy.weight",
        base=base,
        source_tensors=[source_a, base - delta],
        features_by_source=[shared_feat, shared_feat],
        cfg=paper_cfg,
    )
    if float(torch.linalg.vector_norm(opposite_paper.tensor.float())) > 1.0e-4:
        raise AssertionError("opposing-source LOT-paper with shared features should collapse to base (T=0)")

    # --- Canonical paper-faithful LOT for RMSNorm (Eq. 12) ---
    rms_base = torch.ones(4, dtype=torch.float32)
    rms_a = torch.tensor([1.2, 0.8, 1.0, 1.0], dtype=torch.float32)
    rms_b = torch.tensor([1.0, 1.0, 1.4, 0.6], dtype=torch.float32)
    # weighted by squared norms: equal-weight features give simple mean of (rms_a, rms_b)
    rms_features = [torch.ones(16, 4), torch.ones(16, 4)]
    rms_result = lot_paper_rmsnorm_delta(
        tensor_name="toy.weight",
        base=rms_base,
        source_tensors=[rms_a, rms_b],
        features_by_source=rms_features,
        cfg=paper_cfg,
    )
    expected_mean = 0.5 * (rms_a + rms_b)
    if not torch.allclose(rms_result.tensor.float(), expected_mean.float(), atol=1.0e-5):
        raise AssertionError("equal-feature LOT-paper RMSNorm should equal per-dim mean of sources")
