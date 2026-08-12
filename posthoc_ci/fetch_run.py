"""P0: pull the frozen VPD run from W&B and prove it loads + matches paper stats.

The paper's 400k-step 4L Pile decomposition lives at `goodfire/spd/runs/s-55ea3f9b`.
`SavedLMRun.from_path` stages it under `PARAM_DECOMP_OUT_DIR/runs/<id>/` (on the pod:
/workspace/out-torch/runs/). The run predates the repo refactor, so two shims may be
needed and both are handled here:
  - the run may ship `final_config.yaml` instead of `experiment_config.yaml`; in that
    case the repo's own `pile_llama_simple_mlp-4L.yaml` (same architecture, current
    schema) is installed as `experiment_config.yaml`;
  - state-dict keys are compared against a freshly built model before loading, and any
    mismatch is printed in full so a remap can be added deliberately, never silently.

Usage:
    python -m posthoc_ci.fetch_run --run goodfire/spd/runs/s-55ea3f9b
    python -m posthoc_ci.fetch_run --smoke shapes     # load + random-batch CI shapes
    python -m posthoc_ci.fetch_run --smoke data       # + one real eval batch: L0, aliveness
"""

import argparse
import shutil
from pathlib import Path

import torch

from param_decomp.log import logger
from param_decomp_lab.infra.run_files import _wandb_cache_dir, resolve_run_files
from param_decomp_lab.infra.wandb import download_wandb_file, parse_wandb_run_path

DEFAULT_RUN = "goodfire/spd/runs/s-55ea3f9b"
REPO_4L_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml"
)
EXPECTED_MEAN_L0 = 205.0
L0_TOLERANCE_FRAC = 0.25


def _stage_run_files(run_path: str) -> Path:
    """Download config + checkpoint into the local cache, shimming the config name."""
    entity, project, run_id = parse_wandb_run_path(run_path)
    run_dir = _wandb_cache_dir(run_id)

    try:
        files = resolve_run_files(
            run_path, config_filename="experiment_config.yaml", checkpoint_prefix="model"
        )
        logger.info(f"Run resolved natively: {files.config_path}, {files.checkpoint_path}")
        return run_dir
    except Exception as e:  # noqa: BLE001 — wandb raises library-specific errors for missing files
        logger.info(f"Native resolution failed ({type(e).__name__}: {e}); applying config shim")

    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    remote_names = [f.name for f in run.files()]
    logger.info(f"Files on the W&B run: {remote_names}")

    for name in ("final_config.yaml", "config.yaml"):
        if name in remote_names and not (run_dir / name).exists():
            download_wandb_file(run, run_dir, name)
            logger.info(f"Downloaded {name} (kept for the record)")

    config_dst = run_dir / "experiment_config.yaml"
    if not config_dst.exists():
        shutil.copy(REPO_4L_CONFIG, config_dst)
        logger.info(
            f"Installed {REPO_4L_CONFIG.name} as experiment_config.yaml "
            "(current-schema config for the same architecture)"
        )

    ckpt_names = [n for n in remote_names if n.startswith("model") and n.endswith((".pt", ".pth"))]
    assert ckpt_names, f"No model checkpoint among W&B files: {remote_names}"
    latest = sorted(ckpt_names, key=lambda n: int(n.rsplit(".", 1)[0].split("_")[-1]))[-1]
    local_ckpt = run_dir / latest
    if not local_ckpt.exists():
        logger.info(f"Downloading {latest} (~2.9GB)")
        download_wandb_file(run, run_dir, latest)
    if local_ckpt.suffix == ".pt":
        pth = local_ckpt.with_suffix(".pth")
        if not pth.exists():
            local_ckpt.rename(pth)
            logger.info(f"Renamed {latest} -> {pth.name} (loader filters on .pth)")

    return run_dir


def _try_load(run_dir: Path) -> None:
    """Attempt the full model load; on state-dict drift, dump the checkpoint's keys.

    `load_state_dict` (strict) already reports missing/unexpected keys — this adds the
    checkpoint-side shapes so a remap table can be written deliberately, never silently.
    """
    from param_decomp_lab.experiments.lm.run import SavedLMRun

    saved = SavedLMRun.from_path(run_dir)
    try:
        saved.load_model()
    except (RuntimeError, AssertionError) as e:
        print(f"LOAD FAILED: {e}")
        ckpt = torch.load(saved.checkpoint_path, map_location="cpu", weights_only=True)
        print(f"--- checkpoint has {len(ckpt)} keys:")
        for k in sorted(ckpt):
            print(f"    {k}  {tuple(ckpt[k].shape)}")
        raise SystemExit("Key drift found — add an explicit remap before proceeding.") from e
    print("LOAD OK — checkpoint state dict matches the current model exactly.")


def _smoke(run_dir: Path, mode: str) -> None:
    from param_decomp.batch_and_loss_fns import move_batch_to_device
    from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = SavedLMRun.from_path(run_dir)
    model = saved.load_model().to(device)
    model.eval()
    logger.info(f"Model loaded onto {device}")

    if mode == "shapes":
        low_token_ids = 1000  # safely below any tokenizer's vocab size
        batch = torch.randint(0, low_token_ids, (2, saved.cfg.data.max_seq_len), device=device)
    else:
        loader = build_lm_loader(
            saved.cfg.target,
            saved.cfg.data,
            split="eval",
            device=device,
            batch_size=8,
            seed=saved.cfg.pd.seed,
        )
        batch = move_batch_to_device(next(iter(loader)), device)

    with torch.no_grad():
        out = model(batch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=out.cache, sampling="continuous", detach_inputs=False
        ).lower_leaky

    total_l0 = 0.0
    n_pos = None
    print(f"{'module':<24} {'C':>6} {'mean L0@0.01':>14} {'max g':>8}")
    for name, g in sorted(ci.items()):
        if n_pos is None:
            n_pos = g.shape[0] * g.shape[1]
        l0 = (g > 0.01).float().sum(dim=-1).mean().item()
        total_l0 += l0
        print(f"{name:<24} {g.shape[-1]:>6} {l0:>14.1f} {g.max().item():>8.3f}")
    print(f"{'TOTAL':<24} {'':>6} {total_l0:>14.1f}")

    if mode == "data":
        rel = abs(total_l0 - EXPECTED_MEAN_L0) / EXPECTED_MEAN_L0
        verdict = "PASS" if rel <= L0_TOLERANCE_FRAC else "FAIL"
        print(
            f"L0 check: {total_l0:.1f} vs expected ~{EXPECTED_MEAN_L0:.0f} "
            f"(rel err {rel:.2%}) -> {verdict}"
        )
        if verdict == "FAIL":
            raise SystemExit("L0 far from the paper's ~205 — investigate before harvesting.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--smoke", choices=["shapes", "data"], default=None)
    parser.add_argument(
        "--try-load", action="store_true", help="Attempt the full model load; dump keys on drift."
    )
    args = parser.parse_args()

    run_dir = _stage_run_files(args.run)
    print(f"Run staged at: {run_dir}")

    if args.try_load:
        _try_load(run_dir)
    if args.smoke:
        _smoke(run_dir, args.smoke)


if __name__ == "__main__":
    main()
