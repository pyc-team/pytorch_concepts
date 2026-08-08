"""Locate *why* a generative concept model reconstructs well but generates badly.

``run_generative_analysis.py`` shows the symptom — sharp reconstructions beside
incoherent prior samples. It does not say which of the several things that
produce that picture is actually responsible, and they call for different fixes:

``C1`` the generative path is too shallow, so the decoder is only valid in a
    thin neighbourhood of the codes it was trained on;
``C2`` the *aggregate* posterior has drifted off the prior, so ``z ~ N(0, I)``
    lands in a hole even though the decoder would be fine there;
``C3`` the decoder's ``BatchNorm`` running statistics were accumulated from
    posterior-driven bottlenecks and are miscalibrated for prior-driven ones.

This script separates them. It writes ``diagnostics.png`` — six rows, each
holding everything fixed but one suspect — and ``diagnostics.csv``, then prints
a decision rule mapping the row that jumps in quality onto the cause.

Everything here is read-only: no training, no checkpoint is written.

Reuses ``run_generative_analysis``'s rebuild/figure helpers rather than
restating them, so the two scripts cannot drift apart.
"""

import csv
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from conceptarium.evaluate import load_job
from conceptarium.resolvers import register_custom_resolvers
from conceptarium.utils import seed_everything
from torch_concepts.nn import AncestralSamplingInference

from run_generative_analysis import (
    concept_variables,
    observation_of,
    rebuild,
    resolve_device,
    resolve_job_dir,
    save_grid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Posterior statistics
# ---------------------------------------------------------------------------
def posterior_statistics(
    model, datamodule, vi_query, max_batches: Optional[int],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Encode the test split and collect what ``q(z | x)`` looks like in bulk.

    Returns the per-sample ``loc`` and ``scale`` of the guide, plus the concept
    probabilities the model assigns at those posterior codes. Everything the
    caller needs to answer "is the aggregate posterior the prior?" and "does the
    bottleneck behave the same at prior codes?" comes out of this one pass, which
    is the expensive one — the guide runs the backbone live.
    """
    loader = datamodule.test_dataloader() or datamodule.val_dataloader()
    concepts = concept_variables(model)

    locs: List[torch.Tensor] = []
    scales: List[torch.Tensor] = []
    probs: Dict[str, List[torch.Tensor]] = {v.name: [] for v in concepts}
    recon_loc: List[torch.Tensor] = []

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            x = batch["inputs"]["x"]
            if device is not None:
                x = x.to(device)
            out = model(query=vi_query, input=x)
            # Straight to the CPU: these accumulate over the whole scan, and the
            # statistics below are cheap enough that keeping them on an
            # accelerator only costs memory.
            locs.append(out.guide_params["loc"]["z"].tensor.cpu())
            scales.append(out.guide_params["scale"]["z"].tensor.cpu())
            for variable in concepts:
                probs[variable.name].append(out.probs[variable.name].tensor.cpu())
            recon_loc.append(observation_of(model, out).tensor.cpu())

    return {
        "loc": torch.cat(locs),
        "scale": torch.cat(scales),
        "probs": {name: torch.cat(parts) for name, parts in probs.items()},
        "recon": torch.cat(recon_loc),
    }


def latent_report(loc: torch.Tensor, scale: torch.Tensor, collapsed_kl: float) -> List[Dict]:
    """Per-dimension KL and the aggregate posterior's first two moments.

    Two different quantities, easy to conflate. The **per-sample** KL is what the
    objective penalises and what reveals collapsed dimensions. The **aggregate**
    posterior — the mixture over the dataset, whose variance is
    ``Var(loc) + E[scale**2]`` rather than either term alone — is what generation
    actually samples against, and it is the one the objective never constrains.
    """
    prior = torch.distributions.Normal(torch.zeros_like(loc), torch.ones_like(scale))
    kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(loc, scale), prior
    )
    kl_per_dim = kl.mean(0)

    aggregate_mean = loc.mean(0)
    aggregate_std = (loc.var(0) + scale.pow(2).mean(0)).sqrt()

    rows = [
        {"metric": "kl_per_sample_total", "value": float(kl.sum(-1).mean())},
        {"metric": "kl_per_dim_mean", "value": float(kl_per_dim.mean())},
        {"metric": "kl_per_dim_max", "value": float(kl_per_dim.max())},
        {
            "metric": f"n_dims_collapsed(kl<{collapsed_kl})",
            "value": float((kl_per_dim < collapsed_kl).sum()),
        },
        {"metric": "n_dims", "value": float(loc.shape[-1])},
        # The three numbers that decide whether N(0, I) is the right thing to
        # sample: a prior-matched aggregate posterior has mean 0 and std 1 in
        # every dimension.
        {"metric": "aggregate_mean_abs_max", "value": float(aggregate_mean.abs().max())},
        {"metric": "aggregate_std_min", "value": float(aggregate_std.min())},
        {"metric": "aggregate_std_max", "value": float(aggregate_std.max())},
        {"metric": "guide_scale_mean", "value": float(scale.mean())},
    ]
    return rows


def concept_sharpness(probs: Dict[str, torch.Tensor], suffix: str) -> List[Dict]:
    """Mean max-probability and entropy per concept.

    The concept bottleneck mixes state embeddings by these probabilities, so a
    *soft* score does not merely mean "uncertain" — it feeds the decoder the
    average of several state embeddings, a point that never occurs in training
    once the relaxation temperature has annealed. Comparing this at posterior
    versus prior codes tells you whether generation leaves the bottleneck's
    training distribution, independently of whether ``z`` itself is off-prior.
    """
    rows = []
    for name, p in probs.items():
        if p.shape[-1] == 1:  # binary: one column holding P(c = 1)
            p = torch.cat([1 - p, p], dim=-1)
        p = p.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(-1)
        rows.append({"metric": f"{name}_max_prob_{suffix}", "value": float(p.max(-1).values.mean())})
        rows.append({"metric": f"{name}_entropy_{suffix}", "value": float(entropy.mean())})
        rows.append(
            {"metric": f"{name}_entropy_uniform", "value": float(math.log(p.shape[-1]))}
        )
    return rows


def range_report(images: torch.Tensor, suffix: str) -> List[Dict]:
    """How far outside ``[0, 1]`` a decode lands.

    ``Normal``'s ``loc`` is identity-activated and therefore unbounded, while the
    figures clip. A generation row that is mostly clipped looks far worse than
    its likelihood says, so this separates "the model is wrong" from "the render
    is saturated".
    """
    outside = ((images < 0) | (images > 1)).float().mean()
    return [
        {"metric": f"frac_outside_unit_{suffix}", "value": float(outside)},
        {"metric": f"min_{suffix}", "value": float(images.min())},
        {"metric": f"max_{suffix}", "value": float(images.max())},
    ]


# ---------------------------------------------------------------------------
# Decoding variants
# ---------------------------------------------------------------------------
def decode(model, engine, query, z: torch.Tensor) -> torch.Tensor:
    """Decode explicit latent codes, drawing every other variable ancestrally."""
    with torch.inference_mode():
        return observation_of(model, engine.query(query=query, evidence={"z": z})).tensor


def decode_and_probs(model, engine, query, z: torch.Tensor, concepts):
    """``decode``, also returning the concept probabilities used on the way."""
    with torch.inference_mode():
        out = engine.query(query=query, evidence={"z": z})
    probs = {v.name: out.probs[v.name].tensor.detach() for v in concepts}
    return observation_of(model, out).tensor, probs


def decoder_batchnorms(model) -> List[nn.Module]:
    """The ``BatchNorm`` layers inside the model's decoder, and only those.

    Restricted to the decoder on purpose: the guide's backbone is a ResNet full
    of ``BatchNorm`` whose statistics are perfectly well calibrated on real
    images. Only the decoder's see a different input distribution at generation
    time, so only those are worth toggling.
    """
    from torch_concepts.nn import ConvDecoder

    return [
        module
        for decoder in model.modules()
        if isinstance(decoder, ConvDecoder)
        for module in decoder.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


# ---------------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="generative_diagnostics", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.get("seed", 42))

    job_dir = resolve_job_dir(cfg)
    job_cfg, ckpt_path = load_job(job_dir)
    if ckpt_path is None:
        raise SystemExit(f"No checkpoint under {job_dir / 'checkpoints'}.")
    logger.info("diagnosing %s", job_dir)

    datamodule, model = rebuild(job_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    device = resolve_device(cfg.get("accelerator"))
    model.to(device)
    logger.info("running on %s", device)

    shape = tuple(datamodule.n_features)
    concepts = concept_variables(model)
    query = ["input", "z", *(v.name for v in concepts)]
    vi_query = list(model.pgm.variables)

    # Same decoding engine and settings as run_generative_analysis.py, so the
    # rows below are comparable with the figures that script produces.
    hard = cfg.get("hard_sampling")
    if hard is None:
        hard = not getattr(model, "soft_mixing", False)
    temperature = float(model.train_inference.temperature)
    engine = AncestralSamplingInference(
        model.pgm, p_int=1.0, hard=bool(hard),
        initial_temperature=temperature, annealing="constant",
    )

    out_dir = Path(job_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(cfg.get("n_samples", 10))
    max_batches = cfg.get("max_batches")

    # --- the expensive pass: what does q(z | x) look like over the data? ---
    stats = posterior_statistics(model, datamodule, vi_query, max_batches, device)
    loc, scale = stats["loc"], stats["scale"]
    logger.info("encoded %d samples", loc.shape[0])

    rows: List[Dict] = [{"metric": "n_encoded", "value": float(loc.shape[0])}]
    rows += latent_report(loc, scale, float(cfg.get("collapsed_kl", 0.01)))
    rows += concept_sharpness(stats["probs"], "posterior")
    rows += range_report(stats["recon"], "reconstruction")

    aggregate_mean = loc.mean(0)
    aggregate_std = (loc.var(0) + scale.pow(2).mean(0)).sqrt()

    # --- the six rows ---
    batch = next(iter(datamodule.test_dataloader() or datamodule.val_dataloader()))
    images = batch["inputs"]["x"][:n].to(device)

    with torch.inference_mode():
        encoded = model(query=vi_query, input=images)
    z_mean = encoded.guide_params["loc"]["z"].tensor
    z_scale = encoded.guide_params["scale"]["z"].tensor
    aggregate_mean = aggregate_mean.to(device)
    aggregate_std = aggregate_std.to(device)

    # Row 2 vs 3: the analysis script only ever shows row 2, the posterior MEAN.
    # If row 3 — an honest draw from q(z|x), which is what training actually
    # decoded — is markedly worse, the decoder is hypersensitive to its input and
    # no amount of fixing the prior will help.
    recon_mean = decode(model, engine, query, z_mean)
    recon_draw = decode(model, engine, query, z_mean + z_scale * torch.randn_like(z_scale))

    # Row 4 vs 5: the same decoder at prior codes and at codes drawn from the
    # aggregate posterior. A large gap means the decoder is fine and the prior is
    # simply in the wrong place; no gap means the decoder is the problem.
    z_prior = torch.randn(n, z_mean.shape[-1], device=device)
    gen_prior, prior_probs = decode_and_probs(model, engine, query, z_prior, concepts)
    z_aggregate = aggregate_mean + aggregate_std * torch.randn(
        n, z_mean.shape[-1], device=device
    )
    gen_aggregate = decode(model, engine, query, z_aggregate)

    rows += concept_sharpness(prior_probs, "prior")
    rows += range_report(gen_prior, "generation")

    # Row 6: the prior codes again, with the decoder's BatchNorm normalising by
    # THIS batch instead of its stored running statistics. An improvement here is
    # the signature of stale running statistics rather than a bad latent code.
    norms = decoder_batchnorms(model)
    if norms:
        for module in norms:
            module.train()
        gen_batchstats = decode(model, engine, query, z_prior)
        for module in norms:
            module.eval()
    else:
        gen_batchstats = gen_prior

    save_grid(
        [images, recon_mean, recon_draw, gen_prior, gen_aggregate, gen_batchstats],
        shape,
        out_dir / "diagnostics.png",
        row_labels=[
            "original",
            "recon z=E[q]",
            "recon z~q",
            "gen z~N(0,I)",
            "gen z~agg",
            "gen z~N(0,I) BN=batch",
        ],
    )

    csv_out = out_dir / "diagnostics.csv"
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", csv_out)

    for row in rows:
        print(f"{row['metric']:>36}: {row['value']:.4f}")

    print(
        "\nRead diagnostics.png against this:\n"
        "  row 3 much worse than row 2  -> C1, the decoder is hypersensitive;\n"
        "                                  deepen the generative path.\n"
        "  row 5 much better than row 4 -> C2, the aggregate posterior has\n"
        "                                  drifted; fix the KL side.\n"
        "  row 6 much better than row 4 -> C3, stale BatchNorm statistics;\n"
        "                                  use a batch-independent norm.\n"
        "  *_entropy_prior >> *_entropy_posterior -> the concept mixture itself\n"
        "                                  leaves its training distribution.\n"
        "  aggregate_std far from 1, or many collapsed dims -> C2 as well.\n"
    )


if __name__ == "__main__":
    register_custom_resolvers()
    main()
