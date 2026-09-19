"""Plotting and instrumentation helpers for the Qwen-Image step-by-step notebook.

Figure labels are English on purpose: the container's matplotlib has no CJK font,
so Chinese titles would render as boxes. The narrative lives in the notebook's
markdown cells.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

RESULTS = Path(__file__).resolve().parent / "results"

INK = "#1f2328"
MUTED = "#8b949e"
ACCENT = "#d94f2b"
ACCENT2 = "#2b6cb0"
FAINT = "#e6e8eb"

NOISE_CMAP = LinearSegmentedColormap.from_list("noise", ["#10131a", "#4a5568", "#cbd5e0", "#ffffff"])
HEAT_CMAP = LinearSegmentedColormap.from_list("heat", ["#10131a", "#2b6cb0", "#d94f2b", "#ffd166"])


def use_notebook_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
    })


def save(fig, name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / f"{name}.png", bbox_inches="tight", facecolor="white")


# ----------------------------------------------------------------------------
# latent -> pixels
# ----------------------------------------------------------------------------

def decode_packed(pipe, packed: torch.Tensor, height: int, width: int):
    """Packed latent [B, tokens, 64] -> PIL image.

    Mirrors `QwenImagePipeline.__call__` lines 703-714 exactly, including one
    detail that matters: the pipeline writes

        latents_std = 1.0 / torch.tensor(...).view(...).to(device, dtype)

    so the reciprocal is taken AFTER the cast to bfloat16, not before. Doing it
    in float32 and then casting changes the constant by ~4e-3 and moves decoded
    pixels by up to 14/255 — see section 13 of the notebook.
    """
    latents = pipe._unpack_latents(packed, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)

    z_dim = pipe.vae.config.z_dim
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(
        latents.device, latents.dtype)
    inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(
        latents.device, latents.dtype)

    latents = latents / inv_std + mean

    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]

    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def x0_from_velocity(x_t: torch.Tensor, sigma: float, velocity: torch.Tensor) -> torch.Tensor:
    """The model's current guess at the finished latent.

    diffusers states this relation itself in `FlowMatchEulerDiscreteScheduler.step`,
    stochastic branch: `x0 = sample - current_sigma * model_output`.
    """
    return x_t.float() - sigma * velocity.float()


# ----------------------------------------------------------------------------
# attention capture
# ----------------------------------------------------------------------------

class AttentionStore:
    """Collects image-token -> text-token attention for selected layers."""

    def __init__(self) -> None:
        self.armed = False
        self.tag = None
        self.maps: dict[tuple, torch.Tensor] = {}

    def arm(self, tag) -> None:
        self.armed, self.tag = True, tag

    def disarm(self) -> None:
        self.armed, self.tag = False, None


def _make_capturing_processor(base_cls, store: AttentionStore, layer_idx: int, chunk: int = 512):
    import torch.nn.functional as F
    from diffusers.models.transformers.transformer_qwenimage import ROPE_PER_DEVICE, dispatch_attention_fn

    class CapturingProcessor(base_cls):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     encoder_hidden_states_mask=None, attention_mask=None,
                     image_rotary_emb=None):
            # --- verbatim from QwenDoubleStreamAttnProcessor2_0 ------------
            if encoder_hidden_states_mask is not None:
                seq_img = hidden_states.shape[1]
                image_mask = torch.ones((hidden_states.shape[0], seq_img),
                                        dtype=torch.bool, device=hidden_states.device)
                attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
                attention_mask = attention_mask[:, None, None, :]

            seq_txt = encoder_hidden_states.shape[1]

            img_query = attn.to_q(hidden_states)
            img_key = attn.to_k(hidden_states)
            img_value = attn.to_v(hidden_states)

            txt_query = attn.add_q_proj(encoder_hidden_states)
            txt_key = attn.add_k_proj(encoder_hidden_states)
            txt_value = attn.add_v_proj(encoder_hidden_states)

            head_dim = attn.inner_dim // attn.heads
            img_query = img_query.unflatten(-1, (-1, head_dim))
            img_key = img_key.unflatten(-1, (-1, head_dim))
            img_value = img_value.unflatten(-1, (-1, head_dim))
            txt_query = txt_query.unflatten(-1, (-1, head_dim))
            txt_key = txt_key.unflatten(-1, (-1, head_dim))
            txt_value = txt_value.unflatten(-1, (-1, head_dim))

            if attn.norm_q is not None:
                img_query = attn.norm_q(img_query)
            if attn.norm_k is not None:
                img_key = attn.norm_k(img_key)
            if attn.norm_added_q is not None:
                txt_query = attn.norm_added_q(txt_query)
            if attn.norm_added_k is not None:
                txt_key = attn.norm_added_k(txt_key)

            if image_rotary_emb is not None:
                img_freqs, txt_freqs = image_rotary_emb
                apply_rope = ROPE_PER_DEVICE.get(img_query.device.type, ROPE_PER_DEVICE["cuda"])
                img_query = apply_rope(img_query, img_freqs)
                img_key = apply_rope(img_key, img_freqs)
                txt_query = apply_rope(txt_query, txt_freqs)
                txt_key = apply_rope(txt_key, txt_freqs)

            joint_query = torch.cat([txt_query, img_query], dim=1)
            joint_key = torch.cat([txt_key, img_key], dim=1)
            joint_value = torch.cat([txt_value, img_value], dim=1)

            joint_hidden_states = dispatch_attention_fn(
                joint_query, joint_key, joint_value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend, parallel_config=self._parallel_config,
            )
            # --- end verbatim ---------------------------------------------

            if store.armed:
                store.maps[(store.tag, layer_idx)] = _img_to_txt_attention(
                    img_query, joint_key, attention_mask, seq_txt, head_dim, chunk
                )

            joint_hidden_states = joint_hidden_states.flatten(2, 3)
            joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

            txt_attn_output = joint_hidden_states[:, :seq_txt, :]
            img_attn_output = joint_hidden_states[:, seq_txt:, :]

            img_attn_output = attn.to_out[0](img_attn_output.contiguous())
            if len(attn.to_out) > 1:
                img_attn_output = attn.to_out[1](img_attn_output)
            txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

            return img_attn_output, txt_attn_output

    return CapturingProcessor()


def _img_to_txt_attention(img_query, joint_key, attention_mask, seq_txt, head_dim, chunk):
    """Softmax over the *full* joint key set, then keep only the text columns.

    The fused kernel never materialises these weights, so they are recomputed here.
    Done in float32 and chunked over queries: the full [heads, 4096, 4123] logit
    block would otherwise be the largest tensor in the run.
    """
    import torch.nn.functional as F

    q = img_query.permute(0, 2, 1, 3).float()   # [B, H, Sq, D]
    k = joint_key.permute(0, 2, 1, 3).float()   # [B, H, Sk, D]
    scale = 1.0 / math.sqrt(head_dim)

    out = []
    for start in range(0, q.shape[2], chunk):
        logits = torch.matmul(q[:, :, start:start + chunk], k.transpose(-1, -2)) * scale

        if attention_mask is not None:
            logits = logits.masked_fill(~attention_mask.bool(), float("-inf"))

        probs = F.softmax(logits, dim=-1)
        out.append(probs[..., :seq_txt].cpu())

    return torch.cat(out, dim=2)[0]   # [H, Sq, seq_txt]


def install_attention_capture(transformer, layers, store: AttentionStore):
    """Swap in capturing processors on `layers`. Returns a restore callable."""
    originals = {}

    for idx in layers:
        attn = transformer.transformer_blocks[idx].attn
        original = attn.processor
        originals[idx] = original

        proc = _make_capturing_processor(type(original), store, idx)
        proc._attention_backend = original._attention_backend
        proc._parallel_config = original._parallel_config
        attn.processor = proc

    def restore():
        for idx, original in originals.items():
            transformer.transformer_blocks[idx].attn.processor = original

    return restore


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def plot_token_strip(tokens, drop, title, per_row=12):
    """Template tokens laid out in reading order; the dropped prefix greyed out."""
    rows = math.ceil(len(tokens) / per_row)
    fig, ax = plt.subplots(figsize=(min(per_row, len(tokens)) * 1.05, rows * 0.62))

    for i, tok in enumerate(tokens):
        r, c = divmod(i, per_row)
        kept = i >= drop

        ax.add_patch(plt.Rectangle(
            (c, -r), 0.94, 0.8,
            facecolor="#fdf0ec" if kept else FAINT,
            edgecolor=ACCENT if kept else MUTED,
            linewidth=0.9,
        ))
        label = tok.replace("Ġ", "␣").replace("Ċ", "\\n")
        if len(label) > 9:
            label = label[:8] + "…"

        ax.text(c + 0.47, -r + 0.47, label, ha="center", va="center",
                fontsize=7, color=INK if kept else MUTED)
        ax.text(c + 0.47, -r + 0.16, str(i), ha="center", va="center",
                fontsize=5.5, color=MUTED)

    ax.set_xlim(-0.1, per_row + 0.1)
    ax.set_ylim(-rows + 0.1, 1.0)
    ax.axis("off")
    ax.set_title(f"{title}   —   grey = dropped prefix ({drop}), orange = kept ({len(tokens) - drop})",
                 color=INK, loc="left", pad=8)
    fig.tight_layout()
    return fig


def plot_embedding_stats(emb, tokens):
    """Per-token norm and token-token cosine similarity.

    Deliberately modest: a raw [tokens, 3584] heatmap looks informative and is not.
    Neither panel says anything about meaning, only about geometry.
    """
    e = emb[0].float().cpu()
    norms = e.norm(dim=-1).numpy()
    normed = e / e.norm(dim=-1, keepdim=True)
    cos = (normed @ normed.T).numpy()

    labels = [t.replace("Ġ", "␣").replace("Ċ", "\\n")[:10] for t in tokens]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    axes[0].bar(range(len(norms)), norms, color=ACCENT2, width=0.75)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=70, ha="right", fontsize=6.5)
    axes[0].set_ylabel("L2 norm")
    axes[0].set_title("Per-token norm of the last hidden state", loc="left")

    im = axes[1].imshow(cos, cmap=HEAT_CMAP, vmin=cos.min(), vmax=1.0)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=6)
    axes[1].set_title("Cosine similarity between token vectors", loc="left")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    return fig


def plot_latent_channels(latent, title, cmap=NOISE_CMAP):
    """All 16 VAE channels of an unpacked latent [B, C, T, H, W]."""
    grid = latent[0, :, 0].float().cpu().numpy()
    n = grid.shape[0]
    cols = 8
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.55))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i >= n:
            continue
        ch = grid[i]
        axes[i].imshow(ch, cmap=cmap)
        axes[i].set_title(f"ch{i}  σ={ch.std():.2f}", fontsize=7, color=MUTED, pad=3)

    fig.suptitle(title, color=INK, fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_packing_map():
    """Where each of the 64 slots in a packed token comes from.

    Reproduces `_pack_latents` on a labelled tensor, so the layout shown is the
    layout diffusers actually produces rather than a claim about it.
    """
    c_of = np.zeros(64, dtype=int)
    dy_of = np.zeros(64, dtype=int)
    dx_of = np.zeros(64, dtype=int)

    for c in range(16):
        for dy in range(2):
            for dx in range(2):
                c_of[c * 4 + dy * 2 + dx] = c
                dy_of[c * 4 + dy * 2 + dx] = dy
                dx_of[c * 4 + dy * 2 + dx] = dx

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.1),
                             gridspec_kw={"width_ratios": [1, 2.4]})

    ax = axes[0]
    for dy in range(2):
        for dx in range(2):
            ax.add_patch(plt.Rectangle((dx, -dy), 0.92, 0.92,
                                       facecolor="#fdf0ec", edgecolor=ACCENT))
            ax.text(dx + 0.46, -dy + 0.46, f"(dy={dy}\ndx={dx})",
                    ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlim(-0.15, 2.1)
    ax.set_ylim(-1.15, 1.1)
    ax.axis("off")
    ax.set_title("one 2x2 block of latent cells\n(x 16 channels)", loc="left", fontsize=9)

    ax = axes[1]
    ax.imshow(c_of.reshape(1, 64), cmap=HEAT_CMAP, aspect="auto",
              extent=(0, 64, 0, 1))
    for s in range(0, 64, 4):
        ax.axvline(s, color="white", linewidth=1.1)
    for s in range(0, 64, 8):
        ax.text(s + 2, 0.5, f"c{c_of[s]}", ha="center", va="center",
                fontsize=7, color="white")
    ax.set_yticks([])
    ax.set_xticks(range(0, 65, 8))
    ax.set_xlabel("slot inside the 64-wide token")
    ax.set_title("slot = channel x 4 + dy x 2 + dx   (channel-major)", loc="left", fontsize=9)

    fig.tight_layout()
    return fig


def plot_sigma_schedules(schedules):
    """sigma vs step for one or more step counts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

    for (label, sigmas), color in zip(schedules.items(), [ACCENT, ACCENT2, MUTED]):
        s = np.asarray(sigmas, dtype=float)
        x = np.linspace(0, 1, len(s))

        axes[0].plot(x, s, "o-", color=color, label=label, markersize=4.5, linewidth=1.6)
        axes[1].plot(x[:-1], np.diff(s), "o-", color=color, label=label,
                     markersize=4.5, linewidth=1.6)

    axes[0].set_xlabel("progress through the loop")
    axes[0].set_ylabel("sigma  (1 = pure noise)")
    axes[0].set_title("Noise schedule", loc="left")
    axes[0].legend()

    axes[1].set_xlabel("progress through the loop")
    axes[1].set_ylabel("dt = sigma_next - sigma")
    axes[1].set_title("Step size (always negative)", loc="left")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_image_row(images, titles, suptitle=None, height=3.2):
    fig, axes = plt.subplots(1, len(images), figsize=(height * len(images), height + 0.5))
    axes = np.atleast_1d(axes)

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, color=INK, pad=5)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, color=INK, fontsize=11, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()

    return fig


def plot_velocity_maps(velocities, sigmas, grid):
    """Per-token L2 norm of the predicted velocity, folded back to the token grid."""
    mags = [v[0].float().norm(dim=-1).cpu().numpy().reshape(grid) for v in velocities]
    vmax = max(m.max() for m in mags)

    fig, axes = plt.subplots(1, len(mags), figsize=(3.0 * len(mags), 3.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for i, (ax, mag, sigma) in enumerate(zip(axes, mags, sigmas)):
        im = ax.imshow(mag, cmap=HEAT_CMAP, vmin=0, vmax=vmax)
        ax.set_title(f"step {i}   sigma={sigma:.3f}", fontsize=9, color=INK)
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02)
    fig.suptitle("Predicted velocity magnitude per image token  —  where the model is pushing",
                 color=INK, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return fig


def attention_map(store, tag, layer, token_index, grid):
    """Mean over heads of one text token's attention, folded back to the token grid."""
    probs = store.maps[(tag, layer)]          # [heads, image tokens, text tokens]
    return probs[:, :, token_index].mean(0).numpy().reshape(grid)


def _show_attention(ax, amap, clip):
    """Robust per-panel scaling.

    A handful of image tokens act as attention sinks and take values orders of
    magnitude above the rest; on a shared linear scale they flatten everything
    else to black. Clipping at a high percentile is what makes the spatial
    structure visible — the panels are therefore NOT comparable in absolute value.
    """
    vmax = float(np.percentile(amap, clip))
    ax.imshow(amap, cmap=HEAT_CMAP, vmin=float(amap.min()), vmax=vmax,
              interpolation="nearest")
    ax.axis("off")


def plot_attention_steps(store, tags, layer, token_index, token_label, grid,
                         base_image, clip=99.0):
    """One text token's attention map at each denoising step."""
    fig, axes = plt.subplots(1, len(tags) + 1, figsize=(3.1 * (len(tags) + 1), 3.6))
    axes = np.atleast_1d(axes)

    axes[0].imshow(base_image)
    axes[0].set_title("final image (reference)", fontsize=9, color=INK)
    axes[0].axis("off")

    for ax, tag in zip(axes[1:], tags):
        _show_attention(ax, attention_map(store, tag, layer, token_index, grid), clip)
        ax.set_title(tag, fontsize=9, color=INK)

    fig.suptitle(f'Attention to "{token_label}"  —  layer {layer}, mean over 24 heads, '
                 f'each panel scaled to its own p{clip:g}',
                 color=INK, fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_attention_depth(store, tag, layers, token_index, token_label, grid,
                         base_image, clip=99.0):
    """The same token at the same step, seen at three depths."""
    fig, axes = plt.subplots(1, len(layers) + 1, figsize=(3.1 * (len(layers) + 1), 3.6))
    axes = np.atleast_1d(axes)

    axes[0].imshow(base_image)
    axes[0].set_title("final image (reference)", fontsize=9, color=INK)
    axes[0].axis("off")

    for ax, layer in zip(axes[1:], layers):
        _show_attention(ax, attention_map(store, tag, layer, token_index, grid), clip)
        ax.set_title(f"layer {layer}", fontsize=9, color=INK)

    fig.suptitle(f'Attention to "{token_label}" at {tag}, across depth',
                 color=INK, fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_attention_mass(store, tag, layer, tokens):
    """How the image tokens' attention on text is split across words."""
    probs = store.maps[(tag, layer)]
    share = probs.mean(0).sum(0)
    share = (share / share.sum()).numpy()

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.bar(range(len(share)), share, color=ACCENT, width=0.75)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([t.replace("\u0120", "\u2423") for t in tokens],
                       rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("share of the text attention")
    ax.set_title(f"Which words the image tokens look at  ({tag}, layer {layer})", loc="left")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# additions for qwen_image_guide.ipynb (zero-background edition)
# ----------------------------------------------------------------------------

def plot_image_grid(images, titles, ncols=5, suptitle=None, height=2.4):
    """Same idea as plot_image_row, but wraps onto several rows.

    Needed once the denoising loop runs 10 steps: 10+ panels do not fit on one row.
    """
    n = len(images)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(height * ncols, (height + 0.45) * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=8, color=INK, pad=4)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, color=INK, fontsize=11, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.05 / nrows))
    else:
        fig.tight_layout()

    return fig


def plot_vae_roundtrip(original, reconstructed, title):
    """Original / VAE reconstruction / amplified absolute difference, side by side."""
    orig = np.asarray(original, dtype=np.float32) / 255.0
    recon = np.asarray(reconstructed, dtype=np.float32) / 255.0
    diff = np.abs(orig - recon).mean(axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    for ax, img, sub in [(axes[0], orig, "original (pixels)"),
                         (axes[1], recon, "after encode -> decode")]:
        ax.imshow(img)
        ax.set_title(sub, fontsize=9, color=INK, pad=5)
        ax.axis("off")

    im = axes[2].imshow(diff, cmap=HEAT_CMAP, vmin=0, vmax=max(diff.max(), 1e-6))
    axes[2].set_title(f"|difference| averaged over RGB, max {diff.max() * 255:.1f}/255",
                      fontsize=9, color=INK, pad=5)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)

    fig.suptitle(title, color=INK, fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_noise_ladder(clean, noised, sigmas, title):
    """A clean image and the same image at increasing noise levels."""
    images = [clean] + list(noised)
    titles = ["sigma=0.00  (clean)"] + [f"sigma={s:.2f}" for s in sigmas]
    return plot_image_grid(images, titles, ncols=len(images), suptitle=title, height=2.2)


def plot_token_heatmaps(mags, titles, suptitle, grid, cmap=None):
    """Per-image-token scalars folded back to the token grid, on one shared color scale."""
    cmap = cmap or HEAT_CMAP
    maps = [m.reshape(grid) for m in mags]
    vmax = max(float(m.max()) for m in maps)

    fig, axes = plt.subplots(1, len(maps), figsize=(3.0 * len(maps), 3.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, m, title in zip(axes, maps, titles):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9, color=INK)
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02)
    fig.suptitle(suptitle, color=INK, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return fig
