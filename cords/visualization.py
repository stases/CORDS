"""Notebook figures. Importing CORDS itself never imports plotting libraries."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from .molecules import ELEMENTS

INK = "#253248"
TEAL = "#008b8b"
ORANGE = "#e68136"
COLORS = ["#a8b4c2", "#394d68", "#426cdb", "#e46363", "#65aa62"]
STYLE = {"font.family": "DejaVu Sans", "axes.spines.top": False,
         "axes.spines.right": False, "axes.labelcolor": INK,
         "text.color": INK, "axes.titleweight": "semibold", "figure.dpi": 115}


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _axis3d(ax, positions):
    center = positions.mean(0)
    radius = max(float(np.ptp(positions, axis=0).max()) * .65, 1.0)
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
           zlim=(center[2]-radius, center[2]+radius), xlabel="x / Å", ylabel="y / Å", zlabel="z / Å")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=45)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(alpha=.2)


def _atoms(ax, positions, features, *, bonds=None, alpha=1, marker="o", label_elements=False):
    pos, types = _np(positions), _np(features).argmax(-1)
    if bonds is not None:
        for edge in _np(bonds):
            p = pos[edge]
            ax.plot(*p.T, color="#a8b4c2", lw=2, alpha=.7, zorder=1)
    for i in np.unique(types):
        subset = pos[types == i]
        ax.scatter(*subset.T, color=COLORS[i], s=55 if i == 0 else 105,
                   edgecolors="white", linewidths=.6, marker=marker, alpha=alpha,
                   label=ELEMENTS[i] if label_elements else None, depthshade=True)


def plot_molecule_roundtrip(molecule, fields, decoded):
    """Static source / sampled density / reconstruction / error overlay panels."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 4), layout="constrained")
        axes = [fig.add_subplot(1, 4, i+1, projection="3d") for i in range(4)]
        pos, samples = _np(molecule.positions), _np(fields.coordinates)
        _atoms(axes[0], molecule.positions, molecule.features, bonds=molecule.bonds, label_elements=True)
        axes[0].legend(loc="upper left", fontsize=8, frameon=False)
        axes[0].set_title(f"01  Original · {molecule.name}", loc="left", fontsize=10)
        rho = _np(fields.density[:, 0])
        axes[1].scatter(*samples.T, c=np.log10(rho.clip(1e-12)), cmap="viridis", s=3, alpha=.32)
        axes[1].set_title(f"02  Field · {len(samples):,} samples", loc="left", fontsize=10)
        _atoms(axes[2], decoded.positions, decoded.features, label_elements=False)
        axes[2].set_title(f"03  Recovered · {decoded.count} atoms", loc="left", fontsize=10)
        axes[3].scatter(*pos.T, s=120, facecolors="none", edgecolors=ORANGE, linewidths=1.5,
                        label="original")
        axes[3].scatter(*_np(decoded.positions).T, s=35, c=TEAL, marker="x", label="recovered")
        axes[3].legend(loc="upper left", fontsize=8, frameon=False)
        axes[3].set_title("04  Position overlay", loc="left", fontsize=10)
        for ax in axes:
            _axis3d(ax, pos)
        return fig


def plot_molecular_slice(molecule, transform, resolution=100):
    """Show physical density and five atom-type fields on a common XY slice."""
    pos = molecule.positions
    margin = 3 * transform.sigma
    x = torch.linspace(float(pos[:, 0].min()-margin), float(pos[:, 0].max()+margin), resolution, dtype=pos.dtype)
    y = torch.linspace(float(pos[:, 1].min()-margin), float(pos[:, 1].max()+margin), resolution, dtype=pos.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    plane_z = float(pos[:, 2].mean())
    query = torch.stack((xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), plane_z)), dim=-1)
    rho, features = transform.evaluate(pos, molecule.features, query)
    channels = torch.cat((rho, features), -1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 6, figsize=(15, 2.8), layout="constrained")
        for i, ax in enumerate(axes):
            field = _np(channels[:, i].reshape(resolution, resolution))
            im = ax.imshow(field, origin="lower", extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
                           cmap="magma", vmin=0, vmax=max(float(channels.max()), 1e-12), aspect="equal")
            ax.set_title("Density ρ" if i == 0 else f"Feature h · {ELEMENTS[i-1]}", fontsize=10)
            ax.set_xlabel("x / Å", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("y / Å", fontsize=8)
        fig.colorbar(im, ax=axes, shrink=.65, label="physical field value")
        fig.suptitle(f"One slice through the continuous fields · z = {plane_z:.2f} Å", fontsize=11)
        return fig


def interactive_molecule(molecule, fields, decoded):
    """Return a rotatable Plotly figure; no network or external molecular viewer."""
    import plotly.graph_objects as go
    fig = go.Figure()
    points = _np(fields.coordinates)
    rho = _np(fields.density[:, 0])
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                             marker=dict(size=2, color=np.log10(rho.clip(1e-12)), colorscale="Viridis", opacity=.15),
                             name="importance samples", hoverinfo="skip"))
    for title, pos, features, symbol, size in [
        ("original", molecule.positions, molecule.features, "circle", 7),
        ("recovered", decoded.positions, decoded.features, "cross", 5),
    ]:
        xyz, types = _np(pos), _np(features).argmax(-1)
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                                 marker=dict(size=size, color=[COLORS[i] for i in types], symbol=symbol),
                                 text=[ELEMENTS[i] for i in types], name=title,
                                 hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}) Å<extra>"+title+"</extra>"))
    fig.update_layout(template="plotly_white", height=470, margin=dict(l=0, r=0, t=40, b=0),
                      title=f"{molecule.name} · drag to rotate, click legend to isolate layers",
                      scene=dict(aspectmode="data", xaxis_title="x / Å", yaxis_title="y / Å", zaxis_title="z / Å"),
                      legend=dict(orientation="h", y=0, x=0), uirevision="cords")
    return fig


def _boxes(ax, boxes, labels, color=TEAL):
    for box, label in zip(_np(boxes), _np(labels)):
        x0, y0, x1, y1 = box
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=color, linewidth=1.5))
        ax.text(x0, max(0, y0-2), str(int(label)), color=color, fontsize=9, weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=.3))


def plot_scene_roundtrip(scene, fields, decoded):
    """Image, density, class/size fields and decoded bounding boxes."""
    from .detection import decode_boxes
    height, width = scene.image_shape
    image = _np(scene.image).squeeze()
    if image.ndim == 3:
        image = image.mean(0)
    density = _np(fields.density).reshape(height, width)
    feats = _np(fields.features).reshape(height, width, -1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
        axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        _boxes(axes[0, 0], scene.boxes, scene.labels)
        axes[0, 0].set_title("01  Image + source boxes", loc="left", fontsize=10)
        im = axes[0, 1].imshow(density, cmap="magma", vmin=0)
        axes[0, 1].set_title(f"02  Density mass · sum = {density.sum():.3f}", loc="left", fontsize=10)
        fig.colorbar(im, ax=axes[0, 1], shrink=.7)
        class_colors = plt.get_cmap("tab10")(feats[..., :10].argmax(-1))
        class_colors[..., 3] = np.sqrt(density / max(density.max(), 1e-12))
        axes[0, 2].imshow(class_colors)
        axes[0, 2].set_title("03  Dominant class · opacity ∝ √density", loc="left", fontsize=10)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import BoundaryNorm
        legend = ScalarMappable(norm=BoundaryNorm(np.arange(-.5, 10, 1), 10), cmap="tab10")
        fig.colorbar(legend, ax=axes[0, 2], shrink=.7, ticks=range(10), label="digit class")
        for ax, index, title in [(axes[1, 0], 10, "04  Width mass"), (axes[1, 1], 11, "05  Height mass")]:
            im = ax.imshow(feats[..., index], cmap="viridis", vmin=0)
            ax.set_title(title, loc="left", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=.7)
        axes[1, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
        _boxes(axes[1, 2], decode_boxes(decoded, scene.image_shape), decoded.features[:, :10].argmax(-1), ORANGE)
        axes[1, 2].set_title(f"06  Recovered · {decoded.count} boxes", loc="left", fontsize=10)
        for ax in axes.flat:
            ax.set(xlim=(-.5, width-.5), ylim=(height-.5, -.5))
            ax.set_xticks([]); ax.set_yticks([])
        return fig


def detection_metrics(scene, decoded):
    from scipy.optimize import linear_sum_assignment
    from .detection import decode_boxes
    gt, pred = _np(scene.boxes), _np(decode_boxes(decoded, scene.image_shape))
    out = {"true_count": len(gt), "decoded_count": decoded.count, "count_correct": len(gt) == decoded.count}
    if not len(gt) or not len(pred):
        return {**out, "matched_iou": 1.0 if len(gt) == len(pred) else 0.0, "class_accuracy": float(len(gt) == len(pred))}
    centers = _np(scene.positions)
    cost = np.linalg.norm(centers[:, None] - _np(decoded.positions)[None], axis=-1)
    i, j = linear_sum_assignment(cost)
    a, b = gt[i], pred[j]
    wh = np.maximum(0, np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2]))
    intersection = wh.prod(-1)
    union = (a[:, 2:]-a[:, :2]).prod(-1) + (b[:, 2:]-b[:, :2]).prod(-1) - intersection
    labels = _np(decoded.features[:, :10]).argmax(-1)
    return {**out, "matched_iou": float(np.mean(intersection / union.clip(1e-12))),
            "class_accuracy": float((labels[j] == _np(scene.labels)[i]).sum() / len(gt)),
            "center_rms_pixels": float(np.sqrt(np.mean(cost[i, j]**2)))}


def plot_training_history(history):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
        steps = [row["step"] for row in history]
        ax.plot(steps, [row.get("train_loss", np.nan) for row in history], color=TEAL, alpha=.6, label="training")
        if "eval_loss" in history[0]:
            ax.plot(steps, [row["eval_loss"] for row in history], color=ORANGE, label="fixed evaluation batch")
        ax.set(xlabel="Optimization steps", ylabel="Loss", title="Optional fixed-batch overfit")
        ax.legend(frameon=False)
        return fig


def plot_learning_example(result):
    """Fixed-noise denoising or image-to-field predictions, never called a generator."""
    with plt.rc_context(STYLE):
        if result["config"]["task"] == "qm9":
            fig = plt.figure(figsize=(12, 4), layout="constrained")
            for i, (title, tensors) in enumerate([
                ("Before optimization", result["before"]),
                ("After optimization", result["after"]),
                ("Clean field target", result["targets"]),
            ]):
                ax = fig.add_subplot(1, 3, i+1, projection="3d")
                coordinates, values = tensors
                positions = _np(coordinates[0]) * result["config"]["coordinate_scale"]
                ax.scatter(*positions.T, c=_np(values[0, :, 0]), cmap="viridis", s=8, alpha=.6)
                _axis3d(ax, _np(result["targets"][0][0]) * result["config"]["coordinate_scale"])
                ax.set_title(title, fontsize=11)
            fig.suptitle("Fixed noisy input → denoised field samples (not decoded atoms)", fontsize=12)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), layout="constrained")
            axes[0].imshow(_np(result["inputs"][0]).squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Input image", fontsize=10)
            maximum = float(result["targets"][0, 0].max())
            for ax, title, values in zip(axes[1:], ["Target density", "Before optimization", "After optimization"],
                                         [result["targets"], result["before"], result["after"]]):
                rho = _np(values[0, 0])
                ax.imshow(rho, cmap="magma", vmin=0, vmax=maximum)
                ax.set_title(f"{title}\nmass = {rho.sum():.2f}", fontsize=10)
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle("Image → predicted fields on the fixed training batch", fontsize=12)
        return fig
