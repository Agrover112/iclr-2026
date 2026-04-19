"""
Export model predictions as CSV (for ParaView) and PNG renders (for Claude analysis).

Usage:
    /home/agrov/gram/bin/python scripts/viz/export_predictions.py \
        --model residual_mlp \
        --checkpoint results/residual_mlp/seed_42/best_model.pt \
        --data-path data/ \
        --sample-index 0 \
        --output-dir figures/predictions

Outputs per output timestep (0-4):
    - CSV:  {sample}_t{i}.csv   (for ParaView)
    - PNGs: {sample}_t{i}_{view}.png  (for Claude / quick inspection)
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, checkpoint_path, features, device):
    """Load model class, instantiate, load checkpoint weights."""
    from src.train import get_model_class

    ModelClass = get_model_class(model_name)

    # Instantiate with features (TypeError fallback like train_one_seed)
    try:
        model = ModelClass(features=features)
    except TypeError:
        model = ModelClass()

    # Load checkpoint (handles GPU -> CPU via map_location)
    state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"ERROR: Checkpoint shape mismatch.\n{e}\n")
            print("The checkpoint was likely trained with different features "
                  "than the model default.")
            print("Use --features to specify the features the checkpoint "
                  "was trained with.")
            sys.exit(1)
        raise
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample(npz_path, features, device):
    """Load a single sample, add batch dim, move to device."""
    from src.data import GRAMDataset

    ds = GRAMDataset([npz_path], features=features)
    sample = ds[0]

    out = {
        "t":            sample["t"].unsqueeze(0).to(device),
        "pos":          sample["pos"].unsqueeze(0).to(device),
        "idcs_airfoil": [sample["idcs_airfoil"].to(device)],
        "velocity_in":  sample["velocity_in"].unsqueeze(0).to(device),
        "velocity_out": sample["velocity_out"].unsqueeze(0).to(device),
    }
    if "point_features" in sample:
        out["point_features"] = sample["point_features"].unsqueeze(0).to(device)
    if "knn_graph" in sample:
        out["knn_graph"] = sample["knn_graph"].unsqueeze(0).to(device)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, sample):
    """Run model forward pass, return (1, 5, N, 3) prediction."""
    with torch.no_grad():
        pred = model(
            sample["t"],
            sample["pos"],
            sample["idcs_airfoil"],
            sample["velocity_in"],
            sample.get("point_features"),
            sample.get("knn_graph"),
        )
    return pred


# ---------------------------------------------------------------------------
# Velocity gradients (approximate)
# ---------------------------------------------------------------------------

def compute_velocity_gradients(velocity, pos, knn_graph=None):
    """Approximate gradient magnitude via k-NN finite differences.

    Args:
        velocity: (N, 3) velocity field
        pos:      (N, 3) point coordinates
        knn_graph: (N, k) neighbor indices, -1 = unused. None = build on-the-fly.

    Returns:
        (N,) scalar gradient magnitude per point.
    """
    if knn_graph is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(pos.numpy())
        _, knn_graph = tree.query(pos.numpy(), k=17)  # k+1 (includes self)
        knn_graph = torch.from_numpy(knn_graph[:, 1:]).long()  # drop self

    N, k = knn_graph.shape

    # Mask invalid neighbors (-1 padding from adaptive knn)
    valid = knn_graph >= 0
    # Replace -1 with 0 for indexing (values will be masked out)
    safe_idx = knn_graph.clamp(min=0)

    nbr_pos = pos[safe_idx]            # (N, k, 3)
    nbr_vel = velocity[safe_idx]       # (N, k, 3)

    dx = nbr_pos - pos.unsqueeze(1)    # (N, k, 3)
    dv = nbr_vel - velocity.unsqueeze(1)  # (N, k, 3)

    dx_norm = dx.norm(dim=2).clamp(min=1e-8)  # (N, k)
    dv_norm = dv.norm(dim=2)                   # (N, k)

    ratio = dv_norm / dx_norm  # (N, k) — |dv|/|dx| per neighbor

    # Masked mean over valid neighbors
    ratio[~valid] = 0.0
    count = valid.float().sum(dim=1).clamp(min=1.0)  # (N,)
    grad_mag = ratio.sum(dim=1) / count  # (N,)

    return grad_mag


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(pos, gt_vel, pred_vel, is_airfoil, grad_gt, grad_pred,
               timestep_idx, output_path):
    """Write one CSV per timestep for ParaView."""
    N = pos.shape[0]
    err = pred_vel - gt_vel
    err_mag = np.linalg.norm(err, axis=1)
    vmag_gt = np.linalg.norm(gt_vel, axis=1)
    vmag_pred = np.linalg.norm(pred_vel, axis=1)

    # Build array: (N, 18)
    data = np.column_stack([
        pos,                    # x, y, z
        gt_vel,                 # vx_gt, vy_gt, vz_gt
        pred_vel,               # vx_pred, vy_pred, vz_pred
        err,                    # vx_err, vy_err, vz_err
        err_mag.reshape(-1, 1), # error_magnitude
        vmag_gt.reshape(-1, 1), # vmag_gt
        vmag_pred.reshape(-1, 1),  # vmag_pred
        grad_gt.reshape(-1, 1),    # grad_mag_gt
        grad_pred.reshape(-1, 1),  # grad_mag_pred
        is_airfoil.reshape(-1, 1), # is_airfoil
    ])

    header = ("x,y,z,"
              "vx_gt,vy_gt,vz_gt,"
              "vx_pred,vy_pred,vz_pred,"
              "vx_err,vy_err,vz_err,"
              "error_magnitude,"
              "vmag_gt,vmag_pred,"
              "grad_mag_gt,grad_mag_pred,"
              "is_airfoil")

    np.savetxt(output_path, data, delimiter=",", header=header,
               comments="", fmt="%.6f")


# ---------------------------------------------------------------------------
# PyVista rendering
# ---------------------------------------------------------------------------

def render_pyvista(pos, scalars, is_airfoil, timestep_idx, sample_name,
                   output_dir):
    """Render multiple views of a point cloud, save as PNGs.

    Args:
        pos:       (N, 3) numpy array
        scalars:   dict of {name: (N,) array} — fields to visualize
        is_airfoil: (N,) bool mask
        timestep_idx: int
        sample_name: str
        output_dir: str
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    cloud = pv.PolyData(pos)
    for name, arr in scalars.items():
        cloud[name] = arr

    # Airfoil centroid for close-up camera
    airfoil_center = pos[is_airfoil].mean(axis=0)

    prefix = os.path.join(output_dir, f"{sample_name}_t{timestep_idx}")

    views = {
        "xy": dict(position=(airfoil_center[0], airfoil_center[1], 5.0),
                    focal_point=airfoil_center,
                    viewup=(0, 1, 0)),
        "xz": dict(position=(airfoil_center[0], 5.0, airfoil_center[2]),
                    focal_point=airfoil_center,
                    viewup=(0, 0, 1)),
    }

    render_fields = ["error_magnitude", "vmag_gt", "vmag_pred",
                     "grad_mag_gt", "grad_mag_pred"]

    for field in render_fields:
        if field not in scalars:
            continue
        for view_name, cam in views.items():
            pl = pv.Plotter(off_screen=True, window_size=(1600, 1000))
            pl.add_mesh(cloud, scalars=field, cmap="turbo",
                        point_size=1.5, render_points_as_spheres=False,
                        show_scalar_bar=True)
            pl.camera.position = cam["position"]
            pl.camera.focal_point = cam["focal_point"]
            pl.camera.up = cam["viewup"]
            out_path = f"{prefix}_{field}_{view_name}.png"
            pl.screenshot(out_path)
            pl.close()

    # Airfoil close-up: error magnitude, tight zoom
    if "error_magnitude" in scalars:
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1000))
        pl.add_mesh(cloud, scalars="error_magnitude", cmap="turbo",
                    point_size=2.5, render_points_as_spheres=False,
                    show_scalar_bar=True)
        pl.camera.position = (airfoil_center[0], airfoil_center[1],
                              airfoil_center[2] + 0.5)
        pl.camera.focal_point = airfoil_center
        pl.camera.up = (0, 1, 0)
        out_path = f"{prefix}_error_closeup.png"
        pl.screenshot(out_path)
        pl.close()


# ---------------------------------------------------------------------------
# Interactive HTML export
# ---------------------------------------------------------------------------

def export_html(pos, scalars, is_airfoil, timestep_idx, sample_name,
                output_dir, subsample=0):
    """Export interactive 3D HTML viewer. Open in browser to rotate/zoom/pan.

    Args:
        pos:        (N, 3) numpy
        scalars:    dict {name: (N,) array}
        is_airfoil: (N,) bool
        timestep_idx: int
        sample_name: str
        output_dir: str
        subsample:  max points to include (0 = auto 50k)
    """
    import pyvista as pv

    N = pos.shape[0]
    max_pts = subsample if subsample > 0 else 50_000

    if N > max_pts:
        idx = np.random.RandomState(42).choice(N, max_pts, replace=False)
        # Always include airfoil points
        airfoil_idx = np.where(is_airfoil)[0]
        idx = np.unique(np.concatenate([idx, airfoil_idx]))
        pos_s = pos[idx]
        scalars_s = {k: v[idx] for k, v in scalars.items()}
    else:
        pos_s = pos
        scalars_s = scalars

    cloud = pv.PolyData(pos_s.astype(np.float32))
    for name, arr in scalars_s.items():
        cloud[name] = arr.astype(np.float32)

    # Clamp error color range to 95th percentile for readability
    if "error_magnitude" in scalars_s:
        err = scalars_s["error_magnitude"]
        clim = [0, float(np.percentile(err, 95))]
    else:
        clim = None

    pl = pv.Plotter()
    default_field = "error_magnitude" if "error_magnitude" in scalars_s else \
        list(scalars_s.keys())[0]
    pl.add_mesh(cloud, scalars=default_field, cmap="turbo",
                point_size=3.0, render_points_as_spheres=False,
                show_scalar_bar=True, clim=clim)

    # Focus camera on airfoil center
    airfoil_center = pos_s[is_airfoil[idx] if N > max_pts else is_airfoil].mean(axis=0) \
        if is_airfoil.any() else pos_s.mean(axis=0)
    pl.camera.focal_point = airfoil_center
    pl.camera.position = (airfoil_center[0], airfoil_center[1],
                          airfoil_center[2] + 2.0)
    pl.camera.up = (0, 1, 0)

    out_path = os.path.join(output_dir, f"{sample_name}_t{timestep_idx}.html")
    pl.export_html(out_path)
    pl.close()
    return out_path


# ---------------------------------------------------------------------------
# Vorticity magnitude (curl of velocity) — CFD-style vortex visualization
# ---------------------------------------------------------------------------

def compute_vorticity_magnitude(velocity, pos, knn_graph=None):
    """Per-point |∇ × v| via k-NN least-squares on the velocity gradient tensor.

    The velocity gradient J_ij = ∂v_i/∂x_j is estimated per point by solving
    a small least-squares problem over its k-NN neighbors. Vorticity is then
    the curl (antisymmetric part of J, extracted as a vector):
        ω_x = J[2,1] - J[1,2]
        ω_y = J[0,2] - J[2,0]
        ω_z = J[1,0] - J[0,1]
        |ω| = sqrt(ω_x² + ω_y² + ω_z²)

    All 100k points are solved in one batched operation.

    Args:
        velocity: (N, 3) torch Tensor
        pos:      (N, 3) torch Tensor
        knn_graph: (N, k) int Tensor of neighbor indices (-1 = padding). If None,
                   builds k=16 via scipy.

    Returns:
        (N,) torch Tensor of vorticity magnitude per point.
    """
    if knn_graph is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(pos.numpy())
        _, knn_graph = tree.query(pos.numpy(), k=17)
        knn_graph = torch.from_numpy(knn_graph[:, 1:]).long()

    N, k = knn_graph.shape
    valid = knn_graph >= 0
    safe_idx = knn_graph.clamp(min=0)

    # Local displacement + velocity difference vectors per neighbor.
    nbr_pos = pos[safe_idx]            # (N, k, 3)
    nbr_vel = velocity[safe_idx]       # (N, k, 3)
    dx = nbr_pos - pos.unsqueeze(1)    # (N, k, 3)
    dv = nbr_vel - velocity.unsqueeze(1)

    # Zero out invalid neighbors so they don't contribute to the LS.
    mask = valid.unsqueeze(-1).to(dx.dtype)
    dx = dx * mask
    dv = dv * mask

    # Normal equations: J^T = (dxᵀ dx)⁻¹ dxᵀ dv   (all batched)
    dxT_dx = torch.einsum('nki,nkj->nij', dx, dx)   # (N, 3, 3)
    dxT_dv = torch.einsum('nki,nkj->nij', dx, dv)   # (N, 3, 3)
    # Tikhonov regularization for rank-deficient neighborhoods (e.g. surface points).
    eye = torch.eye(3, dtype=dxT_dx.dtype).expand(N, 3, 3) * 1e-6
    J_T = torch.linalg.solve(dxT_dx + eye, dxT_dv)  # (N, 3, 3)
    J = J_T.transpose(-1, -2)                       # (N, 3, 3) — velocity Jacobian

    # Curl of v (vorticity vector) from antisymmetric part of J.
    omega_x = J[:, 2, 1] - J[:, 1, 2]
    omega_y = J[:, 0, 2] - J[:, 2, 0]
    omega_z = J[:, 1, 0] - J[:, 0, 1]
    return torch.stack([omega_x, omega_y, omega_z], dim=-1).norm(dim=-1)


def export_turbulent_arrows_html(
    pos, velocity, vorticity_mag, is_airfoil, timestep_idx,
    sample_name, output_dir,
    vort_percentile=80,       # keep top N% vorticity regions
    bl_max_dist=0.0,          # if > 0: keep only points with UDF ≤ this (boundary layer)
    udf=None,                 # (N,) precomputed UDF — required if bl_max_dist > 0
    glyph_scale=None,
    which="gt",               # "gt" or "pred" — label only
):
    """Arrows in turbulent regions only, colored by velocity magnitude.

    Filters points by:
      (a) vorticity ≥ N-th percentile (default 80 → top 20% most turbulent), AND
      (b) optional: UDF ≤ bl_max_dist (keep only boundary-layer points)

    Args:
        pos:           (N, 3) numpy
        velocity:      (N, 3) numpy — either GT or pred velocity
        vorticity_mag: (N,) numpy — |∇×v|
        vort_percentile: keep points above this vorticity percentile
        bl_max_dist: 0 = disabled. If > 0, only keep points with UDF ≤ this
                     value (e.g., 0.05 for boundary-layer-only view).
        udf: (N,) UDF (distance to airfoil). Required when bl_max_dist > 0.
    """
    import pyvista as pv

    # --- Filter to turbulent regions ---
    threshold = float(np.percentile(vorticity_mag, vort_percentile))
    mask = vorticity_mag >= threshold

    # --- Optional: restrict to the boundary layer by distance to surface ---
    if bl_max_dist > 0 and udf is not None:
        mask = mask & (udf <= bl_max_dist)

    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"  WARNING: filter too strict — 0 points match. "
              f"Try lower --turb-percentile or larger --bl-max-dist.")
        return None

    pos_s = pos[idx].astype(np.float32)
    vel_s = velocity[idx].astype(np.float32)
    vmag_s = np.linalg.norm(vel_s, axis=1)

    # --- Auto-scale arrow length ---
    bmin, bmax = pos.min(axis=0), pos.max(axis=0)  # full mesh bounds
    diag = float(np.linalg.norm(bmax - bmin))
    n_rendered = len(pos_s)
    spacing = diag / max(n_rendered ** (1/3), 1.0)
    max_vel = max(float(vmag_s.max()), 1e-6)
    if glyph_scale is None:
        glyph_scale = 1.5 * spacing / max_vel

    cloud = pv.PolyData(pos_s)
    cloud["velocity"] = vel_s
    cloud["vmag"] = vmag_s

    arrow_geom = pv.Arrow(tip_length=0.15, tip_radius=0.005, shaft_radius=0.001)
    arrows = cloud.glyph(
        orient="velocity", scale="vmag",
        factor=glyph_scale, geom=arrow_geom,
    )
    # Colormap scalars carry over through glyph; confirm by setting explicitly.
    arrows["vmag"] = np.repeat(vmag_s, int(len(arrows.points) / max(len(pos_s), 1)) or 1)

    pl = pv.Plotter()
    pl.add_mesh(arrows, scalars="vmag", cmap="turbo",
                show_scalar_bar=True, clim=[0, float(np.percentile(vmag_s, 95))])
    pl.add_axes(interactive=False)
    # Faint context points for orientation.
    pl.add_mesh(pv.PolyData(pos.astype(np.float32)),
                color="lightgray", point_size=0.6, opacity=0.15)

    # Camera — zoomed-out view of full mesh.
    center = 0.5 * (bmin + bmax)
    pl.camera.focal_point = tuple(center)
    pl.camera.position = (center[0], center[1], center[2] + 1.8 * diag)
    pl.camera.up = (0, 1, 0)

    out_path = os.path.join(output_dir,
                            f"{sample_name}_t{timestep_idx}_turb_arrows_{which}.html")
    pl.export_html(out_path)
    pl.close()
    return out_path


def export_quiver_png(
    pos, gt_vec, pred_vec, vorticity_mag, is_airfoil, timestep_idx,
    sample_name, output_dir,
    vort_percentile=80,
    bl_max_dist=0.0,
    udf=None,
    slice_axis="z",
    slice_thickness=None,
):
    """2D matplotlib quiver PNG — 3 panels: GT velocity | pred velocity | error mag.

    Opens in any image viewer. No trame mouse issues. Zoom with normal
    image-viewer shortcuts. Fast to generate, easy to read.
    """
    import matplotlib.pyplot as plt

    threshold = float(np.percentile(vorticity_mag, vort_percentile))
    mask = vorticity_mag >= threshold
    if bl_max_dist > 0 and udf is not None:
        mask = mask & (udf <= bl_max_dist)

    axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
    if slice_thickness is not None and slice_thickness > 0:
        slice_center = pos[is_airfoil, axis_idx].mean()
        mask = mask & (np.abs(pos[:, axis_idx] - slice_center) <= slice_thickness)

    idx = np.where(mask)[0]
    if len(idx) == 0:
        print(f"  WARNING: quiver filter too strict — 0 points match.")
        return None

    plot_axes = [a for a in (0, 1, 2) if a != axis_idx]
    x_lbl = "xyz"[plot_axes[0]]
    y_lbl = "xyz"[plot_axes[1]]

    p2d = pos[idx][:, plot_axes]
    gt2d = gt_vec[idx][:, plot_axes]
    pred2d = pred_vec[idx][:, plot_axes]
    gt_mag = np.linalg.norm(gt_vec[idx], axis=1)
    pred_mag = np.linalg.norm(pred_vec[idx], axis=1)
    err_mag = np.linalg.norm(pred_vec[idx] - gt_vec[idx], axis=1)

    af_mask = is_airfoil.copy()
    if slice_thickness is not None and slice_thickness > 0:
        slice_center = pos[is_airfoil, axis_idx].mean()
        af_mask = af_mask & (np.abs(pos[:, axis_idx] - slice_center) <= slice_thickness)
    af_2d = pos[np.where(af_mask)[0]][:, plot_axes]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    vmax = float(np.percentile(gt_mag, 95))

    for ax, title, vec2d, mag in [
        (axes[0], f"Ground truth (t={timestep_idx})", gt2d, gt_mag),
        (axes[1], f"Prediction (t={timestep_idx})", pred2d, pred_mag),
    ]:
        ax.scatter(af_2d[:, 0], af_2d[:, 1], s=0.5, c="black", alpha=0.5, zorder=1)
        q = ax.quiver(
            p2d[:, 0], p2d[:, 1],
            vec2d[:, 0], vec2d[:, 1],
            mag, cmap="turbo", clim=(0, vmax),
            angles="xy", scale_units="xy", scale=80,
            width=0.0015, headwidth=3, headlength=4,
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(x_lbl); ax.set_ylabel(y_lbl); ax.set_aspect("equal")
        fig.colorbar(q, ax=ax, label="|v|")

    err_vmax = float(np.percentile(err_mag, 95))
    axes[2].scatter(af_2d[:, 0], af_2d[:, 1], s=0.5, c="black", alpha=0.5, zorder=1)
    sc = axes[2].scatter(p2d[:, 0], p2d[:, 1], c=err_mag, cmap="hot",
                         s=2, vmin=0, vmax=err_vmax)
    axes[2].set_title(f"Error magnitude |pred − gt| (t={timestep_idx})", fontsize=13)
    axes[2].set_xlabel(x_lbl); axes[2].set_ylabel(y_lbl); axes[2].set_aspect("equal")
    fig.colorbar(sc, ax=axes[2], label="error")

    out_path = os.path.join(output_dir,
                            f"{sample_name}_t{timestep_idx}_quiver_{slice_axis}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_streamlines_png(
    pos, gt_vec, pred_vec, is_airfoil, timestep_idx,
    sample_name, output_dir,
    slice_axis="z",
    slice_value=None,           # slice location along slice_axis; None = airfoil-median
    slice_thickness=0.05,       # keep points within ±thickness of slice_value
    grid_resolution=200,        # 2D regular grid size (NxN)
    density=1.5,                # streamplot density (higher = more streamlines)
):
    """2D streamlines on a slice — classic CFD viz.

    Pipeline:
      1) Pick a thin slice of 3D points near `slice_value` along `slice_axis`.
      2) Interpolate 2D-projected velocity onto a regular grid (scipy griddata).
      3) matplotlib streamplot draws continuously integrated flow lines.
      4) Overlay airfoil outline (points in slice mask).

    Three panels: GT streamlines | Pred streamlines | Error heatmap.
    Streamlines colored by |v|.

    Returns output PNG path.
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
    plot_axes = [a for a in (0, 1, 2) if a != axis_idx]
    x_lbl = "xyz"[plot_axes[0]]
    y_lbl = "xyz"[plot_axes[1]]

    # Default slice location = median z among airfoil points (stays on one body).
    if slice_value is None:
        slice_value = float(np.median(pos[is_airfoil, axis_idx]))

    mask = np.abs(pos[:, axis_idx] - slice_value) <= slice_thickness
    idx = np.where(mask)[0]
    if len(idx) < 100:
        print(f"  WARNING: only {len(idx)} points in slice — try larger --stream-thickness.")
        return None

    p2d = pos[idx][:, plot_axes]
    gt2d = gt_vec[idx][:, plot_axes]
    pred2d = pred_vec[idx][:, plot_axes]

    # Build regular 2D grid spanning the slice extent (bounded by the full mesh,
    # so streamlines don't run off the edge).
    xmin, ymin = p2d.min(axis=0)
    xmax, ymax = p2d.max(axis=0)
    # matplotlib.streamplot requires EXACTLY evenly-spaced 1D arrays. We build
    # those and let meshgrid broadcast for the griddata interpolation.
    x_1d = np.linspace(xmin, xmax, grid_resolution)
    y_1d = np.linspace(ymin, ymax, grid_resolution)
    grid_x, grid_y = np.meshgrid(x_1d, y_1d)

    def interp_field(vec2d):
        # griddata returns NaN where out of hull; fill with 0 so streamplot doesn't choke.
        u = griddata(p2d, vec2d[:, 0], (grid_x, grid_y), method="linear", fill_value=0.0)
        v = griddata(p2d, vec2d[:, 1], (grid_x, grid_y), method="linear", fill_value=0.0)
        return u, v

    gt_u, gt_v = interp_field(gt2d)
    pred_u, pred_v = interp_field(pred2d)
    gt_mag = np.sqrt(gt_u ** 2 + gt_v ** 2)
    pred_mag = np.sqrt(pred_u ** 2 + pred_v ** 2)
    err_u, err_v = pred_u - gt_u, pred_v - gt_v
    err_mag = np.sqrt(err_u ** 2 + err_v ** 2)

    # Airfoil points in slice (for outline)
    af_slice_idx = np.where(mask & is_airfoil)[0]
    af_2d = pos[af_slice_idx][:, plot_axes] if len(af_slice_idx) else None

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    vmax = float(np.percentile(gt_mag[gt_mag > 0], 95)) if (gt_mag > 0).any() else 1.0

    for ax, title, U, V, mag in [
        (axes[0], f"Ground truth streamlines (t={timestep_idx})", gt_u, gt_v, gt_mag),
        (axes[1], f"Predicted streamlines (t={timestep_idx})", pred_u, pred_v, pred_mag),
    ]:
        strm = ax.streamplot(
            x_1d, y_1d, U, V,
            color=mag, cmap="turbo", density=density,
            linewidth=1.0, arrowsize=1.0,
        )
        if af_2d is not None:
            ax.scatter(af_2d[:, 0], af_2d[:, 1], s=1.0, c="black", alpha=0.6, zorder=10)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(title); ax.set_xlabel(x_lbl); ax.set_ylabel(y_lbl)
        fig.colorbar(strm.lines, ax=ax, label="|v|")

    # Error panel: error magnitude as imshow heatmap
    im = axes[2].imshow(
        err_mag, extent=(xmin, xmax, ymin, ymax),
        origin="lower", cmap="hot", aspect="equal",
        vmin=0, vmax=float(np.percentile(err_mag, 95)),
    )
    if af_2d is not None:
        axes[2].scatter(af_2d[:, 0], af_2d[:, 1], s=1.0, c="cyan", alpha=0.8, zorder=10)
    axes[2].set_title(f"Streamline error magnitude (t={timestep_idx})")
    axes[2].set_xlabel(x_lbl); axes[2].set_ylabel(y_lbl)
    fig.colorbar(im, ax=axes[2], label="|pred−gt|")

    fig.suptitle(
        f"Slice: {slice_axis}={slice_value:.3f} ±{slice_thickness:.3f}  "
        f"({len(idx):,} points interpolated to {grid_resolution}² grid)",
        fontsize=11,
    )

    out_path = os.path.join(output_dir,
                            f"{sample_name}_t{timestep_idx}_streamlines_{slice_axis}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_vorticity_html(pos, vort_gt, vort_pred, is_airfoil, timestep_idx,
                           sample_name, output_dir, subsample=50_000):
    """3D heatmap of vorticity magnitude — where the vortices live.

    Exports a single HTML with THREE togglable scalar fields on the same cloud:
      * vorticity_gt   — ground-truth |ω| (where real vortices are)
      * vorticity_pred — model's |ω|   (where model thinks vortices are)
      * vorticity_err  — pred − gt     (diverging: over/under-predicted rotation)

    Default shown: vorticity_gt (see true flow). Switch fields via the trame
    viewer's scalar selector in the browser.
    """
    import pyvista as pv

    N = pos.shape[0]
    if N > subsample:
        rng = np.random.RandomState(42)
        airfoil_idx = np.where(is_airfoil)[0]
        idx = np.unique(np.concatenate([
            rng.choice(N, subsample, replace=False),
            airfoil_idx,
        ]))
        pos_s = pos[idx]
        vort_gt_s = vort_gt[idx]
        vort_pred_s = vort_pred[idx]
        airfoil_s = is_airfoil[idx]
    else:
        pos_s = pos
        vort_gt_s = vort_gt
        vort_pred_s = vort_pred
        airfoil_s = is_airfoil

    cloud = pv.PolyData(pos_s.astype(np.float32))
    cloud["vorticity_gt"]   = vort_gt_s.astype(np.float32)
    cloud["vorticity_pred"] = vort_pred_s.astype(np.float32)
    cloud["vorticity_err"]  = (vort_pred_s - vort_gt_s).astype(np.float32)

    # Color range: use 95th percentile of GT magnitudes (vorticity can have
    # long tails right at vortex cores).
    clim_mag = [0.0, float(np.percentile(vort_gt_s, 95))]

    pl = pv.Plotter()
    pl.add_mesh(cloud, scalars="vorticity_gt", cmap="hot",
                point_size=3.0, render_points_as_spheres=False,
                show_scalar_bar=True, clim=clim_mag)
    pl.add_axes(interactive=False)

    bmin, bmax = pos_s.min(axis=0), pos_s.max(axis=0)
    center = 0.5 * (bmin + bmax)
    diag = float(np.linalg.norm(bmax - bmin))
    pl.camera.focal_point = tuple(center)
    pl.camera.position = (center[0], center[1], center[2] + 1.8 * diag)
    pl.camera.up = (0, 1, 0)

    out_path = os.path.join(output_dir, f"{sample_name}_t{timestep_idx}_vorticity.html")
    pl.export_html(out_path)
    pl.close()
    return out_path


# ---------------------------------------------------------------------------
# Arrow (vector glyph) viz — side-by-side GT vs pred vectors
# ---------------------------------------------------------------------------

def export_arrows_html(pos, gt_vec, pred_vec, is_airfoil, timestep_idx,
                       sample_name, output_dir, subsample=1500,
                       glyph_scale=None, camera_distance=None):
    """Interactive 3D viz: GT velocity arrows (blue) + pred arrows (red) overlaid.

    Where the two colors diverge → model is wrong about direction/magnitude.
    Where they overlap → model is accurate.

    Args:
        pos:        (N, 3) numpy
        gt_vec:     (N, 3) numpy — ground truth velocity at timestep t
        pred_vec:   (N, 3) numpy — predicted velocity at timestep t
        is_airfoil: (N,) bool
        subsample:  arrows to render. `0` = no subsampling (one vector per point,
                    all 100k). Finite value = sampled subset, biased to airfoil.
        glyph_scale: arrow length = scale · |v|. If None, auto-sized so the
                    largest arrow is ~4% of the mesh diagonal (dense regimes get
                    smaller arrows automatically).
        camera_distance: None = auto (1.8x mesh diagonal). Larger = start zoomed out.

    Browser controls (standard vtk.js, same in all HTML exports):
      - Left-click + drag: ROTATE
      - Right-click + drag (or Ctrl+left-drag): PAN
      - Scroll wheel: ZOOM
      - Middle-click + drag: alternate PAN

    NOTE on `subsample=0` (one vector per point):
      - File size can hit ~100-200 MB for 100k arrows × 2 (GT + pred).
      - Browser can handle it but takes ~10-30s to load the HTML.
      - Arrows get auto-scaled smaller for visual legibility at density.
    """
    import pyvista as pv

    N = pos.shape[0]

    if subsample <= 0 or subsample >= N:
        # Render one vector per point — no subsampling.
        idx = np.arange(N)
    else:
        # Subsample biased toward airfoil + near-surface points.
        rng = np.random.RandomState(42)
        airfoil_idx = np.where(is_airfoil)[0]
        non_airfoil_idx = np.where(~is_airfoil)[0]
        n_af = min(len(airfoil_idx), subsample // 2)
        n_other = subsample - n_af
        idx = np.concatenate([
            rng.choice(airfoil_idx, n_af, replace=False) if n_af > 0 else np.array([], dtype=int),
            rng.choice(non_airfoil_idx, min(n_other, len(non_airfoil_idx)), replace=False)
                if n_other > 0 else np.array([], dtype=int),
        ])

    pos_s = pos[idx].astype(np.float32)
    gt_s = gt_vec[idx].astype(np.float32)
    pred_s = pred_vec[idx].astype(np.float32)

    # --- Auto-scale arrows relative to mesh geometry AND point density ---
    # The more points we render, the shorter each arrow needs to be so they
    # don't overlap. Scale target ~= typical local spacing, which goes as
    # (volume / N_points)^(1/3).
    bmin, bmax = pos_s.min(axis=0), pos_s.max(axis=0)
    diag = float(np.linalg.norm(bmax - bmin))
    # Rough local spacing estimate from density.
    n_rendered = len(pos_s)
    spacing = diag / max(n_rendered ** (1/3), 1.0)
    max_vel = max(float(np.linalg.norm(gt_s, axis=1).max()),
                  float(np.linalg.norm(pred_s, axis=1).max()), 1e-6)
    if glyph_scale is None:
        # Largest arrow ≈ 1.5x local spacing — visible but not overlapping too much.
        glyph_scale = 1.5 * spacing / max_vel

    cloud = pv.PolyData(pos_s)
    cloud["gt_vel"] = gt_s
    cloud["pred_vel"] = pred_s
    cloud["gt_mag"] = np.linalg.norm(gt_s, axis=1)
    cloud["pred_mag"] = np.linalg.norm(pred_s, axis=1)

    # Arrow glyphs — smaller shaft, smaller tip = less visual clutter.
    arrow_geom = pv.Arrow(tip_length=0.15, tip_radius=0.005, shaft_radius=0.001)
    gt_arrows = cloud.glyph(orient="gt_vel", scale="gt_mag",
                             factor=glyph_scale, geom=arrow_geom)
    pred_arrows = cloud.glyph(orient="pred_vel", scale="pred_mag",
                               factor=glyph_scale, geom=arrow_geom)

    pl = pv.Plotter()
    pl.add_mesh(gt_arrows, color="royalblue", opacity=0.9, label="Ground truth")
    pl.add_mesh(pred_arrows, color="crimson", opacity=0.9, label="Prediction")
    # Context points (very faint, small) so you see mesh structure.
    pl.add_mesh(cloud, color="lightgray", point_size=0.8, opacity=0.25)
    pl.add_legend()
    # Orientation indicator (XYZ axes cube in the corner).
    pl.add_axes(interactive=False)

    # --- Camera setup: position so the whole mesh is visible, zoomed-out start ---
    center = 0.5 * (bmin + bmax)
    if camera_distance is None:
        camera_distance = 1.8 * diag   # ~1.8x diagonal = comfortable starting view
    # View from +Z looking down — simple default that matches "top-down airfoil" convention.
    pl.camera.focal_point = tuple(center)
    pl.camera.position = (center[0], center[1], center[2] + camera_distance)
    pl.camera.up = (0, 1, 0)
    pl.camera.parallel_projection = False   # perspective gives depth cues

    out_path = os.path.join(output_dir, f"{sample_name}_t{timestep_idx}_arrows.html")
    pl.export_html(out_path)
    pl.close()
    return out_path


# ---------------------------------------------------------------------------
# Pressure field viz (scalar per point)
# ---------------------------------------------------------------------------

def export_pressure_html(pos, pressure, is_airfoil, timestep_idx,
                         sample_name, output_dir, subsample=50_000):
    """Points colored by pressure — shows stagnation, shocks, wake structure.

    Args:
        pos:        (N, 3) numpy
        pressure:   (N,) numpy scalar field at this timestep
        is_airfoil: (N,) bool
        timestep_idx, sample_name, output_dir: strings/ints
        subsample:  max points (for HTML file-size / renderer perf)
    """
    import pyvista as pv

    N = pos.shape[0]
    if N > subsample:
        rng = np.random.RandomState(42)
        airfoil_idx = np.where(is_airfoil)[0]
        idx = np.unique(np.concatenate([
            rng.choice(N, subsample, replace=False),
            airfoil_idx,  # always include surface
        ]))
        pos_s = pos[idx]
        pressure_s = pressure[idx]
        airfoil_s = is_airfoil[idx]
    else:
        pos_s = pos
        pressure_s = pressure
        airfoil_s = is_airfoil

    cloud = pv.PolyData(pos_s.astype(np.float32))
    cloud["pressure"] = pressure_s.astype(np.float32)

    # Diverging colormap centered on freestream pressure (approx 0).
    # Pressure range is large (-3600 to 1200), so clamp to 95th percentile.
    lo = float(np.percentile(pressure_s, 2))
    hi = float(np.percentile(pressure_s, 98))
    clim = [lo, hi]

    pl = pv.Plotter()
    pl.add_mesh(cloud, scalars="pressure", cmap="coolwarm",
                point_size=3.0, render_points_as_spheres=False,
                show_scalar_bar=True, clim=clim)
    pl.add_axes(interactive=False)  # XYZ orientation widget in corner

    # Camera: full-mesh starting view (not crammed at the airfoil center).
    bmin, bmax = pos_s.min(axis=0), pos_s.max(axis=0)
    center = 0.5 * (bmin + bmax)
    diag = float(np.linalg.norm(bmax - bmin))
    pl.camera.focal_point = tuple(center)
    pl.camera.position = (center[0], center[1], center[2] + 1.8 * diag)
    pl.camera.up = (0, 1, 0)

    out_path = os.path.join(output_dir, f"{sample_name}_t{timestep_idx}_pressure.html")
    pl.export_html(out_path)
    pl.close()
    return out_path


# ---------------------------------------------------------------------------
# CFD-pro view — solid airfoil mesh + velocity glyphs colored by |v|
# ---------------------------------------------------------------------------

def _reconstruct_airfoil_mesh(airfoil_pts):
    """Reconstruct a surface mesh from the airfoil point set.

    Uses alpha-shape Delaunay3D with alpha auto-tuned from local point spacing.
    Returns a pv.PolyData surface, or None if reconstruction fails.
    """
    import pyvista as pv
    from scipy.spatial import cKDTree

    pts32 = airfoil_pts.astype(np.float32)
    tree = cKDTree(pts32)
    dists, _ = tree.query(pts32, k=2)   # nearest non-self neighbor
    median_nn = float(np.median(dists[:, 1]))
    # alpha = max edge length; ~4x median spacing keeps thin airfoil features.
    alpha = 4.0 * median_nn

    cloud = pv.PolyData(pts32)
    try:
        tets = cloud.delaunay_3d(alpha=alpha)
        surf = tets.extract_surface().triangulate()
        if surf.n_points == 0 or surf.n_cells == 0:
            return None
        return surf
    except Exception:
        return None


def export_cfd_view(pos, velocity_pred, velocity_gt, idcs_airfoil,
                    timestep_idx, sample_name, output_dir,
                    n_arrows=3000, bl_fraction=0.15,
                    which="pred"):
    """CFD presentation view: solid airfoil + velocity arrows colored by |v|.

    which: "pred" — model output only
           "gt"   — ground truth only
           "both" — GT (blue) + pred (red) overlaid on the same airfoil

    n_arrows: number of glyphs (too many = unreadable mess).
    bl_fraction: keep arrows within this fraction of the mesh diagonal to the
                 airfoil — biases density to the boundary layer + wake.
    """
    import pyvista as pv
    from scipy.spatial import cKDTree

    airfoil_pts = pos[idcs_airfoil]
    bmin, bmax = pos.min(axis=0), pos.max(axis=0)
    diag = float(np.linalg.norm(bmax - bmin))

    # Filter arrow candidates to the near-body region for clarity.
    tree = cKDTree(airfoil_pts.astype(np.float32))
    udf, _ = tree.query(pos.astype(np.float32), k=1)
    band = bl_fraction * diag
    near = np.where(udf < band)[0]
    if len(near) < n_arrows:
        near = np.argsort(udf)[:max(n_arrows, 500)]

    rng = np.random.RandomState(42)
    if len(near) > n_arrows:
        near = rng.choice(near, n_arrows, replace=False)

    sub_pos = pos[near].astype(np.float32)
    vel_pred = velocity_pred[near].astype(np.float32)
    vel_gt   = velocity_gt[near].astype(np.float32)
    vmag_pred = np.linalg.norm(vel_pred, axis=1)
    vmag_gt   = np.linalg.norm(vel_gt, axis=1)

    max_v = float(max(vmag_pred.max(), vmag_gt.max(), 1e-6))
    spacing = diag / max(len(sub_pos) ** (1/3), 1.0)
    glyph_scale = 1.5 * spacing / max_v

    arrow_geom = pv.Arrow(tip_length=0.15, tip_radius=0.01, shaft_radius=0.002)

    # Build glyph meshes
    cloud = pv.PolyData(sub_pos)
    cloud["velocity_pred"] = vel_pred
    cloud["velocity_gt"]   = vel_gt
    cloud["vmag_pred"]     = vmag_pred
    cloud["vmag_gt"]       = vmag_gt

    pred_arrows = cloud.glyph(orient="velocity_pred", scale="vmag_pred",
                              factor=glyph_scale, geom=arrow_geom)
    gt_arrows   = cloud.glyph(orient="velocity_gt", scale="vmag_gt",
                              factor=glyph_scale, geom=arrow_geom)

    # Airfoil surface reconstruction (falls back to dense points on failure)
    airfoil_mesh = _reconstruct_airfoil_mesh(airfoil_pts)

    pl = pv.Plotter()
    if airfoil_mesh is not None:
        pl.add_mesh(airfoil_mesh, color="lightgray", opacity=1.0,
                    show_edges=False, smooth_shading=True, ambient=0.3,
                    diffuse=0.7, specular=0.2)
    else:
        pl.add_mesh(pv.PolyData(airfoil_pts.astype(np.float32)),
                    color="dimgray", point_size=3.0,
                    render_points_as_spheres=True)

    # Arrow rendering
    clim = [0.0, float(np.percentile(
        np.concatenate([vmag_pred, vmag_gt]), 95))]

    if which == "pred":
        pl.add_mesh(pred_arrows, scalars="vmag_pred", cmap="turbo",
                    show_scalar_bar=True, clim=clim,
                    scalar_bar_args={"title": "|v| (pred)"})
    elif which == "gt":
        pl.add_mesh(gt_arrows, scalars="vmag_gt", cmap="turbo",
                    show_scalar_bar=True, clim=clim,
                    scalar_bar_args={"title": "|v| (GT)"})
    elif which == "both":
        pl.add_mesh(gt_arrows, color="royalblue", opacity=0.85,
                    label="Ground truth")
        pl.add_mesh(pred_arrows, color="crimson", opacity=0.85,
                    label="Prediction")
        pl.add_legend()
    else:
        raise ValueError(f"which must be 'pred'|'gt'|'both', got {which!r}")

    pl.add_axes(interactive=False)

    center = 0.5 * (bmin + bmax)
    pl.camera.focal_point = tuple(center)
    pl.camera.position = (center[0], center[1], center[2] + 1.5 * diag)
    pl.camera.up = (0, 1, 0)

    out_path = os.path.join(output_dir,
                            f"{sample_name}_t{timestep_idx}_cfd_{which}.html")
    pl.export_html(out_path)
    pl.close()
    return out_path


# ---------------------------------------------------------------------------
# Native VTK export (.vtp per timestep + .pvd time-series collection)
# ---------------------------------------------------------------------------

def export_vtp(pos, gt_vel, pred_vel, is_airfoil, err_mag, vmag_gt, vmag_pred,
               grad_gt, grad_pred, vort_gt, vort_pred, pressure,
               timestep_idx, sample_name, output_dir):
    """Write one .vtp (VTK PolyData) with vectors + scalars as a point cloud.

    In ParaView:
      - File -> Open the .pvd, hit Play on the time slider
      - Glyph filter -> orient by `velocity_pred` or `velocity_error`
      - Color by `error_magnitude`, `vorticity_gt`, etc.

    Args that may be None are simply skipped as arrays.
    """
    import pyvista as pv

    cloud = pv.PolyData(pos.astype(np.float32))

    cloud["velocity_gt"]    = gt_vel.astype(np.float32)
    cloud["velocity_pred"]  = pred_vel.astype(np.float32)
    cloud["velocity_error"] = (pred_vel - gt_vel).astype(np.float32)

    cloud["error_magnitude"] = err_mag.astype(np.float32)
    cloud["vmag_gt"]         = vmag_gt.astype(np.float32)
    cloud["vmag_pred"]       = vmag_pred.astype(np.float32)
    cloud["grad_mag_gt"]     = grad_gt.astype(np.float32)
    cloud["grad_mag_pred"]   = grad_pred.astype(np.float32)

    if vort_gt is not None:
        cloud["vorticity_gt"]   = vort_gt.astype(np.float32)
    if vort_pred is not None:
        cloud["vorticity_pred"] = vort_pred.astype(np.float32)
        if vort_gt is not None:
            cloud["vorticity_error"] = (vort_pred - vort_gt).astype(np.float32)

    if pressure is not None:
        cloud["pressure"] = pressure.astype(np.float32)

    cloud["is_airfoil"] = is_airfoil.astype(np.uint8)

    out_path = os.path.join(output_dir, f"{sample_name}_t{timestep_idx}.vtp")
    cloud.save(out_path, binary=True)
    return out_path


def write_pvd(vtp_paths, times, sample_name, output_dir):
    """Write a .pvd collection so ParaView treats the .vtp files as a time series."""
    rel = [os.path.basename(p) for p in vtp_paths]
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             '  <Collection>']
    for t, f in zip(times, rel):
        lines.append(f'    <DataSet timestep="{float(t):.6f}" group="" part="0" '
                     f'file="{f}"/>')
    lines.append('  </Collection>')
    lines.append('</VTKFile>')

    pvd_path = os.path.join(output_dir, f"{sample_name}.pvd")
    with open(pvd_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return pvd_path


# ---------------------------------------------------------------------------
# Competition metric (inlined — no import from src.train)
# ---------------------------------------------------------------------------

def competition_metric_per_timestep(pred, gt):
    """Per-timestep L2 error: (5,) array."""
    # pred, gt: (5, N, 3)
    return (pred - gt).norm(dim=2).mean(dim=1).numpy()  # (5,)


def competition_metric(pred, gt):
    """Overall metric matching competition formula."""
    # pred, gt: (5, N, 3)
    return (pred - gt).norm(dim=2).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Export model predictions as CSV + PNG for visualization")
    p.add_argument("--model", default=None, help="Model name (e.g. residual_mlp)")
    p.add_argument("--checkpoint", default=None, help="Path to best_model.pt")
    p.add_argument("--predictions", default=None,
                   help="Path to saved predictions .pt file (skips model inference)")
    p.add_argument("--data-path", default="data/",
                   help="Single .npz file or directory")
    p.add_argument("--sample-index", type=int, default=0,
                   help="Index into sorted file list (when data-path is a dir)")
    p.add_argument("--output-dir", default="figures/predictions")
    p.add_argument("--timesteps", type=int, nargs="*", default=None,
                   help="Output timesteps to export (0-4, default: all)")
    p.add_argument("--features", type=str, nargs="*", default=None,
                   help="Override model features")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    p.add_argument("--no-render", action="store_true", help="Skip PyVista PNGs")
    p.add_argument("--html", action="store_true",
                   help="Export interactive 3D HTML (open in browser)")
    p.add_argument("--subsample", type=int, default=0,
                   help="Subsample points for HTML (0 = auto: 50k)")
    p.add_argument("--arrows", action="store_true",
                   help="Export HTML with GT (blue) + pred (red) velocity arrows "
                        "— directly shows where the model disagrees on direction.")
    p.add_argument("--arrows-subsample", type=int, default=1500,
                   help="Number of arrows to render (too many = visual soup). Default 1500.")
    p.add_argument("--arrows-scale", type=float, default=None,
                   help="Manual arrow-length scale factor. If omitted, auto-sized "
                        "so the largest arrow is ~4%% of the mesh diagonal.")
    p.add_argument("--pressure", action="store_true",
                   help="Export HTML with pressure field (loaded from NPZ). "
                        "Shows stagnation points, shocks, wake structure.")
    p.add_argument("--vorticity", action="store_true",
                   help="Export HTML with vorticity magnitude |∇×v| (per-point, "
                        "computed via k-NN least-squares on velocity gradient). "
                        "Toggleable fields: vorticity_gt / _pred / _err.")
    p.add_argument("--turb-arrows", action="store_true",
                   help="Simplified arrow viz: only in turbulent regions (top "
                        "20%% vorticity points), colored by velocity magnitude. "
                        "One file per timestep for GT and one for pred.")
    p.add_argument("--turb-percentile", type=float, default=80.0,
                   help="Keep arrows where vorticity > N-th percentile. Default 80.")
    p.add_argument("--bl-max-dist", type=float, default=0.0,
                   help="Boundary-layer-only arrow filter: keep only points with "
                        "UDF (distance to airfoil) ≤ this value. 0 = disabled. "
                        "Try 0.05 for BL, 0.01 for viscous sublayer only.")
    p.add_argument("--vtp", action="store_true",
                   help="Export native .vtp per timestep + .pvd collection for "
                        "ParaView (velocity as true vector arrays, all scalars). "
                        "Open the .pvd in ParaView, Glyph filter -> orient by "
                        "velocity_pred/_gt/_error.")
    p.add_argument("--cfd", action="store_true",
                   help="CFD-pro presentation view: solid reconstructed airfoil "
                        "mesh + velocity arrows colored by |v|. Produces three "
                        "interactive HTML files per timestep: _pred, _gt, _both.")
    p.add_argument("--cfd-arrows", type=int, default=3000,
                   help="Number of velocity glyphs in CFD view. Default 3000.")
    p.add_argument("--cfd-band", type=float, default=0.15,
                   help="Keep arrows within this fraction of the mesh diagonal "
                        "from the airfoil (1.0 = whole domain). Default 0.15.")
    p.add_argument("--quiver", action="store_true",
                   help="2D matplotlib PNG (GT | pred | error panels). No trame "
                        "mouse controls — opens in any image viewer, pan via "
                        "standard shortcuts.")
    p.add_argument("--quiver-slice-axis", default="z", choices=["x", "y", "z"],
                   help="Project along this axis (default z → xy plane).")
    p.add_argument("--quiver-slice-thickness", type=float, default=None,
                   help="Keep only points within ±thickness of airfoil-center "
                        "plane (e.g. 0.1 for a thin wake slice).")
    p.add_argument("--include-inputs", action="store_true",
                   help="Also render the 5 INPUT velocity frames (single-panel "
                        "streamlines, GT only — no prediction since the model "
                        "doesn't predict input frames). Lets you flip through "
                        "all 10 frames (5 in, 5 out) to see flow evolution.")
    p.add_argument("--streamlines", action="store_true",
                   help="2D matplotlib streamplot PNG: continuous integrated "
                        "flow lines on a thin slice. Classic CFD textbook viz.")
    p.add_argument("--stream-slice-axis", default="z", choices=["x", "y", "z"],
                   help="Slice along this axis for streamlines (default z).")
    p.add_argument("--stream-slice-value", type=float, default=None,
                   help="Slice location. None = airfoil median along that axis "
                        "(stays on one body when z has multiple clusters).")
    p.add_argument("--stream-slice-thickness", type=float, default=0.05,
                   help="Include points within ±thickness of slice_value. Default 0.05.")
    p.add_argument("--stream-resolution", type=int, default=200,
                   help="Regular grid resolution for interpolation. Default 200.")
    p.add_argument("--stream-density", type=float, default=1.5,
                   help="matplotlib streamplot density (higher = more lines). Default 1.5.")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve data path
    if os.path.isdir(args.data_path):
        npz_files = sorted(glob.glob(os.path.join(args.data_path, "*.npz")))
        if not npz_files:
            print(f"No .npz files found in {args.data_path}")
            sys.exit(1)
        if args.sample_index >= len(npz_files):
            print(f"sample-index {args.sample_index} out of range "
                  f"(have {len(npz_files)} files)")
            sys.exit(1)
        npz_path = npz_files[args.sample_index]
    else:
        npz_path = args.data_path

    sample_name = os.path.splitext(os.path.basename(npz_path))[0]
    timesteps = args.timesteps if args.timesteps is not None else list(range(5))

    for ts in timesteps:
        if ts < 0 or ts > 4:
            print(f"Timestep {ts} out of range (0-4)")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    use_predictions_file = args.predictions is not None

    if use_predictions_file:
        # Load pre-saved predictions — no model needed
        if not os.path.exists(args.predictions):
            print(f"Predictions file not found: {args.predictions}")
            sys.exit(1)

        saved = torch.load(args.predictions, weights_only=True, map_location="cpu")
        pred = saved["pred"]    # (5, N, 3)
        gt = saved["gt"]        # (5, N, 3)
        pos = saved["pos"]      # (N, 3)
        idcs_airfoil = saved["idcs_airfoil"]  # (M,)
        knn = saved.get("knn_graph", None)    # (N, k) or None

        features = []  # not needed

        print(f"Sample:      {npz_path}")
        print(f"Predictions: {args.predictions}")
        print(f"Timesteps:   {timesteps}")
        print()
    else:
        # Run model inference
        if args.model is None or args.checkpoint is None:
            print("ERROR: --model and --checkpoint required (or use --predictions)")
            sys.exit(1)

        from src.train import get_model_class
        ModelClass = get_model_class(args.model)
        features = args.features if args.features is not None else \
            getattr(ModelClass, "FEATURES", []) or []

        print(f"Sample:     {npz_path}")
        print(f"Model:      {args.model}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Features:   {features}")
        print(f"Device:     {device}")
        print(f"Timesteps:  {timesteps}")
        print()

        model = load_model(args.model, args.checkpoint, features, device)
        sample = load_sample(npz_path, features, device)

        print("Running inference...")
        pred = run_inference(model, sample)

        # Remove batch dim: (5, N, 3)
        pred = pred[0].cpu()
        gt = sample["velocity_out"][0].cpu()
        pos = sample["pos"][0].cpu()
        idcs_airfoil = sample["idcs_airfoil"][0].cpu()
        knn = sample["knn_graph"][0].cpu() if "knn_graph" in sample else None

    # Airfoil mask
    is_airfoil = torch.zeros(pos.shape[0], dtype=torch.bool)
    is_airfoil[idcs_airfoil] = True

    N = pos.shape[0]
    M = idcs_airfoil.shape[0]

    # Overall metric
    overall = competition_metric(pred, gt)
    per_ts = competition_metric_per_timestep(pred, gt)
    print(f"Overall metric: {overall:.6f}")
    print(f"Points: {N:,} ({M:,} airfoil)")
    print()

    # Load pressure field from NPZ if --pressure requested.
    # pressure shape: (10, N) — first 5 = input timesteps, last 5 = output timesteps.
    pressure_all = None
    if args.pressure:
        _d = np.load(npz_path)
        if "pressure" in _d.files:
            pressure_all = _d["pressure"]  # (10, N)
            print(f"Pressure loaded: shape {pressure_all.shape}")
        else:
            print("WARNING: --pressure requested but 'pressure' field not in NPZ")
            args.pressure = False

    # Per-timestep export
    header = f"{'Step':>4}  {'Metric':>8}"
    if not args.no_csv:
        header += f"  {'CSV':>12}"
    if not args.no_render:
        header += f"  {'PNGs':>6}"
    if args.html:
        header += f"  {'HTML':>8}"
    if args.arrows:
        header += f"  {'Arrows':>8}"
    if args.pressure:
        header += f"  {'Pressure':>10}"
    if args.vorticity:
        header += f"  {'Vorticity':>10}"
    if args.turb_arrows:
        header += f"  {'TurbArr':>9}"
    if args.cfd:
        header += f"  {'CFD':>8}"
    if args.vtp:
        header += f"  {'VTP':>8}"
    if args.quiver:
        header += f"  {'Quiver':>10}"
    if args.streamlines:
        header += f"  {'Stream':>8}"
    print(header)
    print("-" * len(header))

    vtp_paths = []

    for ts in timesteps:
        gt_t = gt[ts]      # (N, 3)
        pred_t = pred[ts]  # (N, 3)
        metric = per_ts[ts]

        # Gradient magnitudes
        grad_gt = compute_velocity_gradients(gt_t, pos, knn)
        grad_pred = compute_velocity_gradients(pred_t, pos, knn)

        # To numpy
        pos_np = pos.numpy()
        gt_np = gt_t.numpy()
        pred_np = pred_t.numpy()
        is_airfoil_np = is_airfoil.numpy().astype(np.float32)
        is_airfoil_bool = is_airfoil.numpy()
        grad_gt_np = grad_gt.numpy()
        grad_pred_np = grad_pred.numpy()

        err_np = pred_np - gt_np
        err_mag = np.linalg.norm(err_np, axis=1)
        vmag_gt = np.linalg.norm(gt_np, axis=1)
        vmag_pred = np.linalg.norm(pred_np, axis=1)

        scalars = {
            "error_magnitude": err_mag,
            "vmag_gt": vmag_gt,
            "vmag_pred": vmag_pred,
            "grad_mag_gt": grad_gt_np,
            "grad_mag_pred": grad_pred_np,
        }

        line = f"  t={ts}  {metric:>8.4f}"

        # CSV export
        if not args.no_csv:
            csv_path = os.path.join(args.output_dir, f"{sample_name}_t{ts}.csv")
            export_csv(pos_np, gt_np, pred_np, is_airfoil_np,
                       grad_gt_np, grad_pred_np, ts, csv_path)
            line += f"  {os.path.getsize(csv_path) / 1e6:.1f}MB".rjust(12)

        # PyVista PNGs
        if not args.no_render:
            render_pyvista(pos_np, scalars, is_airfoil_bool,
                           ts, sample_name, args.output_dir)
            prefix = f"{sample_name}_t{ts}_"
            n_pngs = len([f for f in os.listdir(args.output_dir)
                          if f.startswith(prefix) and f.endswith(".png")])
            line += f"  {n_pngs} PNGs".rjust(6)

        # Interactive HTML
        if args.html:
            html_path = export_html(pos_np, scalars, is_airfoil_bool,
                                    ts, sample_name, args.output_dir,
                                    args.subsample)
            size_mb = os.path.getsize(html_path) / 1e6
            line += f"  {size_mb:.1f}MB".rjust(8)

        # Arrow viz (GT vs pred vectors)
        if args.arrows:
            arr_path = export_arrows_html(
                pos_np, gt_np, pred_np, is_airfoil_bool,
                ts, sample_name, args.output_dir,
                subsample=args.arrows_subsample,
                glyph_scale=args.arrows_scale,
            )
            size_mb = os.path.getsize(arr_path) / 1e6
            line += f"  {size_mb:.1f}MB".rjust(8)

        # Pressure viz
        if args.pressure and pressure_all is not None:
            # Output timestep t corresponds to pressure[5 + t] (last 5 frames)
            pressure_t = pressure_all[5 + ts]
            p_path = export_pressure_html(
                pos_np, pressure_t, is_airfoil_bool,
                ts, sample_name, args.output_dir,
            )
            size_mb = os.path.getsize(p_path) / 1e6
            line += f"  {size_mb:.1f}MB".rjust(10)

        # Vorticity viz (|∇×v| per point, GT vs pred)
        # Computed once, potentially reused by turb-arrows and vtp below.
        vort_gt_np = None
        vort_pred_np = None
        if args.vorticity or args.turb_arrows or args.vtp:
            vort_gt_np = compute_vorticity_magnitude(gt_t, pos, knn).numpy()
            vort_pred_np = compute_vorticity_magnitude(pred_t, pos, knn).numpy()

        if args.vorticity:
            v_path = export_vorticity_html(
                pos_np, vort_gt_np, vort_pred_np, is_airfoil_bool,
                ts, sample_name, args.output_dir,
            )
            size_mb = os.path.getsize(v_path) / 1e6
            line += f"  {size_mb:.1f}MB".rjust(10)

        # Turbulent-region arrows — one file for GT, one for pred.
        if args.turb_arrows:
            # Compute UDF once per timestep if BL filter is enabled.
            udf_np = None
            if args.bl_max_dist > 0:
                from src.features import _chunked_min_dist
                surface_pts = pos[idcs_airfoil]
                udf_np = _chunked_min_dist(pos, surface_pts).numpy()

            gt_path = export_turbulent_arrows_html(
                pos_np, gt_np, vort_gt_np, is_airfoil_bool,
                ts, sample_name, args.output_dir,
                vort_percentile=args.turb_percentile,
                bl_max_dist=args.bl_max_dist, udf=udf_np,
                which="gt",
            )
            pred_path = export_turbulent_arrows_html(
                pos_np, pred_np, vort_pred_np, is_airfoil_bool,
                ts, sample_name, args.output_dir,
                vort_percentile=args.turb_percentile,
                bl_max_dist=args.bl_max_dist, udf=udf_np,
                which="pred",
            )
            total_mb = 0.0
            if gt_path is not None:
                total_mb += os.path.getsize(gt_path) / 1e6
            if pred_path is not None:
                total_mb += os.path.getsize(pred_path) / 1e6
            line += f"  {total_mb:.1f}MB".rjust(9)

        # CFD presentation view (airfoil mesh + glyph arrows)
        if args.cfd:
            total_mb = 0.0
            for which in ("pred", "gt", "both"):
                cfd_path = export_cfd_view(
                    pos_np, pred_np, gt_np,
                    idcs_airfoil.numpy(),
                    ts, sample_name, args.output_dir,
                    n_arrows=args.cfd_arrows,
                    bl_fraction=args.cfd_band,
                    which=which,
                )
                total_mb += os.path.getsize(cfd_path) / 1e6
            line += f"  {total_mb:.1f}MB".rjust(8)

        # Native .vtp (vector arrays + scalars, for ParaView)
        if args.vtp:
            pressure_t = pressure_all[5 + ts] if pressure_all is not None else None
            vtp_path = export_vtp(
                pos_np, gt_np, pred_np, is_airfoil_bool,
                err_mag, vmag_gt, vmag_pred, grad_gt_np, grad_pred_np,
                vort_gt_np, vort_pred_np, pressure_t,
                ts, sample_name, args.output_dir,
            )
            vtp_paths.append(vtp_path)
            size_mb = os.path.getsize(vtp_path) / 1e6
            line += f"  {size_mb:.1f}MB".rjust(8)

        # 2D matplotlib quiver PNG (GT | pred | error).
        if args.quiver:
            if vort_gt_np is None:
                vort_gt_np = compute_vorticity_magnitude(gt_t, pos, knn).numpy()
                vort_pred_np = compute_vorticity_magnitude(pred_t, pos, knn).numpy()
            udf_np = None
            if args.bl_max_dist > 0:
                from src.features import _chunked_min_dist
                surface_pts = pos[idcs_airfoil]
                udf_np = _chunked_min_dist(pos, surface_pts).numpy()
            q_path = export_quiver_png(
                pos_np, gt_np, pred_np, vort_gt_np, is_airfoil_bool,
                ts, sample_name, args.output_dir,
                vort_percentile=args.turb_percentile,
                bl_max_dist=args.bl_max_dist, udf=udf_np,
                slice_axis=args.quiver_slice_axis,
                slice_thickness=args.quiver_slice_thickness,
            )
            if q_path is not None:
                size_mb = os.path.getsize(q_path) / 1e6
                line += f"  {size_mb:.2f}MB".rjust(10)

        # 2D streamline PNG (GT | pred | error).
        if args.streamlines:
            s_path = export_streamlines_png(
                pos_np, gt_np, pred_np, is_airfoil_bool,
                ts, sample_name, args.output_dir,
                slice_axis=args.stream_slice_axis,
                slice_value=args.stream_slice_value,
                slice_thickness=args.stream_slice_thickness,
                grid_resolution=args.stream_resolution,
                density=args.stream_density,
            )
            if s_path is not None:
                size_mb = os.path.getsize(s_path) / 1e6
                line += f"  {size_mb:.2f}MB".rjust(8)

        print(line)

    # Write .pvd time-series collection pointing at the per-timestep .vtp files.
    if args.vtp and vtp_paths:
        t_arr = np.load(npz_path)["t"]  # (10,) — first 5 input, last 5 output
        out_times = [float(t_arr[5 + ts]) for ts in timesteps]
        pvd_path = write_pvd(vtp_paths, out_times, sample_name, args.output_dir)
        print(f"\nPVD collection: {pvd_path}")

    # ── Optional: render the 5 INPUT velocity frames as separate PNGs ──
    # (model never predicts these — they're conditioning data — so we just
    #  visualize GT velocity for each input frame so you can flip through the
    #  full 10-frame sequence: 5 in + 5 out.)
    if args.include_inputs:
        print(f"\nRendering input frames (in_t-5 to in_t-1)...")
        velocity_in = sample.get("velocity_in") if "sample" in dir() else None
        if velocity_in is None:
            # We're in --predictions mode, NPZ holds the input frames.
            _d = np.load(npz_path)
            velocity_in_np = _d["velocity_in"]  # (5, N, 3) numpy
        else:
            velocity_in_np = velocity_in[0].cpu().numpy()  # strip batch dim

        for in_ts in range(5):
            vel_t = velocity_in_np[in_ts]  # (N, 3) — single input frame
            in_label = f"input_t{in_ts}"
            # Simplest: reuse the 3-panel function with pred=gt; left/middle
            # panels show identical flow, right (error) panel is empty.
            # That keeps the rendering style consistent with output frames.
            if args.streamlines:
                s_path = export_streamlines_png(
                    pos_np, vel_t, vel_t, is_airfoil_bool,
                    in_ts - 5,             # encode "input frame i" as t = -(5-i)
                    sample_name + "_" + in_label, args.output_dir,
                    slice_axis=args.stream_slice_axis,
                    slice_value=args.stream_slice_value,
                    slice_thickness=args.stream_slice_thickness,
                    grid_resolution=args.stream_resolution,
                    density=args.stream_density,
                )
                if s_path is not None:
                    print(f"  in_t{in_ts}: {os.path.basename(s_path)}")
            if args.quiver:
                udf_np = None
                if args.bl_max_dist > 0:
                    from src.features import _chunked_min_dist
                    surface_pts = pos[idcs_airfoil]
                    udf_np = _chunked_min_dist(pos, surface_pts).numpy()
                # vorticity for input frame
                vort_in = compute_vorticity_magnitude(
                    torch.from_numpy(vel_t).float(), pos, knn,
                ).numpy()
                q_path = export_quiver_png(
                    pos_np, vel_t, vel_t, vort_in, is_airfoil_bool,
                    in_ts - 5,
                    sample_name + "_" + in_label, args.output_dir,
                    vort_percentile=args.turb_percentile,
                    bl_max_dist=args.bl_max_dist, udf=udf_np,
                    slice_axis=args.quiver_slice_axis,
                    slice_thickness=args.quiver_slice_thickness,
                )
                if q_path is not None:
                    print(f"  in_t{in_ts}: {os.path.basename(q_path)}")

    print()
    print(f"Output directory: {args.output_dir}")
    if args.html or args.arrows or args.pressure:
        print("Open .html files in your browser for interactive 3D.")
    if args.vtp:
        print("Open the .pvd in ParaView (File -> Open). Use Glyph filter "
              "(orient by velocity_pred / _gt / _error) to see vectors.")
    print("Done.")


if __name__ == "__main__":
    main()
