import argparse
from pathlib import Path
import torch
import numpy as np
import os
import sys
import time
from typing import List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factory.anysplat.misc.image_io import save_interpolated_video
from factory.anysplat.model.model.anysplat import AnySplat
from factory.anysplat.utils.image import process_image
from factory.anysplat.model.ply_export import export_ply




# help functions for export @yfian
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def try_export_ply_builtin(gaussians, ply_path: str) -> bool:
    """Try common export methods on the gaussians object."""
    for name in ["save_ply", "to_ply", "export_ply", "write_ply", "save_as_ply"]:
        fn = getattr(gaussians, name, None)
        if callable(fn):
            fn(ply_path)
            return True
    return False

def write_gaussian_ply_binary(
    path: str,
    xyz: np.ndarray,          # (N,3) float32
    rgb_u8: np.ndarray,       # (N,3) uint8
    opacity: np.ndarray,      # (N,) float32 in [0,1]
    scale: np.ndarray,        # (N,3) float32 (linear scale)
    quat_wxyz: np.ndarray,    # (N,4) float32
):
    xyz = xyz.astype(np.float32)
    scale = scale.astype(np.float32)
    quat_wxyz = quat_wxyz.astype(np.float32)
    opacity = opacity.reshape(-1).astype(np.float32)
    rgb_u8 = rgb_u8.astype(np.uint8)

    N = xyz.shape[0]
    assert xyz.shape == (N, 3)
    assert scale.shape == (N, 3)
    assert quat_wxyz.shape == (N, 4)
    assert opacity.shape == (N,)
    assert rgb_u8.shape == (N, 3)

    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n"
    ]).encode("ascii")

    dtype = np.dtype([
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])
    data = np.empty(N, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
    data["opacity"] = opacity
    data["scale_0"], data["scale_1"], data["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = (
        quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    )

    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes(order="C"))

def export_gaussians_to_ply(gaussians, ply_path: str):
    """
    Export AnySplat gaussians to a Supersplat-friendly Gaussian PLY.

    This tries:
      1) built-in export if exists
      2) manual export using common field names

    If manual export fails, it prints the available keys/attrs to help mapping.
    """
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    # 1) Built-in export (best)
    if try_export_ply_builtin(gaussians, ply_path):
        print(f"[INFO] Exported gaussians via built-in method: {ply_path}")
        return

    # 2) Manual export: try common structures
    # gaussians may be dict-like or object with attributes
    def get_field(names):
        if isinstance(gaussians, dict):
            for n in names:
                if n in gaussians:
                    return gaussians[n]
        else:
            for n in names:
                if hasattr(gaussians, n):
                    return getattr(gaussians, n)
        return None

    # Common candidates across 3DGS implementations
    xyz = get_field(["xyz", "means", "means3D", "mu", "position", "positions"])
    scale = get_field(["scales", "scale", "log_scales", "log_scale"])
    quat = get_field(["quats", "quat", "rotations", "rotation", "rots", "rot"])
    opacity = get_field(["opacities", "opacity", "alpha", "alphas", "opacity_logit", "opacity_logits"])
    rgb = get_field(["rgb", "colors", "color", "sh_dc", "features_dc", "f_dc"])  # may be SH DC

    # If something is missing, dump info
    if xyz is None or scale is None or quat is None or opacity is None or rgb is None:
        print("[ERROR] Cannot find required fields for manual PLY export.")
        print("        Need xyz/scale/quat/opacity/rgb (or SH DC).")
        if isinstance(gaussians, dict):
            print("        gaussians keys:", list(gaussians.keys()))
        else:
            attrs = [a for a in dir(gaussians) if not a.startswith("_")]
            cand = [a for a in attrs if any(k in a.lower() for k in ["xyz","mean","scale","rot","quat","opac","alpha","rgb","color","sh","feat","dc"])]
            print("        gaussians candidate attrs:", cand[:80])
        raise RuntimeError("Manual PLY export mapping failed. Inspect gaussians and adjust field names.")

    xyz = _to_np(xyz).reshape(-1, 3).astype(np.float32)

    # scale: if it's log-scale, exp it (very common)
    scale_np = _to_np(scale)
    scale_np = scale_np.reshape(-1, 3).astype(np.float32)
    # heuristic: if values look like log-scale (often negative), exp them
    if np.median(scale_np) < 0.0:
        scale_np = np.exp(scale_np).astype(np.float32)

    quat_np = _to_np(quat).reshape(-1, 4).astype(np.float32)

    # opacity: if it's logits, sigmoid it
    opacity_np = _to_np(opacity).reshape(-1).astype(np.float32)
    # heuristic: logits often not in [0,1]
    if opacity_np.min() < -0.01 or opacity_np.max() > 1.01:
        opacity_np = 1.0 / (1.0 + np.exp(-opacity_np))

    # rgb: handle either [0,1] float rgb OR SH DC-like
    rgb_np = _to_np(rgb)
    if rgb_np.ndim == 2 and rgb_np.shape[1] >= 3:
        rgb3 = rgb_np[:, :3]
    elif rgb_np.ndim == 3:
        # sometimes (N,1,3)
        rgb3 = rgb_np.reshape(-1, rgb_np.shape[-1])[:, :3]
    else:
        raise RuntimeError(f"Unexpected rgb field shape: {rgb_np.shape}")

    rgb3 = rgb3.astype(np.float32)
    # clamp/scale to uint8
    if rgb3.max() <= 1.5:
        rgb_u8 = np.clip(rgb3 * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = np.clip(rgb3, 0, 255).astype(np.uint8)

    write_gaussian_ply_binary(
        ply_path,
        xyz=xyz,
        rgb_u8=rgb_u8,
        opacity=opacity_np,
        scale=scale_np,
        quat_wxyz=quat_np,
    )
    print(f"[INFO] Exported gaussians via manual mapping: {ply_path}")

################################################################################################################


# -----------------------------
# VRAM / Model size utilities
# -----------------------------
def sizeof_tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def model_size_report(model: torch.nn.Module, *, topk_modules: int = 20) -> None:
    """Report model parameter/buffer sizes in MiB, plus top-k largest modules."""
    param_bytes = sum(sizeof_tensor_bytes(p) for p in model.parameters())
    buffer_bytes = sum(sizeof_tensor_bytes(b) for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes

    print("\n=== Model size (parameters/buffers) ===")
    print(f"params : {param_bytes/1024/1024:.2f} MiB")
    print(f"buffers: {buffer_bytes/1024/1024:.2f} MiB")
    print(f"total  : {total_bytes/1024/1024:.2f} MiB")

    module_sizes = []
    for name, m in model.named_modules():
        pb = sum(sizeof_tensor_bytes(p) for p in m.parameters(recurse=False))
        bb = sum(sizeof_tensor_bytes(b) for b in m.buffers(recurse=False))
        if pb + bb > 0:
            module_sizes.append((name, pb + bb, pb, bb))

    module_sizes.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== Top {topk_modules} modules by (param+buffer) size ===")
    for name, tb, pb, bb in module_sizes[:topk_modules]:
        print(
            f"{name:55s} total={tb/1024/1024:8.2f} MiB  "
            f"(p={pb/1024/1024:8.2f}, b={bb/1024/1024:8.2f})"
        )


def torch_cuda_mem_report(device: Optional[torch.device] = None, prefix: str = "") -> None:
    """Report PyTorch CUDA memory for THIS process (allocated/reserved and peaks)."""
    if not torch.cuda.is_available():
        print(f"{prefix}[torch] CUDA not available.")
        return

    if device is None:
        device = torch.device("cuda")

    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated(device)
    reserv = torch.cuda.memory_reserved(device)
    max_alloc = torch.cuda.max_memory_allocated(device)
    max_reserv = torch.cuda.max_memory_reserved(device)

    print(f"\n=== {prefix}Torch CUDA memory ({device}) ===")
    print(f"allocated : {alloc/1024/1024:.2f} MiB")
    print(f"reserved  : {reserv/1024/1024:.2f} MiB")
    print(f"max_alloc : {max_alloc/1024/1024:.2f} MiB")
    print(f"max_reserv: {max_reserv/1024/1024:.2f} MiB")


def _nvml_list_processes(gpu_index: int = 0) -> List[Tuple[int, int, str]]:
    """
    Return list of (pid, used_mem_mib, proc_name) for the given GPU.
    If not available due to permissions/driver, return [].
    """
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        procs: List[Tuple[int, int, str]] = []

        # compute processes
        try:
            cps = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(h)
        except Exception:
            cps = pynvml.nvmlDeviceGetComputeRunningProcesses(h)

        for p in cps:
            pid = int(p.pid)
            used = int(p.usedGpuMemory)  # bytes (sometimes -1)
            used_mib = used // (1024 * 1024) if used and used > 0 else -1
            name = "?"
            try:
                name = pynvml.nvmlSystemGetProcessName(pid).decode("utf-8", errors="ignore")
            except Exception:
                pass
            procs.append((pid, used_mib, name))

        # graphics processes (optional)
        try:
            try:
                gps = pynvml.nvmlDeviceGetGraphicsRunningProcesses_v2(h)
            except Exception:
                gps = pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
            for p in gps:
                pid = int(p.pid)
                used = int(p.usedGpuMemory)
                used_mib = used // (1024 * 1024) if used and used > 0 else -1
                name = "?"
                try:
                    name = pynvml.nvmlSystemGetProcessName(pid).decode("utf-8", errors="ignore")
                except Exception:
                    pass
                procs.append((pid, used_mib, name))
        except Exception:
            pass

        procs.sort(key=lambda x: x[1], reverse=True)
        return procs
    except Exception:
        return []


def print_gpu_process_vram(gpu_index: int = 0, highlight_pid: Optional[int] = None, limit: int = 30) -> None:
    """Print per-process VRAM on GPU (NVML)."""
    procs = _nvml_list_processes(gpu_index)
    if not procs:
        print(f"\n[NVML] Cannot query per-process VRAM on GPU {gpu_index} (permission/driver limitation, or pynvml missing).")
        print("       If you want this, install pynvml: pip install pynvml")
        return

    print(f"\n=== GPU {gpu_index} VRAM by process (NVML) ===")
    for i, (pid, used_mib, name) in enumerate(procs[:limit]):
        mark = " <== THIS PROCESS" if (highlight_pid is not None and pid == highlight_pid) else ""
        print(f"{i:02d} pid={pid:<7} mem={used_mib:>6} MiB  name={name}{mark}")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def maybe_write_log_line(log_path: Optional[str], line: str) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def snapshot_all(
    *,
    stage: str,
    gpu_index: int,
    device: torch.device,
    log_path: Optional[str],
    model: Optional[torch.nn.Module] = None,
    topk_modules: int = 20,
) -> None:
    """One call: print (and optionally log) NVML proc VRAM + torch mem + model sizes."""
    pid = os.getpid()

    header = f"\n[{now_ts()}] ===== SNAPSHOT: {stage} ===== pid={pid} gpu={gpu_index}"
    print(header)
    maybe_write_log_line(log_path, header)

    # NVML per-process table
    procs = _nvml_list_processes(gpu_index)
    if procs:
        lines = [f"=== GPU {gpu_index} VRAM by process (NVML) ==="]
        for i, (p, used_mib, name) in enumerate(procs[:30]):
            mark = " <== THIS PROCESS" if p == pid else ""
            lines.append(f"{i:02d} pid={p:<7} mem={used_mib:>6} MiB  name={name}{mark}")
        txt = "\n".join(lines)
        print(txt)
        maybe_write_log_line(log_path, txt)
    else:
        warn = f"[NVML] per-process VRAM not available (install pynvml or permission limitation)."
        print(warn)
        maybe_write_log_line(log_path, warn)

    # Torch mem for this process
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserv = torch.cuda.memory_reserved(device) / 1024 / 1024
        max_alloc = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        max_reserv = torch.cuda.max_memory_reserved(device) / 1024 / 1024
        txt = (
            f"=== Torch CUDA memory ({device}) ===\n"
            f"allocated : {alloc:.2f} MiB\n"
            f"reserved  : {reserv:.2f} MiB\n"
            f"max_alloc : {max_alloc:.2f} MiB\n"
            f"max_reserv: {max_reserv:.2f} MiB"
        )
        print(txt)
        maybe_write_log_line(log_path, txt)
    else:
        txt = "[torch] CUDA not available."
        print(txt)
        maybe_write_log_line(log_path, txt)

    # Model sizes (optional)
    if model is not None:
        # quick totals
        param_bytes = sum(sizeof_tensor_bytes(p) for p in model.parameters())
        buffer_bytes = sum(sizeof_tensor_bytes(b) for b in model.buffers())
        total_bytes = param_bytes + buffer_bytes
        txt = (
            "=== Model size (parameters/buffers) ===\n"
            f"params : {param_bytes/1024/1024:.2f} MiB\n"
            f"buffers: {buffer_bytes/1024/1024:.2f} MiB\n"
            f"total  : {total_bytes/1024/1024:.2f} MiB"
        )
        print(txt)
        maybe_write_log_line(log_path, txt)

        # top-k modules
        module_sizes = []
        for name, m in model.named_modules():
            pb = sum(sizeof_tensor_bytes(p) for p in m.parameters(recurse=False))
            bb = sum(sizeof_tensor_bytes(b) for b in m.buffers(recurse=False))
            if pb + bb > 0:
                module_sizes.append((name, pb + bb, pb, bb))
        module_sizes.sort(key=lambda x: x[1], reverse=True)

        lines = [f"=== Top {topk_modules} modules by (param+buffer) size ==="]
        for name, tb, pb, bb in module_sizes[:topk_modules]:
            lines.append(
                f"{name:55s} total={tb/1024/1024:8.2f} MiB  "
                f"(p={pb/1024/1024:8.2f}, b={bb/1024/1024:8.2f})"
            )
        txt = "\n".join(lines)
        print(txt)
        maybe_write_log_line(log_path, txt)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="anysplat Demo + VRAM monitor")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--output_dir", type=str, help="Directory of output results")

    # monitoring options
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index for NVML process table")
    parser.add_argument("--print_vram", action="store_true", help="Print VRAM snapshots at key stages")
    parser.add_argument("--vram_log", type=str, default=None, help="Optional log file to append VRAM snapshots")

    args = parser.parse_args()

    if args.output_dir is None:
        save_path = os.path.join(args.scene_dir, "anysplat_output")
    else:
        save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.print_vram:
        snapshot_all(
            stage="start",
            gpu_index=args.gpu_index,
            device=device,
            log_path=args.vram_log,
            model=None,
        )

    # Load the model from Hugging Face
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if args.print_vram:
        snapshot_all(
            stage="after model.to(device)",
            gpu_index=args.gpu_index,
            device=device,
            log_path=args.vram_log,
            model=model,
            topk_modules=20,
        )

    image_dir = Path(os.path.join(args.scene_dir, "images"))
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    image_names = sorted(
        [str(p) for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )
    if len(image_names) == 0:
        raise RuntimeError(f"No images found in: {image_dir}")

    # Load and preprocess images
    images = [process_image(image_name) for image_name in image_names]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]
    b, v, c, h, w = images.shape

    assert c == 3, "Images must have 3 channels" # from demo_gradio.py

    if args.print_vram:
        snapshot_all(
            stage="after images.to(device)",
            gpu_index=args.gpu_index,
            device=device,
            log_path=args.vram_log,
            model=None,
        )

    # Run Inference
    with torch.no_grad():
        gaussians, pred_context_pose = model.inference((images + 1) * 0.5)

    pred_all_extrinsic = pred_context_pose["extrinsic"]
    pred_all_intrinsic = pred_context_pose["intrinsic"]

    if args.print_vram:
        snapshot_all(
            stage="after inference",
            gpu_index=args.gpu_index,
            device=device,
            log_path=args.vram_log,
            model=None,
        )

    # Save outputs
    save_interpolated_video(
        pred_all_extrinsic,
        pred_all_intrinsic,
        b, h, w,
        gaussians,
        save_path,
        model.decoder,
    )

    plyfile = os.path.join(save_path, "gaussians.ply")
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        Path(plyfile),
        save_sh_dc_only=True,
    )

    if args.print_vram:
        snapshot_all(
            stage="after save_interpolated_video",
            gpu_index=args.gpu_index,
            device=device,
            log_path=args.vram_log,
            model=None,
        )

    print(f"\n[INFO] Done. Output dir: {save_path}")
    if args.vram_log:
        print(f"[INFO] VRAM snapshots appended to: {args.vram_log}")


if __name__ == "__main__":
    main()
