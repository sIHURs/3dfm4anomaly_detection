import argparse
from pathlib import Path
import torch
import os
import sys
import time
from typing import List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factory.anysplat.misc.image_io import save_interpolated_video
from factory.anysplat.model.model.anysplat import AnySplat
from factory.anysplat.utils.image import process_image


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
    b, v, _, h, w = images.shape

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
