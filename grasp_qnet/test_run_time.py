import time
import math
import torch
import torch.nn as nn
from models.dgcnn import DGCNNGraspQNet


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    obj_points=2048,
    obj_channels=4,      # xyz+feat(=4) 또는 xyz+normal(=6) 등
    grip_points=64,
    batch_sizes=(1, 8, 16, 32),
    warmup=20,
    iters=100,
    use_mc_dropout=False, # 모델에 forward_mc_dropout이 있으면 True로 측정 가능
    mc_samples=10,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    torch.backends.cudnn.benchmark = True

    results = {}
    for B in batch_sizes:
        # 더미 입력 생성
        obj_cloud = torch.randn(B, obj_points, obj_channels, device=device)
        gripper_cloud = torch.randn(B, grip_points, 3, device=device)

        # warm-up
        for _ in range(warmup):
            if use_mc_dropout and hasattr(model, "forward_mc_dropout"):
                _ = model.forward_mc_dropout(obj_cloud, gripper_cloud, N=mc_samples)
            else:
                _ = model(obj_cloud, gripper_cloud)
            if device == "cuda":
                torch.cuda.synchronize()

        # 측정
        times = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_mc_dropout and hasattr(model, "forward_mc_dropout"):
                _ = model.forward_mc_dropout(obj_cloud, gripper_cloud, N=mc_samples)
            else:
                _ = model(obj_cloud, gripper_cloud)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = sum(times) / len(times)
        std = (sum((t-avg)**2 for t in times) / (len(times)-1)) ** 0.5 if len(times) > 1 else 0.0
        p50 = sorted(times)[len(times)//2]
        p90 = sorted(times)[int(math.ceil(0.9*len(times)))-1]
        p95 = sorted(times)[int(math.ceil(0.95*len(times)))-1]
        throughput = B / avg  # samples/sec (1 forward 기준; MC-dropout은 N회 포함 시간)

        results[B] = {
            "avg_sec": avg,
            "std_sec": std,
            "p50_sec": p50,
            "p90_sec": p90,
            "p95_sec": p95,
            "throughput_samples_per_sec": throughput,
        }

    return results

if __name__ == "__main__":
    # 모델 생성 (필요 시 하이퍼 변경)
    model = DGCNNGraspQNet(
        num_classes=1, 
    )

    total, trainable = count_parameters(model)
    print(f"[Params] total: {total:,}  trainable: {trainable:,}")

    # 기본 설정으로 측정 (GPU 있으면 자동 사용)
    res = benchmark_inference(
        model,
        obj_points=1024,
        obj_channels=3,        # xyz+feat(=4) 또는 6으로 변경 가능
        grip_points=64,
        batch_sizes=(1, 8, 16, 32),
        warmup=20,
        iters=100,
        use_mc_dropout=False,  # MC Dropout 추론 시간까지 포함해 보고 싶으면 True
        mc_samples=10,
    )

    print("\n[Inference Time]")
    for B, r in res.items():
        print(f"Batch {B:>3}: "
              f"avg {r['avg_sec']*1000:.3f} ms  "
              f"std {r['std_sec']*1000:.3f} ms  "
              f"p50 {r['p50_sec']*1000:.3f} ms  "
              f"p90 {r['p90_sec']*1000:.3f} ms  "
              f"p95 {r['p95_sec']*1000:.3f} ms  "
              f"throughput {r['throughput_samples_per_sec']:.1f} samples/s")
