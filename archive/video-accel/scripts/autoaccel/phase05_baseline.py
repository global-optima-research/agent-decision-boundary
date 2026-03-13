#!/usr/bin/env python3
"""
AutoAccel Phase 0.5: Single-Technique Baselines + Pairwise Combination
=======================================================================
Benchmarks SageAttention2 (quant) and First Block Cache (cache) individually
and in combination on Wan 2.1-1.3B. Measures speed and quality.

Configurations:
  B0: Baseline (no acceleration)
  B1: SageAttention2 only (quantized attention kernel)
  B2: First Block Cache only (diffusers built-in TeaCache equivalent)
  B3: SageAttention2 + First Block Cache

Usage:
  python scripts/autoaccel/phase05_baseline.py --output_dir results/autoaccel_phase05
  python scripts/autoaccel/phase05_baseline.py --num_prompts 3 --configs B0,B1
  python scripts/autoaccel/phase05_baseline.py --fbc_thresholds 0.05,0.1,0.2
"""

import torch
import numpy as np
import json
import argparse
import gc
import time
import sys
from pathlib import Path
from datetime import datetime

# Reuse Phase 1 utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase1_utils import (
    PROMPTS, load_wan_pipeline, compute_ssim, compute_psnr,
    save_video_frames, get_env_info, json_convert,
)


# ─── SageAttention2 Integration ─────────────────────────────────────────────

def enable_sage_attention(pipe):
    """
    Enable SageAttention2 on the transformer (NOT VAE).
    VAE has head_dim=384 which SageAttention doesn't support.
    """
    from sageattention import sageattn

    original_fn = torch.nn.functional.scaled_dot_product_attention

    def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
        # SageAttention only supports specific head dims
        head_dim = query.shape[-1]
        if head_dim not in (32, 48, 64, 96, 128, 160, 192, 224, 256):
            return original_fn(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal,
                             scale=scale)
        # SageAttention doesn't support attention masks well
        if attn_mask is not None:
            return original_fn(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal,
                             scale=scale)
        try:
            return sageattn(query, key, value, is_causal=is_causal, scale=scale)
        except Exception:
            return original_fn(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal,
                             scale=scale)

    # Monkey-patch only during transformer forward
    torch.nn.functional.scaled_dot_product_attention = sage_sdpa
    return original_fn


def disable_sage_attention(original_fn):
    """Restore original scaled_dot_product_attention."""
    torch.nn.functional.scaled_dot_product_attention = original_fn


# ─── First Block Cache (TeaCache equivalent) ────────────────────────────────

def enable_first_block_cache(pipe, threshold=0.1):
    """Enable First Block Cache on transformer."""
    from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
    config = FirstBlockCacheConfig(threshold=threshold)
    apply_first_block_cache(pipe.transformer, config)
    return config


def disable_first_block_cache(pipe):
    """Disable First Block Cache by removing hooks from all blocks."""
    _ALL_BLOCK_NAMES = ["blocks", "transformer_blocks", "single_transformer_blocks",
                        "temporal_transformer_blocks"]
    transformer = pipe.transformer
    for attr_name in _ALL_BLOCK_NAMES:
        blocks = getattr(transformer, attr_name, None)
        if blocks is None or not isinstance(blocks, torch.nn.ModuleList):
            continue
        for block in blocks:
            if hasattr(block, '_diffusers_hook'):
                hook_registry = block._diffusers_hook
                for name in list(hook_registry.hooks.keys()):
                    hook_registry.remove_hook(name)
                del block._diffusers_hook


# ─── Video Generation ────────────────────────────────────────────────────────

def generate_video_simple(pipe, prompt, height=480, width=832,
                          num_frames=17, num_steps=50, seed=42, device="cuda:0"):
    """Generate video using standard WanPipeline (no custom processors)."""
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            generator=generator,
            output_type="np",
        )

    frames = output.frames[0]
    frames_tensor = torch.from_numpy(np.stack(frames))  # (F, H, W, 3)
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)   # (F, 3, H, W)
    return frames_tensor


# ─── Benchmark Runner ────────────────────────────────────────────────────────

def run_config(pipe, config_name, prompts, args, fbc_threshold=0.1):
    """
    Run a single configuration and return results.

    Returns dict with per-prompt times, frames, and aggregate stats.
    """
    device = args.device
    original_sdpa = None

    # Setup
    if 'B1' in config_name or 'B3' in config_name:
        # Enable SageAttention
        original_sdpa = enable_sage_attention(pipe)
        print(f"  SageAttention2: enabled")

    if 'B2' in config_name or 'B3' in config_name:
        # Enable First Block Cache
        enable_first_block_cache(pipe, threshold=fbc_threshold)
        print(f"  First Block Cache: enabled (threshold={fbc_threshold})")

    results = {
        'times': [],
        'frames': [],  # Will store for quality comparison, then remove
    }

    # Warmup run (first prompt might be slow due to compilation)
    if args.warmup:
        print(f"  Warmup...", end="", flush=True)
        _ = generate_video_simple(
            pipe, "warmup test", height=args.height, width=args.width,
            num_frames=args.num_frames, num_steps=args.num_steps,
            seed=args.seed, device=device,
        )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        print(" done")

    for pi, prompt in enumerate(prompts):
        print(f"  [{pi+1}/{len(prompts)}] \"{prompt[:50]}\"", end="", flush=True)

        torch.cuda.synchronize()
        t0 = time.time()

        frames = generate_video_simple(
            pipe, prompt, height=args.height, width=args.width,
            num_frames=args.num_frames, num_steps=args.num_steps,
            seed=args.seed, device=device,
        )

        torch.cuda.synchronize()
        gen_time = time.time() - t0

        results['times'].append(gen_time)
        results['frames'].append(frames)
        print(f"  {gen_time:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()

    # Teardown
    if 'B2' in config_name or 'B3' in config_name:
        disable_first_block_cache(pipe)

    if original_sdpa is not None:
        disable_sage_attention(original_sdpa)

    # Aggregate timing
    results['time_mean'] = float(np.mean(results['times']))
    results['time_std'] = float(np.std(results['times']))

    return results


def compute_quality_metrics(results, baseline_frames):
    """Compute SSIM and PSNR against baseline for each prompt."""
    ssims = []
    psnrs = []
    for pi in range(len(baseline_frames)):
        ssim = compute_ssim(results['frames'][pi], baseline_frames[pi])
        psnr = compute_psnr(results['frames'][pi], baseline_frames[pi])
        ssims.append(ssim)
        psnrs.append(psnr)

    results['ssim'] = ssims
    results['psnr'] = psnrs
    results['ssim_mean'] = float(np.mean(ssims))
    results['ssim_std'] = float(np.std(ssims))
    results['psnr_mean'] = float(np.mean(psnrs))
    results['psnr_std'] = float(np.std(psnrs))


# ─── Main Experiment ─────────────────────────────────────────────────────────

CONFIG_DESCRIPTIONS = {
    'B0': 'Baseline (no acceleration)',
    'B1': 'SageAttention2 (quantized attention)',
    'B2': 'First Block Cache (step-level caching)',
    'B3': 'SageAttention2 + First Block Cache',
}


def run_experiment(args):
    print("=" * 60)
    print("AutoAccel Phase 0.5: Single-Tech Baselines + Combination")
    print("=" * 60)

    device = args.device
    env_info = get_env_info(device)
    print(f"Environment: {json.dumps(env_info, indent=2)}")

    configs = args.configs.split(',')
    if 'B0' not in configs:
        configs.insert(0, 'B0')
    fbc_thresholds = [float(x) for x in args.fbc_thresholds.split(',')]

    print(f"\nConfigs: {configs}")
    print(f"FBC thresholds: {fbc_thresholds}")

    prompts = PROMPTS[:args.num_prompts]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.model}")
    pipe = load_wan_pipeline(args.model, device)

    # Check SageAttention availability
    sage_available = True
    try:
        from sageattention import sageattn
        print("✅ SageAttention2 available")
    except ImportError:
        sage_available = False
        print("⚠️ SageAttention2 not installed, skipping B1/B3")
        configs = [c for c in configs if c not in ('B1', 'B3')]

    # Check First Block Cache availability
    fbc_available = True
    try:
        from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
        print("✅ First Block Cache available")
    except ImportError:
        fbc_available = False
        print("⚠️ First Block Cache not available, skipping B2/B3")
        configs = [c for c in configs if c not in ('B2', 'B3')]

    all_results = {}
    start_time = time.time()

    # ── Run each configuration ──
    for config in configs:
        # For B2/B3, test multiple FBC thresholds
        if config in ('B2', 'B3'):
            for thresh in fbc_thresholds:
                config_key = f"{config}_t{thresh}"
                desc = f"{CONFIG_DESCRIPTIONS[config]} (threshold={thresh})"
                print(f"\n{'─' * 50}")
                print(f"Config: {config_key} — {desc}")
                print(f"{'─' * 50}")

                results = run_config(pipe, config, prompts, args,
                                     fbc_threshold=thresh)
                results['description'] = desc
                results['config'] = config
                results['fbc_threshold'] = thresh
                all_results[config_key] = results
        else:
            print(f"\n{'─' * 50}")
            print(f"Config: {config} — {CONFIG_DESCRIPTIONS[config]}")
            print(f"{'─' * 50}")

            results = run_config(pipe, config, prompts, args)
            results['description'] = CONFIG_DESCRIPTIONS[config]
            results['config'] = config
            all_results[config] = results

    total_time = time.time() - start_time

    # ── Compute quality metrics ──
    baseline_frames = all_results['B0']['frames']
    for key, results in all_results.items():
        if key == 'B0':
            results['ssim'] = [1.0] * len(prompts)
            results['psnr'] = [100.0] * len(prompts)
            results['ssim_mean'] = 1.0
            results['ssim_std'] = 0.0
            results['psnr_mean'] = 100.0
            results['psnr_std'] = 0.0
        else:
            compute_quality_metrics(results, baseline_frames)

    # ── Compute speedups ──
    baseline_time = all_results['B0']['time_mean']
    for key, results in all_results.items():
        results['speedup'] = baseline_time / results['time_mean']

    # ── Print summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Config':<20} | {'Time (s)':>10} | {'Speedup':>8} | "
          f"{'SSIM':>10} | {'PSNR':>8}")
    print("-" * 68)

    for key in all_results:
        r = all_results[key]
        print(f"{key:<20} | {r['time_mean']:7.1f}±{r['time_std']:.1f} | "
              f"{r['speedup']:7.2f}x | "
              f"{r['ssim_mean']:.4f}±{r['ssim_std']:.3f} | "
              f"{r['psnr_mean']:5.1f}dB")

    # ── Go/No-Go for AutoAccel ──
    print(f"\n{'=' * 70}")
    print("GO/NO-GO ASSESSMENT")
    print(f"{'=' * 70}")

    # Check if combination is better than individuals
    combo_keys = [k for k in all_results if k.startswith('B3')]
    if combo_keys:
        best_combo = max(combo_keys, key=lambda k: all_results[k]['speedup']
                         if all_results[k]['ssim_mean'] > 0.90 else 0)
        best_combo_r = all_results[best_combo]

        sage_r = all_results.get('B1', {})
        sage_speedup = sage_r.get('speedup', 1.0)

        best_fbc = max(
            [k for k in all_results if k.startswith('B2')],
            key=lambda k: all_results[k]['speedup']
            if all_results[k]['ssim_mean'] > 0.90 else 0,
            default=None
        )
        fbc_speedup = all_results[best_fbc]['speedup'] if best_fbc else 1.0

        expected_independent = sage_speedup * fbc_speedup
        actual_combo = best_combo_r['speedup']
        interaction = actual_combo / expected_independent if expected_independent > 0 else 0

        print(f"\nSageAttention2 speedup:        {sage_speedup:.2f}x")
        print(f"Best FBC speedup:              {fbc_speedup:.2f}x (SSIM>0.90)")
        print(f"Expected independent combo:    {expected_independent:.2f}x")
        print(f"Actual combo ({best_combo}):  {actual_combo:.2f}x")
        print(f"Interaction ratio:             {interaction:.2f}")
        print(f"Combo SSIM:                    {best_combo_r['ssim_mean']:.4f}")

        if interaction > 0.9:
            print("\n✅ NEAR-ORTHOGONAL: techniques stack well")
        elif interaction > 0.7:
            print("\n⚠️ PARTIAL INTERFERENCE: some speedup lost in combination")
        else:
            print("\n❌ SIGNIFICANT INTERFERENCE: combination much worse than expected")

        if best_combo_r['ssim_mean'] > 0.90 and actual_combo > 1.5:
            print(f"\n✅ GO: Combination achieves {actual_combo:.2f}x speedup with SSIM {best_combo_r['ssim_mean']:.4f}")
            print("   → AutoAccel搜索有意义，继续 Phase 1")
        elif best_combo_r['ssim_mean'] > 0.85:
            print(f"\n⚠️ CONDITIONAL GO: Speedup {actual_combo:.2f}x, SSIM {best_combo_r['ssim_mean']:.4f}")
            print("   → 需要调参优化")
        else:
            print(f"\n❌ NO GO: SSIM too low ({best_combo_r['ssim_mean']:.4f})")
    else:
        print("\nNo combination configs tested (B3 not available)")

    # ── Save results ──
    # Remove frames from results before saving (too large for JSON)
    save_results = {}
    for key, r in all_results.items():
        save_r = {k: v for k, v in r.items() if k != 'frames'}
        save_results[key] = save_r

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_data = {
        'config': {
            'model': args.model,
            'height': args.height,
            'width': args.width,
            'num_frames': args.num_frames,
            'num_steps': args.num_steps,
            'num_prompts': args.num_prompts,
            'seed': args.seed,
            'configs_tested': configs,
            'fbc_thresholds': fbc_thresholds,
            'prompts': prompts,
            'warmup': args.warmup,
        },
        'env': env_info,
        'total_time_seconds': round(total_time, 1),
        'results': save_results,
    }

    results_file = output_dir / f"phase05_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(save_data, f, default=json_convert, indent=2)

    report_file = output_dir / f"phase05_report_{timestamp}.md"
    generate_report(save_data, report_file)

    print(f"\nResults: {results_file}")
    print(f"Report:  {report_file}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Save sample frames
    if args.save_frames:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for key, r in all_results.items():
            if r['frames']:
                save_video_frames(r['frames'][0], frames_dir / key, prefix=key)
                print(f"Saved frames: {frames_dir / key}")


def generate_report(data, report_file):
    """Generate markdown experiment report."""
    env = data['env']
    cfg = data['config']
    results = data['results']

    lines = [
        "# 实验：AutoAccel Phase 0.5 单技术基线 + 组合测试",
        f"> 日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 环境",
        f"- Commit: `{env.get('git_commit', 'unknown')}`",
        f"- GPU: {env.get('gpu_name', 'unknown')}",
        f"- Python: {env.get('python_version', 'unknown')}",
        f"- PyTorch: {env.get('torch_version', 'unknown')}",
        f"- CUDA: {env.get('cuda_version', 'unknown')}",
        f"- Diffusers: {env.get('diffusers_version', 'unknown')}",
        "",
        "## 运行命令",
        "```bash",
        f"python scripts/autoaccel/phase05_baseline.py \\",
        f"  --num_prompts {cfg['num_prompts']} \\",
        f"  --configs {','.join(cfg['configs_tested'])} \\",
        f"  --fbc_thresholds {','.join(str(t) for t in cfg['fbc_thresholds'])} \\",
        f"  --output_dir results/autoaccel_phase05",
        "```",
        "",
        "## 参数",
        f"| 参数 | 值 |",
        f"|------|-----|",
        f"| 模型 | {cfg['model']} |",
        f"| 分辨率 | {cfg['height']}×{cfg['width']} |",
        f"| 帧数 | {cfg['num_frames']} |",
        f"| 步数 | {cfg['num_steps']} |",
        f"| Prompt 数 | {cfg['num_prompts']} |",
        f"| 种子 | {cfg['seed']} |",
        f"| Warmup | {cfg['warmup']} |",
        "",
        "## 配置说明",
        "",
        "| 配置 | 描述 |",
        "|------|------|",
    ]

    for key in results:
        lines.append(f"| {key} | {results[key].get('description', '')} |")

    lines.extend([
        "",
        "## 结果",
        "",
        "| 配置 | 耗时 (s) | 加速比 | SSIM | PSNR (dB) |",
        "|------|---------|--------|------|-----------|",
    ])

    for key, r in results.items():
        lines.append(
            f"| {key} | {r['time_mean']:.1f}±{r['time_std']:.1f} | "
            f"{r['speedup']:.2f}x | "
            f"{r['ssim_mean']:.4f}±{r['ssim_std']:.3f} | "
            f"{r['psnr_mean']:.1f}±{r['psnr_std']:.1f} |"
        )

    # Per-prompt SSIM details
    lines.extend([
        "",
        "## 逐 Prompt SSIM",
        "",
    ])
    header = "| Prompt |"
    sep = "|--------|"
    for key in results:
        header += f" {key} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for pi in range(cfg['num_prompts']):
        row = f"| {cfg['prompts'][pi][:35]}... |"
        for key in results:
            row += f" {results[key]['ssim'][pi]:.4f} |"
        lines.append(row)

    lines.extend([
        "",
        "## 分析与结论",
        "",
        "（运行后填写）",
        "",
        f"## 总耗时",
        f"{data['total_time_seconds']:.1f}s ({data['total_time_seconds']/60:.1f}min)",
    ])

    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoAccel Phase 0.5: Baselines + Combination")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_prompts", type=int, default=5,
                        help="Number of prompts (max 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", default="B0,B1,B2,B3",
                        help="Configs to test (B0=baseline, B1=sage, B2=fbc, B3=sage+fbc)")
    parser.add_argument("--fbc_thresholds", default="0.05,0.1,0.2",
                        help="Comma-separated First Block Cache thresholds to test")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a warmup generation before timing")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save video frames as PNGs")
    parser.add_argument("--output_dir", default="results/autoaccel_phase05")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_experiment(args)
