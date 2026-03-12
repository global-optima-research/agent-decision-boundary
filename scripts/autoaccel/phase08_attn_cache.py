#!/usr/bin/env python3
"""
AutoAccel Phase 0.8: Attention Output Caching (PAB-style)
=========================================================
Instead of dropping cross-frame tokens (frame-local sparse, which failed),
cache and reuse attention outputs at configurable per-layer intervals.

Key idea:
  - Compute full attention every N steps for each layer
  - Reuse cached attention output on intermediate steps
  - Different layers can have different intervals (pyramid structure)
  - This preserves ALL cross-frame information when attention IS computed

Difference from FBC:
  - FBC: caches entire block output, skips FFN+norms too
  - Attn cache: only caches attention output, still runs FFN+norms
  - They work at different granularities → potentially non-trivial interaction

Configurations:
  P0: Baseline (full attention every step)
  P1-P5: Uniform broadcast interval (all layers same N)
  P6-P8: Pyramid (different intervals per layer depth)
  P9-P12: Best pyramid + FBC combos (interaction test)

Usage:
  python scripts/autoaccel/phase08_attn_cache.py --output_dir results/autoaccel_phase08
  python scripts/autoaccel/phase08_attn_cache.py --num_prompts 3 --configs P0,P1,P2
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import gc
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase1_utils import (
    PROMPTS, compute_ssim, compute_psnr,
    save_video_frames, get_env_info, json_convert,
)


# ─── Step Tracker ────────────────────────────────────────────────────────────

class StepTracker:
    """Shared mutable state for tracking current denoising step."""
    def __init__(self):
        self.step = 0

    def reset(self):
        self.step = 0

    def callback(self, pipe, step_idx, timestep, cb_kwargs):
        self.step = step_idx
        return cb_kwargs


# ─── Attention-Caching Processor ─────────────────────────────────────────────

class CachingWanAttnProcessor:
    """
    Wan attention processor with attention output caching.

    Instead of computing attention every step, caches the attention output
    and reuses it for `interval - 1` steps before recomputing.

    interval=1: compute every step (baseline)
    interval=2: compute every other step, reuse in between
    interval=4: compute every 4th step
    """

    def __init__(self, layer_idx, attn_type='self', interval=1, step_tracker=None):
        self.layer_idx = layer_idx
        self.attn_type = attn_type
        self.interval = interval  # 1 = always compute
        self.step_tracker = step_tracker
        self._cached_output = None
        self._cached_img_output = None

    def _should_compute(self):
        """Determine if we should compute attention or use cache."""
        if self.interval <= 1 or self.step_tracker is None:
            return True
        return self.step_tracker.step % self.interval == 0

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        from diffusers.models.transformers.transformer_wan import (
            _get_qkv_projections,
            dispatch_attention_fn,
        )

        # ── Handle image context (I2V) ──
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if self.attn_type == 'self' and not self._should_compute() and self._cached_output is not None:
            # ── Use cached attention output ──
            hidden_states = self._cached_output
            if self._cached_img_output is not None:
                hidden_states = hidden_states + self._cached_img_output
        else:
            # ── Compute full attention ──
            query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))

            if rotary_emb is not None:
                def apply_rotary_emb(hs, freqs_cos, freqs_sin):
                    x1, x2 = hs.unflatten(-1, (-1, 2)).unbind(-1)
                    cos = freqs_cos[..., 0::2]
                    sin = freqs_sin[..., 1::2]
                    out = torch.empty_like(hs)
                    out[..., 0::2] = x1 * cos - x2 * sin
                    out[..., 1::2] = x1 * sin + x2 * cos
                    return out.type_as(hs)

                query = apply_rotary_emb(query, *rotary_emb)
                key = apply_rotary_emb(key, *rotary_emb)

            # I2V image attention
            hidden_states_img = None
            if encoder_hidden_states_img is not None:
                from diffusers.models.transformers.transformer_wan import _get_added_kv_projections
                key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
                key_img = attn.norm_added_k(key_img)
                key_img = key_img.unflatten(2, (attn.heads, -1))
                value_img = value_img.unflatten(2, (attn.heads, -1))
                hidden_states_img = dispatch_attention_fn(
                    query, key_img, value_img,
                    attn_mask=None, dropout_p=0.0, is_causal=False,
                    backend=None, parallel_config=None,
                )
                hidden_states_img = hidden_states_img.flatten(2, 3)
                hidden_states_img = hidden_states_img.type_as(query)

            # Main attention
            hidden_states = dispatch_attention_fn(
                query, key, value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                backend=None, parallel_config=None,
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.type_as(query)

            # Cache for self-attention only
            if self.attn_type == 'self' and self.interval > 1:
                self._cached_output = hidden_states.clone()
                self._cached_img_output = hidden_states_img.clone() if hidden_states_img is not None else None

            if hidden_states_img is not None:
                hidden_states = hidden_states + hidden_states_img

        # ── Output projection ──
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def clear_cache(self):
        self._cached_output = None
        self._cached_img_output = None


# ─── Model Setup ─────────────────────────────────────────────────────────────

def load_wan_pipeline(model_name, device):
    from diffusers import AutoencoderKLWan, WanPipeline
    vae = AutoencoderKLWan.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        model_name, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    return pipe


def install_caching_processors(transformer, layer_intervals, step_tracker):
    """Replace attention processors with caching versions.

    Args:
        layer_intervals: dict mapping layer_idx -> interval (int).
                         Layers not in dict get interval=1 (always compute).
    """
    processors = {}

    for name, module in transformer.named_modules():
        if type(module).__name__ == 'WanAttention':
            parts = name.split('.')
            layer_idx = None
            attn_type = 'self'
            for j, p in enumerate(parts):
                if p == 'blocks' and j + 1 < len(parts):
                    try:
                        layer_idx = int(parts[j + 1])
                    except ValueError:
                        pass
                if p == 'attn2':
                    attn_type = 'cross'

            if layer_idx is not None:
                interval = layer_intervals.get(layer_idx, 1) if attn_type == 'self' else 1
                proc = CachingWanAttnProcessor(
                    layer_idx, attn_type, interval=interval,
                    step_tracker=step_tracker,
                )
                module.processor = proc
                key = f"{attn_type}_{layer_idx}"
                processors[key] = proc

    cached_count = sum(1 for k, p in processors.items()
                       if k.startswith('self') and p.interval > 1)
    total_self = sum(1 for k in processors if k.startswith('self'))
    intervals_used = sorted(set(layer_intervals.values()))
    print(f"Installed {cached_count}/{total_self} caching self-attn processors "
          f"(intervals={intervals_used})")
    return processors


def reset_processors(transformer):
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0
    for name, module in transformer.named_modules():
        if type(module).__name__ == 'WanAttention':
            module.processor = WanAttnProcessor2_0()


def clear_all_caches(processors):
    for proc in processors.values():
        proc.clear_cache()


# ─── FBC helpers ─────────────────────────────────────────────────────────────

def enable_fbc(pipe, threshold):
    from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
    apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=threshold))


def disable_fbc(pipe):
    _BLOCK_NAMES = ["blocks", "transformer_blocks", "single_transformer_blocks",
                    "temporal_transformer_blocks"]
    for attr_name in _BLOCK_NAMES:
        blocks = getattr(pipe.transformer, attr_name, None)
        if blocks is None or not isinstance(blocks, torch.nn.ModuleList):
            continue
        for block in blocks:
            if hasattr(block, '_diffusers_hook'):
                hook_registry = block._diffusers_hook
                for name in list(hook_registry.hooks.keys()):
                    hook_registry.remove_hook(name)
                del block._diffusers_hook


# ─── Video Generation ────────────────────────────────────────────────────────

def generate_video(pipe, prompt, step_tracker, height=480, width=832,
                   num_frames=17, num_steps=50, seed=42, device="cuda:0"):
    generator = torch.Generator(device=device).manual_seed(seed)
    step_tracker.reset()
    with torch.no_grad():
        output = pipe(
            prompt=prompt, height=height, width=width,
            num_frames=num_frames, num_inference_steps=num_steps,
            generator=generator, output_type="np",
            callback_on_step_end=step_tracker.callback,
        )
    frames = output.frames[0]
    frames_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return frames_tensor


# ─── Config Definitions ─────────────────────────────────────────────────────

def make_uniform_intervals(n_layers, interval):
    """All layers use the same interval."""
    return {i: interval for i in range(n_layers)}


def make_pyramid_intervals(n_layers, tiers):
    """Pyramid: different intervals for different layer ranges.

    tiers: list of (layer_range_end, interval) sorted by layer_range_end.
    Example: [(10, 1), (20, 2), (30, 4)] means:
      layers 0-9: interval=1, layers 10-19: interval=2, layers 20-29: interval=4
    """
    intervals = {}
    prev_end = 0
    for end, interval in tiers:
        for i in range(prev_end, end):
            intervals[i] = interval
        prev_end = end
    return intervals


N_LAYERS = 30

CONFIG_SPECS = {
    # Baseline
    'P0':  {'intervals': make_uniform_intervals(N_LAYERS, 1),
            'fbc': None, 'desc': 'Baseline (compute every step)'},
    # Uniform broadcast intervals
    'P1':  {'intervals': make_uniform_intervals(N_LAYERS, 2),
            'fbc': None, 'desc': 'Uniform interval=2 (50% compute)'},
    'P2':  {'intervals': make_uniform_intervals(N_LAYERS, 3),
            'fbc': None, 'desc': 'Uniform interval=3 (33% compute)'},
    'P3':  {'intervals': make_uniform_intervals(N_LAYERS, 4),
            'fbc': None, 'desc': 'Uniform interval=4 (25% compute)'},
    'P4':  {'intervals': make_uniform_intervals(N_LAYERS, 6),
            'fbc': None, 'desc': 'Uniform interval=6 (17% compute)'},
    'P5':  {'intervals': make_uniform_intervals(N_LAYERS, 8),
            'fbc': None, 'desc': 'Uniform interval=8 (12% compute)'},
    # Pyramid: early layers compute more often
    'P6':  {'intervals': make_pyramid_intervals(N_LAYERS, [(5, 1), (15, 2), (30, 4)]),
            'fbc': None, 'desc': 'Pyramid [1,1,1,1,1, 2x10, 4x15]'},
    'P7':  {'intervals': make_pyramid_intervals(N_LAYERS, [(10, 1), (20, 2), (30, 4)]),
            'fbc': None, 'desc': 'Pyramid [1x10, 2x10, 4x10]'},
    'P8':  {'intervals': make_pyramid_intervals(N_LAYERS, [(5, 1), (15, 2), (25, 3), (30, 4)]),
            'fbc': None, 'desc': 'Pyramid [1x5, 2x10, 3x10, 4x5]'},
    # Uniform + FBC combos (interaction test)
    'P9':  {'intervals': make_uniform_intervals(N_LAYERS, 2),
            'fbc': 0.03, 'desc': 'Uniform N=2 + FBC 0.03'},
    'P10': {'intervals': make_uniform_intervals(N_LAYERS, 3),
            'fbc': 0.03, 'desc': 'Uniform N=3 + FBC 0.03'},
    'P11': {'intervals': make_uniform_intervals(N_LAYERS, 4),
            'fbc': 0.03, 'desc': 'Uniform N=4 + FBC 0.03'},
    # Best pyramid + FBC
    'P12': {'intervals': make_pyramid_intervals(N_LAYERS, [(10, 1), (20, 2), (30, 4)]),
            'fbc': 0.03, 'desc': 'Pyramid [1x10,2x10,4x10] + FBC 0.03'},
}


def compute_theoretical_savings(intervals):
    """Fraction of attention computations saved."""
    total = len(intervals)
    saved = sum(1 - 1/v for v in intervals.values())
    return saved / total


# ─── Experiment Runner ───────────────────────────────────────────────────────

def run_experiment(args):
    print("=" * 60)
    print("AutoAccel Phase 0.8: Attention Output Caching (PAB-style)")
    print("=" * 60)

    device = args.device
    env_info = get_env_info(device)
    print(f"Environment: {json.dumps(env_info, indent=2)}")

    configs = args.configs.split(',')
    if 'P0' not in configs:
        configs.insert(0, 'P0')

    prompts = PROMPTS[:args.num_prompts]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_tracker = StepTracker()

    for cfg_name in configs:
        spec = CONFIG_SPECS[cfg_name]
        savings = compute_theoretical_savings(spec['intervals'])
        print(f"  {cfg_name}: {spec['desc']} (attn savings={savings:.0%})")

    # Load model
    print(f"\nLoading model: {args.model}")
    pipe = load_wan_pipeline(args.model, device)

    all_results = {}
    start_time = time.time()

    for cfg_name in configs:
        spec = CONFIG_SPECS[cfg_name]
        print(f"\n{'─' * 50}")
        print(f"Config: {cfg_name} — {spec['desc']}")
        print(f"{'─' * 50}")

        # Setup caching processors
        reset_processors(pipe.transformer)
        processors = install_caching_processors(
            pipe.transformer, spec['intervals'], step_tracker,
        )

        # Setup FBC
        if spec['fbc'] is not None:
            enable_fbc(pipe, spec['fbc'])
            print(f"  FBC: enabled (threshold={spec['fbc']})")

        # Warmup
        if args.warmup:
            print(f"  Warmup...", end="", flush=True)
            clear_all_caches(processors)
            _ = generate_video(pipe, "warmup test", step_tracker,
                             height=args.height, width=args.width,
                             num_frames=args.num_frames, num_steps=args.num_steps,
                             seed=args.seed, device=device)
            torch.cuda.synchronize()
            gc.collect(); torch.cuda.empty_cache()
            print(" done")

        results = {'times': [], 'frames': []}

        for pi, prompt in enumerate(prompts):
            print(f"  [{pi+1}/{len(prompts)}] \"{prompt[:50]}\"", end="", flush=True)

            clear_all_caches(processors)
            torch.cuda.synchronize()
            t0 = time.time()

            frames = generate_video(pipe, prompt, step_tracker,
                                  height=args.height, width=args.width,
                                  num_frames=args.num_frames, num_steps=args.num_steps,
                                  seed=args.seed, device=device)

            torch.cuda.synchronize()
            gen_time = time.time() - t0
            results['times'].append(gen_time)
            results['frames'].append(frames)
            print(f"  {gen_time:.1f}s")

            gc.collect(); torch.cuda.empty_cache()

        # Teardown FBC
        if spec['fbc'] is not None:
            disable_fbc(pipe)

        results['time_mean'] = float(np.mean(results['times']))
        results['time_std'] = float(np.std(results['times']))
        results['description'] = spec['desc']
        results['fbc_threshold'] = spec['fbc']
        results['attn_savings'] = compute_theoretical_savings(spec['intervals'])
        results['intervals'] = {str(k): v for k, v in spec['intervals'].items()}
        all_results[cfg_name] = results

    total_time = time.time() - start_time

    # Reset processors
    reset_processors(pipe.transformer)

    # ── Quality metrics ──
    baseline_frames = all_results['P0']['frames']
    for key, results in all_results.items():
        if key == 'P0':
            results['ssim'] = [1.0] * len(prompts)
            results['psnr'] = [100.0] * len(prompts)
            results['ssim_mean'] = 1.0
            results['ssim_std'] = 0.0
            results['psnr_mean'] = 100.0
            results['psnr_std'] = 0.0
        else:
            ssims, psnrs = [], []
            for pi in range(len(prompts)):
                ssims.append(compute_ssim(results['frames'][pi], baseline_frames[pi]))
                psnrs.append(compute_psnr(results['frames'][pi], baseline_frames[pi]))
            results['ssim'] = ssims
            results['psnr'] = psnrs
            results['ssim_mean'] = float(np.mean(ssims))
            results['ssim_std'] = float(np.std(ssims))
            results['psnr_mean'] = float(np.mean(psnrs))
            results['psnr_std'] = float(np.std(psnrs))

    # ── Speedups ──
    baseline_time = all_results['P0']['time_mean']
    for key, results in all_results.items():
        results['speedup'] = baseline_time / results['time_mean']

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Config':<8} | {'Desc':<35} | {'Time':>7} | {'Speed':>6} | "
          f"{'SSIM':>10} | {'Savings':>7}")
    print("-" * 85)

    for key in all_results:
        r = all_results[key]
        print(f"{key:<8} | {r['description'][:35]:<35} | "
              f"{r['time_mean']:5.1f}±{r['time_std']:.1f} | "
              f"{r['speedup']:5.2f}x | "
              f"{r['ssim_mean']:.4f}±{r['ssim_std']:.3f} | "
              f"{r['attn_savings']:5.0%}")

    # ── Interaction analysis ──
    print(f"\n{'=' * 80}")
    print("INTERACTION ANALYSIS (AttnCache × FBC)")
    print(f"{'=' * 80}")

    fbc_only_speed = 1.26  # from Phase 0.5
    for combo_key in ['P9', 'P10', 'P11', 'P12']:
        if combo_key not in all_results:
            continue
        # Find the matching attn-cache-only config
        base_map = {'P9': 'P1', 'P10': 'P2', 'P11': 'P3', 'P12': 'P7'}
        base_key = base_map.get(combo_key)
        if base_key and base_key in all_results:
            attn_speed = all_results[base_key]['speedup']
            expected = attn_speed * fbc_only_speed
            actual = all_results[combo_key]['speedup']
            ratio = actual / expected if expected > 0 else 0
            print(f"  {combo_key} ({all_results[combo_key]['description'][:30]}):")
            print(f"    AttnCache={attn_speed:.2f}x × FBC={fbc_only_speed:.2f}x "
                  f"= expected {expected:.2f}x, actual {actual:.2f}x, "
                  f"interaction={ratio:.2f}")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_file = output_dir / f"phase08_results_{timestamp}.json"

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
            'prompts': prompts,
            'warmup': args.warmup,
        },
        'env': env_info,
        'total_time_seconds': round(total_time, 1),
        'results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'frames'}
            for k, v in all_results.items()
        },
    }

    with open(result_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=json_convert)

    print(f"\nResults: {result_file}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--num_frames', type=int, default=17)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--num_prompts', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--configs', default='P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--save_frames', action='store_true')
    parser.add_argument('--output_dir', default='results/autoaccel_phase08')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    run_experiment(args)
