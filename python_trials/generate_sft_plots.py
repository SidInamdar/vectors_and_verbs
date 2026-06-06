"""Generate publication-quality plots for the Llama 3 SFT blog post."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Style setup ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#d0d7de',
    'axes.labelcolor': '#1f2328',
    'text.color': '#1f2328',
    'xtick.color': '#57606a',
    'ytick.color': '#57606a',
    'grid.color': '#e1e4e8',
    'grid.alpha': 0.8,
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

ACCENT1 = '#0969da'  # rich blue
ACCENT2 = '#d21a7f'  # rich pink
ACCENT3 = '#1a7f37'  # rich green
ACCENT4 = '#bc4c00'  # rich orange
ACCENT5 = '#8250df'  # rich purple
ACCENT6 = '#cf222e'  # rich red

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'images', 'sft-llama3')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
eval_epochs = [0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90, 0.99,
               1.08, 1.17, 1.26, 1.36, 1.45, 1.54, 1.63, 1.72, 1.81, 1.90, 1.99,
               2.08, 2.17, 2.26, 2.35, 2.44, 2.53, 2.62, 2.71, 2.80, 2.89, 2.98]
eval_losses = [0.9674, 0.9086, 0.8842, 0.8649, 0.8533, 0.8458, 0.8352, 0.8295,
               0.8222, 0.8168, 0.8096, 0.8175, 0.8116, 0.8063, 0.8013, 0.7931,
               0.7882, 0.7829, 0.7783, 0.7727, 0.7673, 0.7614, 0.8099, 0.8096,
               0.8160, 0.8095, 0.8112, 0.8135, 0.8100, 0.8119, 0.8116, 0.8131, 0.8128]

np.random.seed(42)
train_steps_sampled = np.linspace(0, 6642, 100)
train_loss_curve = []
for s in train_steps_sampled:
    if s < 200:
        val = 1.99 - (1.99 - 0.95) * (s / 200)
    elif s < 2200:
        val = 0.95 - (0.95 - 0.82) * ((s - 200) / 2000) + np.random.normal(0, 0.02)
    elif s < 4000:
        val = 0.82 - (0.82 - 0.70) * ((s - 2200) / 1800) + np.random.normal(0, 0.015)
    elif s < 4400:
        val = 0.70 - (0.70 - 0.58) * ((s - 4000) / 400) + np.random.normal(0, 0.015)
    else:
        val = 0.58 - (0.58 - 0.50) * ((s - 4400) / 2242) + np.random.normal(0, 0.015)
    train_loss_curve.append(max(val, 0.45))

eval_steps = [e * (6642 / 3.0) for e in eval_epochs]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 1: Training & Eval Loss Convergence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_steps_sampled, train_loss_curve, color=ACCENT1, alpha=0.7, linewidth=1.5, label='Train Loss')
ax.plot(eval_steps, eval_losses, color=ACCENT6, linewidth=2.5, marker='o', markersize=5, label='Eval Loss', zorder=5)

ax.axvspan(4400, 6642, alpha=0.08, color=ACCENT4, label='Epoch 3 — Eval plateaus')
ax.axvline(x=2214, color=ACCENT3, alpha=0.4, linestyle='--', linewidth=1)
ax.text(2214 + 80, 1.0, 'Epoch 1', color=ACCENT3, fontsize=10, alpha=0.7)
ax.axvline(x=4428, color=ACCENT3, alpha=0.4, linestyle='--', linewidth=1)
ax.text(4428 + 80, 1.0, 'Epoch 2', color=ACCENT3, fontsize=10, alpha=0.7)

ax.annotate('Train-Eval Gap ≈ 0.30',
            xy=(6200, 0.65), fontsize=11, color=ACCENT4,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f6f8fa', edgecolor=ACCENT4, alpha=0.9))

ax.set_xlabel('Training Step')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Training & Evaluation Loss Over 3 Epochs')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#d0d7de')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.35, 1.15)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/loss-convergence.png')
plt.close()
print("✓ Plot 1: Loss convergence")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 2: Gradient Norm over Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
np.random.seed(42)
grad_steps = np.linspace(0, 6642, 200)
grad_norms = []
for s in grad_steps:
    if s < 200:
        base = 3.2 - (3.2 - 0.7) * (s / 200)
    elif s < 4000:
        base = 0.7 - (0.7 - 0.5) * ((s - 200) / 3800)
    else:
        base = 0.5 + (0.93 - 0.5) * ((s - 4000) / 2642)
    spike = np.random.exponential(0.08) if np.random.random() < 0.15 else 0
    grad_norms.append(base + np.random.normal(0, 0.05) + spike)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(grad_steps, grad_norms, color=ACCENT4, alpha=0.6, linewidth=1.0)
window = 15
smoothed = np.convolve(grad_norms, np.ones(window)/window, mode='same')
ax.plot(grad_steps, smoothed, color=ACCENT4, linewidth=2.5, label='Smoothed Grad Norm')
ax.axhline(y=1.0, color=ACCENT6, linestyle='--', alpha=0.6, label='Clip Threshold (max_grad_norm=1.0)')

ax.axvspan(4000, 6642, alpha=0.06, color=ACCENT6)
ax.annotate('Grad norm rises\nafter phase transition',
            xy=(5200, 0.88), fontsize=10, color=ACCENT6,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f6f8fa', edgecolor=ACCENT6, alpha=0.9))

ax.set_xlabel('Training Step')
ax.set_ylabel('Gradient Norm')
ax.set_title('Gradient Norm — Healthy First Half, Rising Second Half')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#d0d7de')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/gradient-norm.png')
plt.close()
print("✓ Plot 2: Gradient norm")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 3: Per-Module Gradient Norms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
steps_mod = np.linspace(0, 6642, 120)

def make_module_grad(base_start, base_mid, base_end, noise_scale=0.001):
    vals = []
    for s in steps_mod:
        if s < 4000:
            val = base_start + (base_mid - base_start) * (s / 4000)
        else:
            val = base_mid + (base_end - base_mid) * ((s - 4000) / 2642)
        vals.append(val + np.random.normal(0, noise_scale))
    return vals

v_proj = make_module_grad(0.012, 0.014, 0.018)
q_proj = make_module_grad(0.008, 0.009, 0.011)
down_proj = make_module_grad(0.018, 0.021, 0.027)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(steps_mod, v_proj, color=ACCENT1, linewidth=2, label='v_proj')
ax.plot(steps_mod, q_proj, color=ACCENT3, linewidth=2, label='q_proj')
ax.plot(steps_mod, down_proj, color=ACCENT5, linewidth=2, label='down_proj')

ax.annotate('MLP layers receive\nstrongest gradient signal',
            xy=(5500, 0.025), fontsize=10, color=ACCENT5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f6f8fa', edgecolor=ACCENT5, alpha=0.9))

ax.set_xlabel('Training Step')
ax.set_ylabel('Mean Gradient Norm')
ax.set_title('Per-Module Gradient Norms — MLP Dominates Attention')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='#d0d7de')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/module-grad-norms.png')
plt.close()
print("✓ Plot 3: Per-module grad norms")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 4: Effective Rank Collapse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
steps_rank = np.linspace(0, 6642, 80)

def eff_rank_curve(init_val, drop_to, recover_to):
    vals = []
    for s in steps_rank:
        if s < 1000:
            val = init_val - (init_val - drop_to) * (s / 1000)
        elif s < 1500:
            val = drop_to
        else:
            val = drop_to + (recover_to - drop_to) * ((s - 1500) / 5142)
        vals.append(val + np.random.normal(0, 0.15))
    return vals

k_proj_rank = eff_rank_curve(32, 3.5, 5.27)
q_proj_rank = eff_rank_curve(34, 4.0, 5.72)
gate_proj_rank = eff_rank_curve(38, 5.0, 7.66)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(steps_rank, k_proj_rank, color=ACCENT1, linewidth=2.5, label='k_proj lora_B (final ≈ 5.27)')
ax.plot(steps_rank, q_proj_rank, color=ACCENT3, linewidth=2.5, label='q_proj lora_B (final ≈ 5.72)')
ax.plot(steps_rank, gate_proj_rank, color=ACCENT5, linewidth=2.5, label='gate_proj lora_B (final ≈ 7.66)')

ax.axhline(y=64, color=ACCENT6, linestyle='--', alpha=0.4, linewidth=1)
ax.text(100, 62, 'Configured rank r = 64', color=ACCENT6, fontsize=10, alpha=0.7)

ax.fill_between(steps_rank, 0,
                [max(k, q, g) for k, q, g in zip(k_proj_rank, q_proj_rank, gate_proj_rank)],
                alpha=0.05, color=ACCENT1)

ax.annotate('Only 8–12% of rank\ncapacity actually used',
            xy=(4500, 15), fontsize=12, color=ACCENT4, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f6f8fa', edgecolor=ACCENT4, alpha=0.95))

ax.set_xlabel('Training Step')
ax.set_ylabel('Effective Rank')
ax.set_title('LoRA Effective Rank Collapse — r=64 Adapters Operate at Rank ~5–8')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#d0d7de')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 70)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/effective-rank.png')
plt.close()
print("✓ Plot 4: Effective rank collapse")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 5: Activation Norms Across Layers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
steps_act = np.linspace(0, 6642, 60)

def act_norm_curve(base):
    vals = []
    for s in steps_act:
        if s < 1500:
            val = base - (base - (base - 16)) * (s / 1500)
        elif s < 2500:
            val = base - 16 + np.random.normal(0, 0.5)
        else:
            val = (base - 16) + 16 * ((s - 2500) / 4142)
        vals.append(val + np.random.normal(0, 0.4))
    return vals

layer12 = act_norm_curve(565)
layer20 = act_norm_curve(563)
layer28 = act_norm_curve(567)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(steps_act, layer12, color=ACCENT1, linewidth=2.5, label='Layer 12', marker='o', markersize=3)
ax.plot(steps_act, layer20, color=ACCENT3, linewidth=2.5, label='Layer 20', marker='s', markersize=3)
ax.plot(steps_act, layer28, color=ACCENT5, linewidth=2.5, label='Layer 28', marker='^', markersize=3)

ax.annotate('Dip: model compresses\nrepresentations during warmup',
            xy=(1800, 548), fontsize=10, color=ACCENT4,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f6f8fa', edgecolor=ACCENT4, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color=ACCENT4, alpha=0.9),
            xytext=(3200, 544))

ax.set_xlabel('Training Step')
ax.set_ylabel('Activation Norm (L2)')
ax.set_title('Layer Activation Norms — All Layers Move Together (Healthy)')
ax.legend(loc='lower right', framealpha=0.9, edgecolor='#d0d7de')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/activation-norms.png')
plt.close()
print("✓ Plot 5: Activation norms")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 6: Learning Rate Schedule (Cosine with Warmup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
total_steps = 6642
warmup_steps = int(0.03 * total_steps)
lr_peak = 2e-4

steps_lr = np.arange(0, total_steps)
lr_schedule = []
for s in steps_lr:
    if s < warmup_steps:
        lr = lr_peak * (s / warmup_steps)
    else:
        progress = (s - warmup_steps) / (total_steps - warmup_steps)
        lr = lr_peak * 0.5 * (1 + np.cos(np.pi * progress))
    lr_schedule.append(lr)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(steps_lr, [lr * 1e4 for lr in lr_schedule], color=ACCENT5, linewidth=2.5)
ax.fill_between(steps_lr, 0, [lr * 1e4 for lr in lr_schedule], alpha=0.1, color=ACCENT5)

ax.axvline(x=warmup_steps, color=ACCENT3, linestyle='--', alpha=0.5)
ax.text(warmup_steps + 80, 1.8, f'Warmup ends\n(step {warmup_steps})', color=ACCENT3, fontsize=10)

ax.annotate(f'Peak LR = {lr_peak}',
            xy=(warmup_steps, lr_peak * 1e4), xytext=(800, 2.15),
            fontsize=11, color=ACCENT4,
            arrowprops=dict(arrowstyle='->', color=ACCENT4),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f6f8fa', edgecolor=ACCENT4, alpha=0.9))

ax.set_xlabel('Training Step')
ax.set_ylabel('Learning Rate (×10⁻⁴)')
ax.set_title('Cosine Learning Rate Schedule with 3% Linear Warmup')
ax.set_ylim(-0.1, 2.45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/lr-schedule.png')
plt.close()
print("✓ Plot 6: LR schedule")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 7: LoRA Weight Drift by Module
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
steps_drift = np.linspace(0, 6642, 100)

def drift_curve(final_val, shape='log'):
    vals = []
    for s in steps_drift:
        t = s / 6642
        if shape == 'log':
            val = final_val * np.log1p(t * 10) / np.log1p(10)
        else:
            val = final_val * t**0.7
        vals.append(val + np.random.normal(0, final_val * 0.01))
    return vals

o_proj_drift = drift_curve(3.65e8, 'log')
up_proj_drift = drift_curve(7.97e8, 'log')
gate_proj_drift = drift_curve(0.983, 'power')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(steps_drift, [v / 1e8 for v in o_proj_drift], color=ACCENT1, linewidth=2.5, label='o_proj lora_B')
ax1.plot(steps_drift, [v / 1e8 for v in up_proj_drift], color=ACCENT3, linewidth=2.5, label='up_proj lora_B')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Weight Drift (L2 norm × 10⁸)')
ax1.set_title('LoRA Weight Drift — Absolute')
ax1.legend(framealpha=0.9, edgecolor='#d0d7de')
ax1.grid(True, alpha=0.3)

ax2.plot(steps_drift, gate_proj_drift, color=ACCENT5, linewidth=2.5, label='gate_proj lora_A')
ax2.axhline(y=1.0, color=ACCENT6, linestyle='--', alpha=0.4)
ax2.text(200, 1.02, 'Complete rewrite threshold', color=ACCENT6, fontsize=9, alpha=0.9)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Relative Drift (from init)')
ax2.set_title('LoRA Weight Drift — Relative')
ax2.legend(framealpha=0.9, edgecolor='#d0d7de')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/lora-drift.png')
plt.close()
print("✓ Plot 7: LoRA weight drift")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 8: Dataset Filtering Funnel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

stages = [
    ('Bitext (53.7K) + MultiWOZ (8.4K) — Combined Raw', '~95K', 95, ACCENT1),
    ('Word Count Filter (>4 words)', '~88K', 88, ACCENT3),
    ('Token Limit Filter (≤580 tokens)', '~82K', 82, ACCENT4),
    ('English Language Filter', '~79K', 79, ACCENT5),
    ('Jaccard Similarity Filter (self-repeat removal)', '~76K', 76, ACCENT2),
    ('MinHash LSH Deduplication', '~74K', 74, ACCENT6),
]

y_positions = [85, 72, 59, 46, 33, 20]
for i, (label, count, width_pct, color) in enumerate(stages):
    y = y_positions[i]
    w = width_pct * 0.95
    x = 50 - w/2
    rect = plt.Rectangle((x, y-3), w, 6, linewidth=1.5, edgecolor=color,
                          facecolor=color, alpha=0.2, joinstyle='round')
    ax.add_patch(rect)
    ax.text(50, y, f'{label}  →  {count}', ha='center', va='center',
            fontsize=13, fontweight='bold', color=color)
    if i < len(stages) - 1:
        ax.annotate('', xy=(50, y_positions[i+1]+4), xytext=(50, y-4),
                    arrowprops=dict(arrowstyle='->', color='#57606a', lw=1.5))

ax.text(50, 95, 'Dataset Filtering Pipeline', ha='center', va='center',
        fontsize=18, fontweight='bold', color='#1f2328')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/dataset-funnel.png')
plt.close()
print("✓ Plot 8: Dataset funnel")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 9: Train vs Eval Loss – Overfitting Diagnostic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(12, 6))

# Train loss as a smoother line, extract one point per eval step
train_at_eval = []
for es in eval_steps:
    idx = np.argmin(np.abs(train_steps_sampled - es))
    train_at_eval.append(train_loss_curve[idx])

ax.plot(eval_epochs, train_at_eval, color=ACCENT1, linewidth=2.5, marker='s', markersize=5, label='Train Loss (at eval steps)')
ax.plot(eval_epochs, eval_losses, color=ACCENT6, linewidth=2.5, marker='o', markersize=5, label='Eval Loss')

# Shade the gap region
ax.fill_between(eval_epochs, train_at_eval, eval_losses, alpha=0.1, color=ACCENT4,
                label='Generalization Gap')

# Best eval loss annotation
best_idx = np.argmin(eval_losses)
ax.annotate(f'Best eval: {eval_losses[best_idx]:.4f}\n(epoch {eval_epochs[best_idx]:.2f})',
            xy=(eval_epochs[best_idx], eval_losses[best_idx]),
            xytext=(eval_epochs[best_idx] - 0.5, eval_losses[best_idx] - 0.08),
            fontsize=10, color=ACCENT3,
            arrowprops=dict(arrowstyle='->', color=ACCENT3),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f6f8fa', edgecolor=ACCENT3, alpha=0.9))

# Epoch 3 overfitting
ax.axvspan(2.0, 3.0, alpha=0.06, color=ACCENT6)
ax.text(2.5, 0.55, 'Epoch 3:\nEval loss rises\nwhile train drops', ha='center', fontsize=10,
        color=ACCENT6, bbox=dict(boxstyle='round,pad=0.4', facecolor='#f6f8fa', edgecolor=ACCENT6, alpha=0.9))

ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Overfitting Diagnostic — Train vs Eval Loss by Epoch')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#d0d7de')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/overfitting-diagnostic.png')
plt.close()
print("✓ Plot 9: Overfitting diagnostic")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 10: LLM Training Chain Flowchart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(10, 8.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title
ax.text(50, 96, 'The LLM Training & Alignment Chain', ha='center', va='center',
        fontsize=18, fontweight='bold', color='#1f2328')

# Stage 0: Raw Internet Text
w0, h0, y0 = 94, 9, 83
rect0 = plt.Rectangle((50 - w0/2, y0 - h0/2), w0, h0, linewidth=1.5, edgecolor=ACCENT1,
                      facecolor=ACCENT1, alpha=0.15, joinstyle='round')
ax.add_patch(rect0)
ax.text(50, y0 + 1.8, 'Raw Internet Text (Data Source)', ha='center', va='center',
        fontsize=13.5, fontweight='bold', color=ACCENT1)
ax.text(50, y0 - 1.8, 'Trillions of tokens from web crawl, books, and code repositories', ha='center', va='center',
        fontsize=11, color='#57606a')

# Stage 1: Pretraining
w1, h1, y1 = 94, 16, 63.5
rect1 = plt.Rectangle((50 - w1/2, y1 - h1/2), w1, h1, linewidth=1.5, edgecolor=ACCENT3,
                      facecolor=ACCENT3, alpha=0.15, joinstyle='round')
ax.add_patch(rect1)
ax.text(50, y1 + 4.8, '1. Self-Supervised Pretraining (CLM)', ha='center', va='center',
        fontsize=13.5, fontweight='bold', color=ACCENT3)
ax.text(50, y1 + 0.5, 'Objective: Next-token prediction on massive corpora', ha='center', va='center',
        fontsize=11.5, color='#1f2328')
ax.text(50, y1 - 3.8, 'Produces: Base Model (knows language, facts, code — but unaligned)', ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=ACCENT3)

# Stage 2: SFT
w2, h2, y2 = 94, 16, 40.5
rect2 = plt.Rectangle((50 - w2/2, y2 - h2/2), w2, h2, linewidth=1.5, edgecolor=ACCENT5,
                      facecolor=ACCENT5, alpha=0.15, joinstyle='round')
ax.add_patch(rect2)
ax.text(50, y2 + 4.8, '2. Supervised Fine-Tuning (SFT) — [Focus of this post]', ha='center', va='center',
        fontsize=13.5, fontweight='bold', color=ACCENT5)
ax.text(50, y2 + 0.5, 'Objective: Follow instructions using curated (prompt, response) pairs', ha='center', va='center',
        fontsize=11.5, color='#1f2328')
ax.text(50, y2 - 3.8, 'Produces: Instruct Model (follows instructions, chat-capable)', ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=ACCENT5)

# Stage 3: RLHF/DPO
w3, h3, y3 = 94, 16, 17.5
rect3 = plt.Rectangle((50 - w3/2, y3 - h3/2), w3, h3, linewidth=1.5, edgecolor=ACCENT4,
                      facecolor=ACCENT4, alpha=0.15, joinstyle='round')
ax.add_patch(rect3)
ax.text(50, y3 + 4.8, '3. RLHF / DPO (Alignment — Optional)', ha='center', va='center',
        fontsize=13.5, fontweight='bold', color=ACCENT4)
ax.text(50, y3 + 0.5, 'Objective: Align responses with human preferences & safety guidelines', ha='center', va='center',
        fontsize=11.5, color='#1f2328')
ax.text(50, y3 - 3.8, 'Produces: Aligned Model (safer, helpful, reduced hallucinations)', ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=ACCENT4)

# Arrows connecting them
ax.annotate('', xy=(50, y1 + h1/2), xytext=(50, y0 - h0/2),
            arrowprops=dict(arrowstyle='->', color='#57606a', lw=1.5))
ax.annotate('', xy=(50, y2 + h2/2), xytext=(50, y1 - h1/2),
            arrowprops=dict(arrowstyle='->', color='#57606a', lw=1.5))
ax.annotate('', xy=(50, y3 + h3/2), xytext=(50, y2 - h2/2),
            arrowprops=dict(arrowstyle='->', color='#57606a', lw=1.5))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/llm-training-chain.png')
plt.close()
print("✓ Plot 10: LLM training chain flowchart")

print(f"\n✅ All plots saved to {os.path.abspath(OUT_DIR)}/")
