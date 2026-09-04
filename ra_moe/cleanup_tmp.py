# %% ── Cleanup /tmp to free disk space ────────────────────────────────────────
# Safe to run after data_prep — all files already uploaded to GCS.
# Run this BEFORE the training script (it re-downloads only what it needs).

import os, shutil, subprocess

def get_disk_usage():
    result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if len(lines) >= 2:
        parts = lines[1].split()
        return f"Used: {parts[2]} / {parts[1]} ({parts[4]})"
    return "unknown"

print(f"  BEFORE cleanup: {get_disk_usage()}")

# ── /tmp data directories (already on GCS) ──
TMP_DIRS = [
    '/tmp/ra_moe_long_L5_1_multires_v2_data',
    '/tmp/ra_moe_long_L5_1_multires_v2_models',
    '/tmp/ra_moe_long_L5_1_data',
    '/tmp/ra_moe_long_L5_1_models',
    '/tmp/ra_moe_long_asl6_data',
    '/tmp/ra_moe_long_asl6_models',
    '/tmp/ra_moe_short_L5_1_data',
    '/tmp/ra_moe_short_L5_1_models',
    '/tmp/ra_moe_short_L5_2_data',
    '/tmp/ra_moe_short_L5_2_models',
    '/tmp/ra_moe_short_L5_1_multires_v2_data',
    '/tmp/ra_moe_short_L5_1_multires_v2_models',
]

freed_mb = 0
for d in TMP_DIRS:
    if os.path.exists(d):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(d) for f in fns
        ) / 1e6
        shutil.rmtree(d)
        print(f"    Removed {d} ({size:.0f} MB)")
        freed_mb += size

# ── Stale .npy / .npz in /tmp root ──
for f in os.listdir('/tmp'):
    fp = os.path.join('/tmp', f)
    if os.path.isfile(fp) and f.endswith(('.npz', '.npy', '.h5', '.parquet')):
        size = os.path.getsize(fp) / 1e6
        os.remove(fp)
        print(f"    Removed /tmp/{f} ({size:.1f} MB)")
        freed_mb += size

# ── Python/pip caches ──
CACHE_DIRS = [
    os.path.expanduser('~/.cache/pip'),
    '/root/.cache/pip',
    '/tmp/pip-*',
]
for pattern in CACHE_DIRS:
    import glob
    for d in glob.glob(pattern):
        if os.path.exists(d):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(d) for f in fns
            ) / 1e6
            shutil.rmtree(d)
            print(f"    Removed {d} ({size:.0f} MB)")
            freed_mb += size

print(f"\n  Freed: {freed_mb:.0f} MB")
print(f"  AFTER cleanup: {get_disk_usage()}")
print(f"\n  Training script will re-download fold_9_*.npz from GCS as needed.")
