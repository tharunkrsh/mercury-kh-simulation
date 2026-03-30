"""
export_trajectories.py
======================
Run this first — it generates the trajectory JSON that Blender imports.

    python export_trajectories.py

Outputs:
    trajectories.json   — particle paths for Blender
"""

import numpy as np
import json

# ── Boris pusher ───────────────────────────────────────────────────────────────

def boris(v, E, B, q, m, dt):
    v_minus = v + (q * E / m) * (dt / 2.0)
    t       = (q * B / m) * (dt / 2.0)
    s       = 2.0 * t / (1.0 + np.dot(t, t))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus  = v_minus + np.cross(v_prime, s)
    return v_plus + (q * E / m) * (dt / 2.0)

def kh_fields(r, R, m_mode=0, B0=1.0, v0_x=1.0):
    x, y  = r[0], r[1]
    r_sq  = x**2 + y**2
    theta = np.arctan2(y, x)
    sign  = (-1)**(m_mode + 1)
    Ex = sign * r_sq * np.exp(-r_sq / R**2) * np.cos(theta) if R > 0 else 0.0
    Ey = sign * r_sq * np.exp(-r_sq / R**2) * np.sin(theta) if R > 0 else 0.0
    Bz = (-1)**m_mode * B0
    Ey += -v0_x * Bz if y > 0 else v0_x * Bz
    return np.array([Ex, Ey, 0.0]), np.array([0.0, 0.0, Bz])

def run(r_init, charge, R, dt=0.005, steps=8000):
    r, v = r_init.copy(), np.zeros(3)
    traj = []
    escape_r = max(2.0 * R, 15.0)
    for _ in range(steps):
        traj.append(r.tolist())
        if r[0]**2 + r[1]**2 > escape_r**2:
            return traj, False   # escaped
        E, B = kh_fields(r, R)
        v = boris(v, E, B, float(charge), 1.0, dt)
        r += v * dt
    return traj, True            # trapped


# ── Define the showcase particles ─────────────────────────────────────────────
# Each entry: (label, start, charge, R, color_hex, trapped_expected)
particles = [
    # Trapped petal orbits — the money shots
    ("trapped_1",  np.array([ 10.,  3., 0.]), +1, 10, "#00ccff", True),
    ("trapped_2",  np.array([-10., -3., 0.]), +1, 10, "#0088ff", True),
    ("trapped_3",  np.array([  5.,  5., 0.]), +1, 10, "#44aaff", True),

    # Large R — tight beautiful petal
    ("petal_big_1", np.array([15.,  3., 0.]), +1, 20, "#ff6600", True),
    ("petal_big_2", np.array([-8., -8., 0.]), +1, 20, "#ff8800", True),

    # Escaped particles — fly off dramatically
    ("escaped_1",  np.array([30.,  20., 0.]), +1, 10, "#ff3366", False),
    ("escaped_2",  np.array([30.,  50., 0.]), +1, 10, "#ff0044", False),

    # Negative ion — deflected differently
    ("negative_1", np.array([10.,  3., 0.]), -1, 10, "#cc44ff", False),
    ("negative_2", np.array([-10.,-3., 0.]), -1, 10, "#aa22ff", False),
]

print("Generating trajectories...")
output = []

for label, r0, charge, R, color, expected in particles:
    traj, trapped = run(r0, charge, R)

    # Downsample to max 800 points for Blender performance
    step = max(1, len(traj) // 800)
    traj_ds = traj[::step]

    # Normalise into Blender-friendly scale (divide by 5)
    scale = 5.0
    traj_scaled = [[p[0]/scale, p[1]/scale, p[2]/scale] for p in traj_ds]

    output.append({
        "label":   label,
        "color":   color,
        "trapped": trapped,
        "charge":  charge,
        "R":       R,
        "points":  traj_scaled,
        "n_points": len(traj_scaled),
    })
    status = "TRAPPED" if trapped else "ESCAPED"
    print(f"  {label:20s} {len(traj_scaled):4d} pts  {status}  colour={color}")

with open("trajectories.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved trajectories.json — {len(output)} particle paths ready for Blender")
