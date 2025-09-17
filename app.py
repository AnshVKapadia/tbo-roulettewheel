# app.py
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Roulette Spinner", page_icon="🎡", layout="centered")
st.title("🎡 Roulette Wheel Spinner")

# =========================
# HARD-CODED CONSTANTS
# =========================
LABELS = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange"]
WEIGHTS = None                 # None => equal weights across LABELS
SEED = 0                       # 0 => random each spin; nonzero => deterministic
SNAP_TO_CENTER = False         # do not snap to slice center (random point inside slice if False)
FRAMES = 120                   # number of frames in the animation
SPIN_TIME = 4.00               # seconds
SPINS = 10                     # full rotations
SLOW_K = 1.50                  # slow down during the last k seconds
RANDOM_SPIN_TIME_RANGE = (0.8, 1.2)   # multiply SPIN_TIME by U[a,b] each spin
# =========================

# ---- Parse labels/weights from constants ----
labels = [s.strip() for s in LABELS if str(s).strip()]
if not labels:
    st.stop()

if WEIGHTS is None:
    weights = np.ones(len(labels), dtype=float)
else:
    weights = np.asarray(WEIGHTS, dtype=float)
    if len(weights) != len(labels) or np.any(weights < 0):
        st.error("WEIGHTS must be nonnegative and match the number of LABELS.")
        st.stop()

if weights.sum() <= 0:
    st.error("At least one weight must be positive.")
    st.stop()

# ---- Color mapping (labels -> intuitive colors, fallback to HSL) ----
COLOR_MAP = {
    "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
    "purple": "purple", "orange": "orange", "pink": "pink", "black": "black",
    "white": "white", "gray": "gray", "grey": "gray", "brown": "saddlebrown",
}
def choose_colors(lbls):
    out = []
    n = max(1, len(lbls))
    for i, lbl in enumerate(lbls):
        key = lbl.lower()
        out.append(COLOR_MAP.get(key, f"hsl({int(360*i/n)}, 70%, 50%)"))
    return out

colors = choose_colors(labels)

# ---- Session state ----
st.session_state.setdefault("rotation", 0.0)  # wheel's rotation (deg)
st.session_state.setdefault("result", None)
st.session_state.setdefault("slot", st.empty())   # single rendering slot
st.session_state.setdefault("spin_id", 0)         # for unique frame keys
slot = st.session_state.slot

# ---- Geometry helpers ----
total = float(weights.sum())
fractions = weights / total
slice_angles = fractions * 360.0
cum = np.cumsum(slice_angles)
starts = np.insert(cum[:-1], 0, 0.0)   # start angles (deg) from 3 o'clock, clockwise
centers = (starts + cum) / 2.0         # center angles (deg)
TOP_ANGLE = 90.0                        # (winner alignment math unchanged)

# ---- Wheel figure (pointer on the RIGHT at 3 o'clock) ----
def wheel_fig(rotation_deg: float = 0.0) -> go.Figure:
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.25,
            sort=False,
            direction="clockwise",
            rotation=rotation_deg,
            textinfo="label",
            textfont=dict(size=14),
            marker=dict(colors=colors, line=dict(color="white", width=2)),
        )]
    )
    fig.update_layout(
        width=600, height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    # White arrow at 3 o’clock (right side) in paper coords
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=1.05, y0=0.5, x1=0.95, y1=0.5,
        line=dict(color="white", width=5),
        layer="above"
    )
    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path="M 0.95 0.47 L 0.95 0.53 L 0.88 0.5 Z",
        line=dict(color="black", width=1.5),
        fillcolor="white",
        layer="above"
    )
    return fig

# ---- Winner alignment (unchanged) ----
def rotation_to_align(theta_deg: float, current_rot: float, spins_full: int) -> float:
    desired = (TOP_ANGLE - theta_deg) % 360.0
    delta = (desired - (current_rot % 360.0)) % 360.0
    return current_rot + spins_full * 360.0 + delta

# ---- Kinematic schedule: constant speed then constant deceleration to zero over last k seconds ----
def angle_schedule(start_rot: float, final_rot: float, total_time: float, k: float, frames: int):
    total_angle = final_rot - start_rot
    total_time = max(1e-9, total_time)
    k = max(0.0, min(k, total_time))
    T1 = total_time - k

    if k == 0.0:
        times = np.linspace(0.0, total_time, frames)
        return list(start_rot + total_angle * (times / total_time))

    denom = (T1 + 0.5 * k)
    if abs(denom) < 1e-12:
        w1 = 2.0 * total_angle / k
    else:
        w1 = total_angle / denom
    a = -w1 / k  # decel so velocity hits 0 at the end

    times = np.linspace(0.0, total_time, frames)
    angles = []
    for t in times:
        if t <= T1:
            theta = start_rot + w1 * t
        else:
            tau = t - T1
            theta = start_rot + w1 * T1 + (w1 * tau + 0.5 * a * tau * tau)
        angles.append(theta)
    return angles

# ---- UI ----
col1, col2 = st.columns([1, 1])
clicked = col1.button("🎲 Spin", use_container_width=True)

if clicked:
    st.session_state.spin_id += 1
    rng = np.random.default_rng(None if SEED == 0 else SEED)

    # weighted choice of slice
    idx = rng.choice(len(labels), p=fractions)
    st.session_state.result = labels[idx]

    # choose angle within that slice (or center)
    if SNAP_TO_CENTER:
        target_theta = centers[idx]
    else:
        target_theta = rng.uniform(starts[idx], cum[idx])

    start_rot = float(st.session_state.rotation)
    final_rot = rotation_to_align(target_theta, start_rot, SPINS)

    # randomize spin duration per spin
    spin_time_effective = SPIN_TIME * rng.uniform(*RANDOM_SPIN_TIME_RANGE)

    # build schedule with slowdown in last SLOW_K seconds
    angles = angle_schedule(start_rot, final_rot, spin_time_effective, SLOW_K, FRAMES)

    # animate (unique key per frame to avoid duplicate-key errors)
    for i, rot in enumerate(angles):
        frame_key = f"spin_{st.session_state.spin_id}_{i}"
        slot.plotly_chart(wheel_fig(rot), use_container_width=False, key=frame_key)
        time.sleep(spin_time_effective / max(FRAMES, 1))

    st.session_state.rotation = final_rot % 360.0
    st.rerun()

# Idle wheel
slot.plotly_chart(wheel_fig(st.session_state.rotation), use_container_width=False, key="idle_wheel")

# Result display
if st.session_state.result:
    col2.success(f"Result: **{st.session_state.result}**")


