# app.py
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Roulette Spinner", page_icon="🎡", layout="centered")
st.title("🎡 American Roulette Wheel")

# =========================
# HARD-CODED AMERICAN ROULETTE CONSTANTS
# =========================
LABELS = [
    "0", "28", "9", "26", "30", "11", "7", "20", "32", "17", "5", "22",
    "34", "15", "3", "24", "36", "13", "1", "00", "27", "10", "25", "29",
    "12", "8", "19", "31", "18", "6", "21", "33", "16", "4", "23", "35", "14", "2"
]
# Red numbers in American roulette
RED_NUMS = {"1","3","5","7","9","12","14","16","18","19","21","23",
            "25","27","30","32","34","36"}
# Green numbers
GREEN_NUMS = {"0","00"}

WEIGHTS = None                 # None => equal weights
SEED = 0                       # 0 => random each spin
SNAP_TO_CENTER = False
FRAMES = 90
SPIN_TIME = 4.01
SPINS = 10
SLOW_K = 1.50
RANDOM_SPIN_TIME_RANGE = (0.8, 1.2)
# =========================

# ---- Parse labels/weights ----
labels = LABELS
if WEIGHTS is None:
    weights = np.ones(len(labels), dtype=float)
else:
    weights = np.asarray(WEIGHTS, dtype=float)
    if len(weights) != len(labels) or np.any(weights < 0):
        st.error("Bad weights array")
        st.stop()
if weights.sum() <= 0:
    st.error("At least one weight must be positive.")
    st.stop()

# ---- Color chooser ----
def choose_colors(lbls):
    out = []
    for lbl in lbls:
        if lbl in GREEN_NUMS:
            out.append("green")
        elif lbl in RED_NUMS:
            out.append("red")
        else:
            out.append("black")
    return out

colors = choose_colors(labels)

# ---- Session state ----
st.session_state.setdefault("rotation", 0.0)
st.session_state.setdefault("result", None)
st.session_state.setdefault("slot", st.empty())
st.session_state.setdefault("spin_id", 0)
slot = st.session_state.slot

# ---- Geometry helpers ----
total = float(weights.sum())
fractions = weights / total
slice_angles = fractions * 360.0
cum = np.cumsum(slice_angles)
starts = np.insert(cum[:-1], 0, 0.0)
centers = (starts + cum) / 2.0
TOP_ANGLE = 90.0  # alignment math unchanged

# ---- Wheel figure (pointer on right at 3 o’clock) ----
def wheel_fig(rotation_deg: float = 0.0) -> go.Figure:
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.25,
            sort=False,
            direction="clockwise",
            rotation=rotation_deg,
            textinfo="label",  # show only labels
            textfont=dict(size=14),
            marker=dict(colors=colors, line=dict(color="white", width=2)),
        )]
    )
    fig.update_layout(width=600, height=600, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    
    # Pointer head
    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        # base still at (0.95, 0.5), tip pulled inward a bit
        path="M 1.0 0.485 L 1.0 0.515 L 0.96 0.5 Z",
        line=dict(color="black", width=1.2),
        fillcolor="white",
        layer="above"
    )

    return fig

# ---- Rotation math ----
def rotation_to_align(theta_deg: float, current_rot: float, spins_full: int) -> float:
    desired = (TOP_ANGLE - theta_deg) % 360.0
    delta = (desired - (current_rot % 360.0)) % 360.0
    return current_rot + spins_full * 360.0 + delta

# ---- Angle schedule (constant speed then uniform slowdown in last k sec) ----
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
    a = -w1 / k
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

    idx = rng.choice(len(labels), p=fractions)
    st.session_state.result = labels[idx]

    if SNAP_TO_CENTER:
        target_theta = centers[idx]
    else:
        target_theta = rng.uniform(starts[idx], cum[idx])

    start_rot = float(st.session_state.rotation)
    final_rot = rotation_to_align(target_theta, start_rot, SPINS)

    spin_time_effective = SPIN_TIME * rng.uniform(*RANDOM_SPIN_TIME_RANGE)
    angles = angle_schedule(start_rot, final_rot, spin_time_effective, SLOW_K, FRAMES)

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
