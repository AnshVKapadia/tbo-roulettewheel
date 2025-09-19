# app.py
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TAMS Roulette Spinner", page_icon="🎡", layout="centered")
st.title("🎡 TAMS Roulette Wheel")

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
FRAMES = 120
SPIN_TIME = 4.00
SPINS = 5
SLOW_K = 1.50                  # (kept for backwards-compat; unused by new schedule)
RANDOM_SPIN_TIME_RANGE = (0.8, 1.2)
DECEL_POWER = 2.0              # >=0. Higher = stronger late braking (2 good, try 1..4)

WIDTH = 400
HEIGHT = 400
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
TOP_ANGLE = 90.0  # your alignment constant that worked for your pointer setup

# === POINTER ANGLE CALCULATIONS ===
POINTER_ANGLE = TOP_ANGLE  # keep using your current constant (90.0 in your code)

def visible_index(rotation_deg: float) -> int:
    """
    Given the go.Pie(rotation=rotation_deg), return the index of the slice
    under the pointer (POINTER_ANGLE). Works with direction='clockwise', sort=False.
    """
    alpha = (POINTER_ANGLE - (rotation_deg % 360.0)) % 360.0  # angle seen by pointer
    # find the first cumulative end >= alpha
    idx = int(np.searchsorted(cum, alpha, side="right"))
    if idx >= len(labels):
        idx = len(labels) - 1
    return idx

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
    fig.update_layout(width=WIDTH, height=HEIGHT, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    
    # Pointer head (triangle) tangent at right edge
    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
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

# ---- NEW: Smooth, continuously decreasing speed over entire spin ----
def angle_schedule(start_rot: float, final_rot: float, total_time: float, frames: int, power: float):
    """
    Monotonic deceleration from t=0 to t=total_time with omega(t) decreasing to 0.
      theta(t) = start + Δ * (1 - (1 - t/T)^(power+1))
    Properties:
      - omega(t) = Δ * (power+1)/T * (1 - t/T)^power  (strictly decreasing, hits 0 at T)
      - power in [1..4] works nicely (2 is a good default)
    """
    T = max(1e-9, total_time)
    n = max(0.0, power)
    times = np.linspace(0.0, T, frames)
    frac = 1.0 - np.power(1.0 - times / T, n + 1.0)  # 0->1, concave, smooth
    return list(start_rot + (final_rot - start_rot) * frac)

# ---- UI ----
col1, col2 = st.columns([1, 1])
clicked = col1.button("🎲 Spin", use_container_width=True)

if clicked:
    st.session_state.spin_id += 1
    rng = np.random.default_rng(None if SEED == 0 else SEED)

    idx = rng.choice(len(labels), p=fractions)
    #st.session_state.result = labels[idx]

    if SNAP_TO_CENTER:
        target_theta = centers[idx]
    else:
        target_theta = rng.uniform(starts[idx], cum[idx])

    spins_effective = SPINS * rng.uniform(*RANDOM_SPIN_TIME_RANGE)

    start_rot = float(st.session_state.rotation)
    final_rot = rotation_to_align(target_theta, start_rot, spins_effective)

    spin_time_effective = SPIN_TIME

    # Predicting the result before it happens
    predicted_idx = visible_index(final_rot % 360.0)
    predicted_result = labels[predicted_idx]
    
    # (optional) show a spoiler:
    #st.info(f"Incoming: **{predicted_result}**")  # or gate behind a checkbox

    # Build continuously-decelerating schedule (no flat phase)
    angles = angle_schedule(start_rot, final_rot, spin_time_effective, FRAMES, DECEL_POWER)

    for i, rot in enumerate(angles):
        frame_key = f"spin_{st.session_state.spin_id}_{i}"
        slot.plotly_chart(wheel_fig(rot), use_container_width=False, key=frame_key)
        time.sleep(spin_time_effective / max(FRAMES, 1))

    # Save final rotation and compute visible result from where we actually landed
    st.session_state.rotation = final_rot % 360.0
    result_idx = visible_index(st.session_state.rotation)
    st.session_state.result = labels[result_idx]

    st.rerun()

# Idle wheel
slot.plotly_chart(wheel_fig(st.session_state.rotation), use_container_width=False, key="idle_wheel")

# Result display
if st.session_state.result:
    col2.success(f"Result: **{st.session_state.result}**")


