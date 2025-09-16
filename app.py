# app.py
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import random

st.set_page_config(page_title="Roulette Spinner", page_icon="🎡", layout="centered")
st.title("🎡 Roulette Wheel Spinner")

# ---- Sidebar options ----
st.sidebar.header("Wheel Options")
labels_text = st.sidebar.text_area(
    "Labels (comma-separated):",
    "Red, Blue, Green, Yellow, Purple, Orange",
    height=100
)
weights_text = st.sidebar.text_input("Weights (optional, e.g., 1,2,1,...):", "")
seed = st.sidebar.number_input("Seed (0 = random each spin):", value=0, step=1)
snap_to_center = st.sidebar.checkbox("Snap to slice center", value=True)
frames = st.sidebar.slider("Spin frames:", 10, 60, 30)
spin_time = st.sidebar.slider("Spin duration (seconds):", 1.0, 5.0, 2.0)
spins = st.sidebar.slider("Full rotations:", 1, 8, 3)

# ---- Parse labels/weights ----
labels = [s.strip() for s in labels_text.split(",") if s.strip()]
if not labels:
    st.stop()

if weights_text.strip():
    try:
        weights = [float(x.strip()) for x in weights_text.split(",")]
        if len(weights) != len(labels) or any(w < 0 for w in weights):
            st.error("Weights must be nonnegative and match number of labels.")
            st.stop()
    except Exception:
        st.error("Could not parse weights. Example: 1, 2, 1.5")
        st.stop()
else:
    weights = [1.0] * len(labels)

weights = np.asarray(weights, dtype=float)
if weights.sum() <= 0:
    st.error("At least one weight must be positive.")
    st.stop()

# ---- Color mapping (labels -> intuitive colors, fallback to HSL) ----
COLOR_MAP = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "purple": "purple",
    "orange": "orange",
    "pink": "pink",
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "brown": "saddlebrown",
}
def choose_colors(labels):
    out = []
    n = max(1, len(labels))
    for i, lbl in enumerate(labels):
        key = lbl.lower()
        if key in COLOR_MAP:
            out.append(COLOR_MAP[key])
        else:
            out.append(f"hsl({int(360*i/n)}, 70%, 50%)")
    return out

colors = choose_colors(labels)

# ---- Session state ----
st.session_state.setdefault("rotation", 0.0)   # this is the wheel's rotation in degrees
st.session_state.setdefault("result", None)
st.session_state.setdefault("slot", st.empty())   # single rendering slot
st.session_state.setdefault("spin_id", 0)         # increments per spin (for unique frame keys)
slot = st.session_state.slot

# ---- Geometry helpers ----
total = float(weights.sum())
fractions = weights / total
slice_angles = fractions * 360.0
cum = np.cumsum(slice_angles)
starts = np.insert(cum[:-1], 0, 0.0)       # start angles (deg) measured from 3 o'clock, clockwise
centers = (starts + cum) / 2.0             # center angles (deg)

TOP_ANGLE = 90.0  # 12 o'clock in plotly's coordinate system (0° is 3 o'clock)

# ---- Wheel figure (with visible white arrow pointer in paper coords) ----
def wheel_fig(rotation_deg: float = 0.0) -> go.Figure:
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.25,
            sort=False,
            direction="clockwise",
            rotation=rotation_deg,
            textinfo="label+percent",
            textfont=dict(size=14),
            marker=dict(colors=colors, line=dict(color="white", width=2)),
        )]
    )
    fig.update_layout(
        width=600, height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    # White arrow pointer at 3 o’clock (right side)
    # Shaft
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=1.05, y0=0.5, x1=0.95, y1=0.5,
        line=dict(color="white", width=5),
        layer="above"
    )
    # Arrowhead
    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path="M 0.95 0.47 L 0.95 0.53 L 0.88 0.5 Z",
        line=dict(color="black", width=1.5),
        fillcolor="white",
        layer="above"
    )
    return fig


# ---- Rotation math ----
def rotation_to_align(theta_deg: float, current_rot: float, spins: int) -> float:
    """
    Given a target polar angle theta_deg (center or random point inside the chosen slice),
    return a final rotation so that theta_deg appears at TOP_ANGLE (12 o'clock).
    """
    # desired rotation that would put theta at TOP_ANGLE:
    desired = (TOP_ANGLE - theta_deg) % 360.0
    # ensure multiple full rotations for drama, then align precisely
    # Compute the delta from current rotation to desired (mod 360)
    delta = (desired - (current_rot % 360.0)) % 360.0
    return current_rot + spins * 360.0 + delta

# ---- UI ----
col1, col2 = st.columns([1, 1])
clicked = col1.button("🎲 Spin", use_container_width=True)

if clicked:
    st.session_state.spin_id += 1
    rng = np.random.default_rng(None if seed == 0 else seed)
    # pick a slice by weight
    idx = rng.choice(len(labels), p=fractions)
    st.session_state.result = labels[idx]

    # choose angle within that slice
    if snap_to_center:
        target_theta = centers[idx]
    else:
        target_theta = rng.uniform(starts[idx], cum[idx])

    # compute final rotation that brings target_theta to the top
    start_rot = float(st.session_state.rotation)
    final_rot = rotation_to_align(target_theta, start_rot, spins)

    # Animate rotation with ease-out
    for i in range(frames):
        t = i / max(1, frames - 1)
        ease = 1 - (1 - t) ** 2
        rot = start_rot + (final_rot - start_rot) * ease
        frame_key = f"spin_{st.session_state.spin_id}_{i}"   # unique key per frame
        slot.plotly_chart(wheel_fig(rot), use_container_width=False, key=frame_key)

        time_offset = random.random()-0.5
        print(time_offset)
        time.sleep(spin_time / frames + time_offset)

    # Save final rotation and rerun so only the idle wheel renders once
    st.session_state.rotation = final_rot % 360.0
    st.rerun()

# Idle wheel (single stable key so it never overlaps with animation frames)
slot.plotly_chart(wheel_fig(st.session_state.rotation), use_container_width=False, key="idle_wheel")

# Result display
if st.session_state.result:
    col2.success(f"Result: **{st.session_state.result}**")

