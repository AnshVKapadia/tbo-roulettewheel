# app.py
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Roulette Spinner", page_icon="🎡", layout="centered")
st.title("🎡 Roulette Wheel Spinner")

# ---- Sidebar: options ----
st.sidebar.header("Wheel Options")
default_labels = "Red, Blue, Green, Yellow, Purple, Orange"
labels_text = st.sidebar.text_area("Labels (comma-separated):", default_labels, height=100)
weights_text = st.sidebar.text_input("Weights (optional, comma-separated, e.g., 1,2,1,...):", "")
seed = st.sidebar.number_input("Seed (0 = random each spin):", value=0, step=1)
snap_to_label_center = st.sidebar.checkbox("Snap final angle to slice center", value=True)
frames = st.sidebar.slider("Spin smoothness (frames):", 10, 60, 30)
spin_time = st.sidebar.slider("Spin duration (seconds):", 1.0, 5.0, 2.0)
spins = st.sidebar.slider("Number of full rotations:", 1, 8, 3)

# ---- Parse labels/weights ----
labels = [s.strip() for s in labels_text.split(",") if s.strip()]
if not labels:
    st.stop()

if weights_text.strip():
    try:
        weights = [float(x.strip()) for x in weights_text.split(",")]
        if len(weights) != len(labels) or any(w < 0 for w in weights):
            st.error("Weights must be nonnegative and match the number of labels.")
            st.stop()
    except Exception:
        st.error("Could not parse weights. Use numbers like: 1, 2, 1.5")
        st.stop()
else:
    weights = [1.0] * len(labels)

weights = np.array(weights, dtype=float)
if weights.sum() <= 0:
    st.error("At least one weight must be positive.")
    st.stop()

# ---- Colors ----
def default_colors(n):
    return [f"hsl({int(360*i/n)}, 70%, 50%)" for i in range(n)]
colors = default_colors(len(labels))

# ---- Session state ----
if "angle" not in st.session_state:
    st.session_state.angle = 0.0
if "result" not in st.session_state:
    st.session_state.result = None
# Create the placeholder ONCE and reuse it every rerun
if "wheel_slot" not in st.session_state:
    st.session_state.wheel_slot = st.empty()
WHEEL_KEY = "wheel_plot"

# ---- Wheel fig ----
def wheel_fig(angle_deg=0):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=weights,
                hole=0.25,
                sort=False,
                direction="clockwise",
                rotation=angle_deg,
                textinfo="label+percent",
                textfont=dict(size=14),
                marker=dict(colors=colors, line=dict(color="white", width=2)),
            )
        ]
    )
    fig.update_layout(width=600, height=600, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    # pointer at 12 o'clock
    fig.add_shape(
        type="path",
        path="M 300 20 L 285 60 L 315 60 Z",
        line=dict(color="black"),
        fillcolor="black",
    )
    return fig

# ---- Slice geometry ----
total = weights.sum()
fractions = weights / total
slice_angles = fractions * 360.0
cumulative = np.cumsum(slice_angles)
starts = np.insert(cumulative[:-1], 0, 0.0)
centers = (starts + cumulative) / 2.0

# ---- UI ----
col1, col2 = st.columns([1, 1])
clicked = col1.button("🎲 Spin", use_container_width=True)

slot = st.session_state.wheel_slot  # reuse the same placeholder every run

if clicked:
    # ensure only one chart exists in this run
    slot.empty()

    rng = np.random.default_rng(None if seed == 0 else seed)
    idx = rng.choice(len(labels), p=fractions)
    chosen = labels[idx]

    target_angle = centers[idx] if snap_to_label_center else rng.uniform(starts[idx], cumulative[idx])

    start = st.session_state.angle % 360.0
    final_angle = (spins * 360.0 + target_angle) % 360.0

    base_diff = (final_angle - start + 540) % 360 - 180
    diff = base_diff + spins * 360.0

    # animate (always same key + same slot)
    for i in range(frames):
        t = i / max(1, frames - 1)
        ease = 1 - (1 - t) ** 2
        angle = start + diff * ease
        slot.plotly_chart(wheel_fig(angle), use_container_width=False, key=WHEEL_KEY)
        time.sleep(spin_time / frames)

    # final snap
    st.session_state.angle = (start + diff) % 360.0
    st.session_state.result = chosen
    slot.plotly_chart(wheel_fig(st.session_state.angle), use_container_width=False, key=WHEEL_KEY)
    st.balloons()
else:
    # idle (only one render in this run)
    slot.plotly_chart(wheel_fig(st.session_state.angle), use_container_width=False, key=WHEEL_KEY)

# ---- Result ----
if st.session_state.result:
    col2.success(f"Result: **{st.session_state.result}**")
