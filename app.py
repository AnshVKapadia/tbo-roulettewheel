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
seed = st.sidebar.number_input("Seed (optional):", value=0, step=1)
snap_to_label_center = st.sidebar.checkbox("Snap final angle to slice center", value=True)
frames = st.sidebar.slider("Spin smoothness (frames):", 10, 60, 30)
spin_time = st.sidebar.slider("Spin duration (seconds):", 1.0, 5.0, 2.0)
spins = st.sidebar.slider("Number of full rotations:", 1, 8, 3)

# Parse labels
labels = [s.strip() for s in labels_text.split(",") if s.strip()]
if not labels:
    st.stop()

# Parse weights
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

# Make colors (distinct-ish without hardcoding palettes)
def default_colors(n):
    # evenly spaced hues
    return [f"hsl({int(360*i/n)}, 70%, 50%)" for i in range(n)]
colors = default_colors(len(labels))

# Keep session state for angle/result
if "angle" not in st.session_state:
    st.session_state.angle = 0.0
if "result" not in st.session_state:
    st.session_state.result = None

# Helper: build wheel figure
def wheel_fig(angle_deg=0):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=weights,
                hole=0.25,
                sort=False,
                direction="clockwise",
                rotation=angle_deg,            # rotate the wheel
                textinfo="label+percent",
                textfont=dict(size=14),
                marker=dict(colors=colors, line=dict(color="white", width=2)),
            )
        ]
    )
    fig.update_layout(
        width=600,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    # add a pointer at the top
    fig.add_shape(
        type="path",
        path="M 300 20 L 285 60 L 315 60 Z",
        line=dict(color="black"),
        fillcolor="black",
    )
    return fig

# Draw initial wheel
placeholder = st.empty()
placeholder.plotly_chart(wheel_fig(st.session_state.angle), use_container_width=False)

# Compute cumulative slice spans (for snapping / result mapping)
total = weights.sum()
fractions = weights / total
slice_angles = fractions * 360.0  # degrees
cumulative = np.cumsum(slice_angles)  # end angles for each slice
starts = np.insert(cumulative[:-1], 0, 0.0)   # start angle of each slice
centers = (starts + cumulative) / 2.0         # center angles

# Spin button
col1, col2 = st.columns([1, 1])
clicked = col1.button("🎲 Spin", use_container_width=True)

if clicked:
    # Random choice (weighted)
    rng = np.random.default_rng(None if seed == 0 else seed)
    idx = rng.choice(len(labels), p=fractions)
    chosen = labels[idx]

    # We want the pointer (fixed at ~0° at the top) to land on the chosen slice.
    # So rotate the wheel so that the chosen slice center (or some random point inside it)
    # aligns with the pointer at 0° (12 o'clock).
    if snap_to_label_center:
        target_angle = centers[idx]  # degrees
    else:
        # random point inside the slice
        low, high = starts[idx], cumulative[idx]
        target_angle = rng.uniform(low, high)

    # Compute final rotation:
    # We spin 'spins' full turns plus the offset to align chosen segment under the pointer.
    final_angle = (spins * 360.0 + target_angle) % 360.0

    # Animate from current angle to final_angle with easing
    start = st.session_state.angle % 360.0
    end = final_angle
    # choose shortest direction around circle to end
    diff = (end - start + 540) % 360 - 180   # range (-180, 180]
    # add full rotations back in for animation effect
    diff += spins * 360.0
    frame_angles = np.linspace(start, start + diff, frames)

    t0 = time.time()
    for i, a in enumerate(frame_angles):
        # ease-out timing (quadratic)
        t = i / max(1, frames - 1)
        ease = 1 - (1 - t) ** 2
        angle = start + diff * ease
        placeholder.plotly_chart(wheel_fig(angle), use_container_width=False)
        time.sleep(spin_time / frames)

    # Snap to exact final
    st.session_state.angle = (start + diff) % 360.0
    st.session_state.result = chosen
    placeholder.plotly_chart(wheel_fig(st.session_state.angle), use_container_width=False)
    st.balloons()

# Result display
if st.session_state.result:
    col2.success(f"Result: **{st.session_state.result}**")
