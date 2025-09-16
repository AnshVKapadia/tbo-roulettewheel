# app.py
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

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

# ---- Color mapping ----
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
    "brown": "saddlebrown",
}
def choose_colors(labels):
    out = []
    for i, lbl in enumerate(labels):
        key = lbl.lower()
        if key in COLOR_MAP:
            out.append(COLOR_MAP[key])
        else:
            # fallback: evenly spaced hues
            out.append(f"hsl({int(360*i/len(labels))}, 70%, 50%)")
    return out

colors = choose_colors(labels)

# ---- Session state ----
st.session_state.setdefault("angle", 0.0)
st.session_state.setdefault("result", None)
st.session_state.setdefault("slot", st.empty())
st.session_state.setdefault("spin_id", 0)
slot = st.session_state.slot

# ---- Wheel figure ----
def wheel_fig(angle_deg: float = 0.0) -> go.Figure:
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.25,
            sort=False,
            direction="clockwise",
            rotation=angle_deg,
            textinfo="label+percent",
            textfont=dict(size=14),
            marker=dict(colors=colors, line=dict(color="white", width=2)),
        )]
    )
    fig.update_layout(width=600, height=600, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)

    # pointer arrow at 12 o'clock
    fig.add_shape(
        type="line",
        x0=300, y0=20, x1=300, y1=60,
        line=dict(color="white", width=4)
    )
    fig.add_shape(
        type="path",
        path="M 285 60 L 315 60 L 300 90 Z",
        line=dict(color="black"),
        fillcolor="white",
    )
    return fig

# ---- Slice geometry ----
total = float(weights.sum())
fractions = weights / total
slice_angles = fractions * 360.0
cum = np.cumsum(slice_angles)
starts = np.insert(cum[:-1], 0, 0.0)
centers = (starts + cum) / 2.0

# ---- UI ----
col1, col2 = st.columns([1, 1])
clicked = col1.button("🎲 Spin", use_container_width=True)

if clicked:
    st.session_state.spin_id += 1
    rng = np.random.default_rng(None if seed == 0 else seed)
    idx = rng.choice(len(labels), p=fractions)
    chosen = labels[idx]
    target = centers[idx] if snap_to_center else rng.uniform(starts[idx], cum[idx])

    start = st.session_state.angle % 360.0
    final = (spins * 360.0 + target) % 360.0
    base_diff = (final - start + 540) % 360 - 180
    diff = base_diff + spins * 360.0

    # Animate: unique key per frame
    for i in range(frames):
        t = i / max(1, frames - 1)
        ease = 1 - (1 - t) ** 2
        angle = start + diff * ease
        frame_key = f"spin_{st.session_state.spin_id}_{i}"
        slot.plotly_chart(wheel_fig(angle), use_container_width=False, key=frame_key)
        time.sleep(spin_time / frames)

    st.session_state.angle = (start + diff) % 360.0
    st.session_state.result = chosen
    st.rerun()

# Idle wheel
slot.plotly_chart(wheel_fig(st.session_state.angle), use_container_width=False, key="idle_wheel")

if st.session_state.result:
    col2.success(f"Result: **{st.session_state.result}**")
