# Baseline code written by my own brain and hands
# this code is baseline for all the further test subject apps 
# test subject apps are all modifications: having code improved by multiple AI's and prompts and flow of improvement is described by my brain  

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# page configuration ------------------------------------------------------
st.set_page_config(
    page_title="Protein Secondary Structure Predictor (Q3)",
    layout="wide"
)

# global plot style  ------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "#AAAAAA",
    "axes.labelcolor": "#DDDDDD",
    "text.color": "#DDDDDD",
    "xtick.color": "#BBBBBB",
    "ytick.color": "#BBBBBB",
})


# loading model and metadata ------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model/protein_bilstm_q3.keras")

    with open("model/aa_id.pkl", "rb") as f:
        aa_id = pickle.load(f)

    with open("model/id_ss.pkl", "rb") as f:
        id_ss = pickle.load(f)

    with open("model/meta.pkl", "rb") as f:
        meta = pickle.load(f)

    return model, aa_id, id_ss, meta


model, aa_id, id_ss, meta = load_artifacts()

WINDOW_SIZE = meta["window_size"]
PAD_ID = meta.get("pad_ID", 0)
HALF_W = WINDOW_SIZE // 2



# utility functions --------------------------------------------------------
def validate_sequence(seq):
    valid_aa = set(aa_id.keys())
    invalid = set(seq) - valid_aa
    return len(invalid) == 0, invalid


def encode_sequence(seq):
    return [aa_id[a] for a in seq]


def predict_with_confidence(seq):
    encoded = encode_sequence(seq)
    preds, probs = [], []

    for i in range(len(encoded)):
        left = max(0, i - HALF_W)
        right = min(len(encoded), i + HALF_W + 1)
        window = encoded[left:right]

        if len(window) < WINDOW_SIZE:
            pad_left = max(0, HALF_W - i)
            pad_right = WINDOW_SIZE - len(window) - pad_left
            window = [PAD_ID] * pad_left + window + [PAD_ID] * pad_right

        window = np.array(window).reshape(1, -1)
        prob = model.predict(window, verbose=0)[0]

        preds.append(np.argmax(prob))
        probs.append(prob)

    sst3 = "".join(id_ss[p] for p in preds)
    return sst3, np.array(probs)


def color_sst3(sst3):
    colors = {"H": "#e74c3c", "E": "#3498db", "C": "#2ecc71"}
    return "".join(
        f"<span style='color:{colors[c]}; font-weight:600'>{c}</span>"
        for c in sst3
    )


def segment_domains(sst3, window=20):
    segments = []
    for i in range(0, len(sst3), window):
        chunk = sst3[i:i + window]
        majority = max(set(chunk), key=chunk.count)
        segments.append((i, i + len(chunk), majority))
    return segments

# sidebar
st.sidebar.title("Model Information")

with st.sidebar.expander("Model Architecture"):
    st.markdown(f"""
- **Type:** BiLSTM (Sliding Window)
- **Window Size:** {WINDOW_SIZE}
- **Output:** Q3 (H, E, C)
- **Loss:** Sparse Categorical Crossentropy
- **Training Data:** DSSP-derived PDB structures
""")

with st.sidebar.expander("About"):
    st.markdown("""
### Secondary Structure
- **H** → Alpha helix  
- **E** → Beta sheet  
- **C** → Coil / loop  

### Why Sliding Window?
To focus on **local** motifs within a protein sequnce, allowing 
more accurate secondary structure predictions without requiring 
the entire sequence in the memory.

### Limitations
- ❌ No 3D coordinates
- ❌ No tertiary structure
""")

# main UI
st.title("🧬 Protein Secondary Structure Prediction (Q3)")

sequence = st.text_area(
    "Enter Amino Acid Sequence (FASTA without header):",
    value="MNIDSTKAVLEQLKDLG",
    height=150
).strip().upper()

if sequence:
    valid, invalid = validate_sequence(sequence)

    if not valid:
        st.error(f"Invalid amino acids found: {', '.join(invalid)}")
        st.stop()

    st.success("Valid protein sequence detected.")

    # basic stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Sequence Length", len(sequence))
    col2.metric("Unique Residues", len(set(sequence)))
    col3.metric(
        "Hydrophobic %",
        round(sum(sequence.count(a) for a in "AILMFWV") / len(sequence) * 100, 2)
    )

    # prediction
    with st.spinner("Running model inference..."):
        sst3, probs = predict_with_confidence(sequence)

    # output tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Linear Map",
        "Composition",
        "Confidence Plot",
        "Domain Segments",
        "FASTA Annotation",
        "Visualizations"
    ])

    # tab 1 : linear map
    with tab1:
        st.subheader("Linear Secondary Structure Map")
        st.markdown(color_sst3(sst3), unsafe_allow_html=True)

    # tab 2: composition
    with tab2:
        counts = pd.Series(list(sst3)).value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_map={
                "H": "#e74c3c",
                "E": "#3498db",
                "C": "#2ecc71"
            },
            hole=0.4
        )
        fig.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # tab 3 : confidence plot
    with tab3:
        df_prob = pd.DataFrame(
            probs,
            columns=["Helix (H)", "Sheet (E)", "Coil (C)"]
        )
        df_prob["Residue"] = np.arange(1, len(df_prob) + 1)

        fig = px.line(
            df_prob,
            x="Residue",
            y=["Helix (H)", "Sheet (E)", "Coil (C)"],
            height=350
        )
        fig.update_layout(
            yaxis_title="Probability",
            legend_title="Structure",
            margin=dict(t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # tab 4 : domain segmentation
    with tab4:
        segments = segment_domains(sst3)
        for start, end, label in segments:
            st.markdown(f"**Residues {start+1}–{end}:** Dominant → `{label}`")

    # tab 5 : fasta annotation
    with tab5:
        fasta = f">Predicted_Protein\n{sequence}\n\n>SST3\n{sst3}"
        st.code(fasta)
        st.download_button(
            "Download Annotated FASTA",
            fasta,
            file_name="predicted_sst3.fasta"
        )

    # tab 6 : advanced visualizations
    with tab6:
        st.subheader("Residue-wise Secondary Structure Heatmap")

        sst_map = {"H": 0, "E": 1, "C": 2}
        heat_vals = [sst_map[c] for c in sst3]

        fig = px.imshow(
            [heat_vals],
            aspect="auto",
            color_continuous_scale=["#e74c3c", "#3498db", "#2ecc71"],
            labels=dict(x="Residue Index", color="Structure")
        )
        fig.update_layout(height=200, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("3D Probability Trajectory (H–E–C Space)")

        fig = go.Figure(
            data=go.Scatter3d(
                x=probs[:, 0],
                y=probs[:, 1],
                z=probs[:, 2],
                mode="lines+markers",
                marker=dict(
                    size=3,
                    color=np.arange(len(probs)),
                    colorscale="Viridis"
                )
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="P(Helix)",
                yaxis_title="P(Sheet)",
                zaxis_title="P(Coil)"
            ),
            height=400,
            margin=dict(t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Amino Acid vs Predicted Structure Association")

        df_assoc = pd.DataFrame({
            "AA": list(sequence),
            "SST3": list(sst3)
        })

        pivot = pd.crosstab(df_assoc["AA"], df_assoc["SST3"])

        fig = px.imshow(
            pivot,
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig.update_layout(height=350, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
