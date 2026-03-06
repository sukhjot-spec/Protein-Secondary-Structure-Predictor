# this app has many deafaults: load medium is not working, many static elements having zero mathematical logic, advanced analysis is broken at some places

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, List, Set
import io
from datetime import datetime
import time

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Protein Structure Predictor | ML Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    
    /* Container separation with shadows */
    .container-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
    }
    
    /* Section headings - different styles */
    .section-heading {
        font-size: 1.4em;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 3px solid rgba(102, 126, 234, 0.5);
        letter-spacing: 0.5px;
    }
    
    .subsection-heading {
        font-size: 1.1em;
        font-weight: 600;
        color: #e0e0e0;
        margin-top: 15px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        opacity: 0.85;
    }
    
    .tab-heading {
        font-size: 1.15em;
        font-weight: 600;
        color: #f0f0f0;
        margin-bottom: 12px;
        padding-left: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Configure matplotlib for dark theme
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


# MODEL LOADING & CACHING
@st.cache_resource
def load_model_artifacts() -> Tuple:
    """
    Load pre-trained BiLSTM model and metadata with caching.
    
    Returns:
        Tuple: (model, aa_id_dict, id_ss_dict, metadata_dict)
    """
    try:
        model = tf.keras.models.load_model("model/protein_bilstm_q3.keras")
        
        with open("model/aa_id.pkl", "rb") as f:
            aa_id = pickle.load(f)
        
        with open("model/id_ss.pkl", "rb") as f:
            id_ss = pickle.load(f)
        
        with open("model/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        
        return model, aa_id, id_ss, meta
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()


# Load model and metadata
model, aa_id, id_ss, meta = load_model_artifacts()

WINDOW_SIZE = meta["window_size"]
PAD_ID = meta.get("pad_ID", 0)
HALF_W = WINDOW_SIZE // 2

# Structure color scheme
STRUCTURE_COLORS = {
    "H": {"hex": "#e74c3c", "name": "Helix", "description": "Alpha helix"},
    "E": {"hex": "#3498db", "name": "Sheet", "description": "Beta sheet"},
    "C": {"hex": "#2ecc71", "name": "Coil", "description": "Coil / loop"}
}


# UTILITY FUNCTIONS
def validate_sequence(seq: str) -> Tuple[bool, Set[str]]:
    """
    Validate amino acid sequence.
    
    Args:
        seq: Protein sequence string
        
    Returns:
        Tuple: (is_valid, set_of_invalid_chars)
    """
    valid_aa = set(aa_id.keys())
    invalid = set(seq) - valid_aa
    return len(invalid) == 0, invalid


def encode_sequence(seq: str) -> List[int]:
    """Encode amino acid sequence to integer IDs."""
    return [aa_id[a] for a in seq]


def predict_with_confidence(seq: str) -> Tuple[str, np.ndarray, float]:
    """
    Predict secondary structure with confidence scores using sliding window.
    
    Args:
        seq: Protein sequence string
        
    Returns:
        Tuple: (predicted_structure, probability_matrix, processing_time_seconds)
    """
    start_time = time.time()
    
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
    processing_time = time.time() - start_time
    
    return sst3, np.array(probs), processing_time


def get_max_confidence(probs: np.ndarray) -> float:
    """Calculate maximum confidence score across all predictions."""
    return float(np.max(probs))


def get_avg_confidence(probs: np.ndarray) -> float:
    """Calculate average confidence score."""
    return float(np.mean(np.max(probs, axis=1)))


def color_sst3(sst3: str) -> str:
    """Generate colored HTML representation of secondary structure."""
    return "".join(
        f"<span style='color:{STRUCTURE_COLORS[c]['hex']}; font-weight:600; font-size:16px'>{c}</span>"
        for c in sst3
    )


def segment_domains(sst3: str, window: int = 20) -> List[Tuple[int, int, str]]:
    """
    Segment protein into domains based on dominant secondary structure.
    
    Args:
        sst3: Secondary structure string
        window: Window size for segmentation
        
    Returns:
        List of (start, end, dominant_structure) tuples
    """
    segments = []
    for i in range(0, len(sst3), window):
        chunk = sst3[i:i + window]
        majority = max(set(chunk), key=chunk.count)
        segments.append((i, i + len(chunk), majority))
    return segments


def compute_structure_statistics(sst3: str) -> Dict[str, float]:
    """Compute detailed statistics for secondary structure."""
    total = len(sst3)
    stats = {}
    
    for code, info in STRUCTURE_COLORS.items():
        count = sst3.count(code)
        percentage = (count / total * 100) if total > 0 else 0
        stats[code] = {
            "count": count,
            "percentage": round(percentage, 2),
            "name": info["name"]
        }
    
    return stats


def compute_aa_statistics(seq: str) -> Dict[str, float]:
    """Compute amino acid composition statistics."""
    total = len(seq)
    aa_counts = pd.Series(list(seq)).value_counts().to_dict()
    
    stats = {
        "total_residues": total,
        "unique_residues": len(set(seq)),
        "hydrophobic_%": round(
            sum(seq.count(a) for a in "AILMFWV") / total * 100, 2
        ),
        "polar_%": round(
            sum(seq.count(a) for a in "STNQ") / total * 100, 2
        ),
        "charged_%": round(
            sum(seq.count(a) for a in "DEKR") / total * 100, 2
        ),
        "composition": aa_counts
    }
    
    return stats


def export_results(seq: str, sst3: str, probs: np.ndarray) -> str:
    """Generate exportable results in FASTA format with annotations."""
    fasta = f">Predicted_Protein\n{seq}\n\n>SST3_Prediction\n{sst3}\n"
    
    # Add confidence scores
    confidence = np.max(probs, axis=1)
    fasta += f"\n>Confidence_Scores\n{''.join([f'{c:.2f},' for c in confidence])}"
    
    return fasta


# SIDEBAR - MODEL INFO & DOCUMENTATION
with st.sidebar:
    st.title("🧬 ML Platform")
    
    st.markdown("---")
    
    # Model Information
    with st.expander("📊 Model Architecture", expanded=False):
        st.markdown(f"""
**BiLSTM Neural Network**
- **Architecture:** Bidirectional LSTM
- **Window Size:** {WINDOW_SIZE} residues
- **Input:** Amino acid sequences
- **Output:** Q3 classification (H, E, C)
- **Loss Function:** Sparse Categorical Crossentropy
- **Training Data:** DSSP-annotated PDB structures
- **Accuracy:** Trained on large-scale protein databases
        """)
    
    # Secondary Structure Guide
    with st.expander("📖 Secondary Structure Guide", expanded=False):
        for code, info in STRUCTURE_COLORS.items():
            st.markdown(f"""
**{code} - {info['name']}**
- {info['description']}
- Color indicator: <span style='color:{info['hex']}'>■</span>
            """, unsafe_allow_html=True)
    
    # Technical Details
    with st.expander("⚙️ Technical Details", expanded=False):
        st.markdown("""
**Why Sliding Window?**
- Focuses on local sequence motifs
- Reduces memory requirements
- Enables accurate local predictions
- Captures structural context

**Confidence Score:**
- Maximum probability from model output
- Higher = more confident prediction
- Used for reliability assessment

**Limitations:**
- No 3D coordinate information
- No tertiary structure modeling
- Sequence-based predictions only
        """)
    
    # About Section
    with st.expander("ℹ️ About This Platform", expanded=False):
        st.markdown("""
**Protein Secondary Structure Predictor (Q3)**

This machine learning platform predicts the secondary structure 
of protein sequences using state-of-the-art BiLSTM neural networks.

**Features:**
- Real-time structure prediction
- Confidence score analysis
- Domain segmentation
- Composition analysis
- Multiple visualization formats
- Batch processing support

**Created for:** Protein structure research and analysis
        """)


# MAIN APPLICATION UI
# Header
st.title("🧬 Protein Secondary Structure Predictor (Q3)")
st.markdown("**ML-powered prediction of protein secondary structures using BiLSTM neural networks**")
st.markdown("---")

# Input Section
st.markdown("<div class='section-heading'>1️⃣ Input Sequence</div>", unsafe_allow_html=True)
st.markdown("<div class='container-box'>", unsafe_allow_html=True)

col1, col2 = st.columns([2.5, 1.5])

with col1:
    sequence = st.text_area(
        "Enter Amino Acid Sequence (uppercase, no spaces):",
        value="MNIDSTKAVLEQLKDLG",
        height=120,
        placeholder="Paste your protein sequence here...",
        help="Enter a protein sequence using standard amino acid codes (A-Z)"
    ).strip().upper()

with col2:
    st.markdown("<div class='subsection-heading'>Quick Load</div>", unsafe_allow_html=True)
    
    example_sequences = {
        "Small (17 AA)": "MNIDSTKAVLEQLKDLG",
        "Medium (54 AA)": "MKVLIVGAGPNASVVIQVGDLFRPIGFGQPQIGKEVDLVDIQGQGVFGYFDQVGP",
        "Clear": ""
    }
    
    st.markdown("**Preset Examples:**")
    for seq_name, seq_val in example_sequences.items():
        if st.button(f"Load {seq_name}", use_container_width=True):
            sequence = seq_val
            st.rerun()
    
    st.divider()
    
    st.markdown("**Info:**")
    if sequence:
        st.info(f"📝 **Sequence Length:** {len(sequence)} residues")
        st.caption(f"Valid amino acids: {', '.join(sorted(aa_id.keys()))}")

st.markdown("</div>", unsafe_allow_html=True)

# Validate and Process
if sequence:
    valid, invalid = validate_sequence(sequence)
    
    if not valid:
        st.error(f"❌ Invalid amino acids detected: {', '.join(sorted(invalid))}")
        st.info(f"Valid amino acids: {', '.join(sorted(aa_id.keys()))}")
        st.stop()
    
    st.success("✅ Valid protein sequence detected")
    

    # STATISTICS & ANALYSIS    
    st.markdown("<div class='section-heading'>2️⃣ Sequence Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='container-box'>", unsafe_allow_html=True)
    
    aa_stats = compute_aa_statistics(sequence)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📏 Length", aa_stats["total_residues"])
    col2.metric("🔢 Unique AA", aa_stats["unique_residues"])
    col3.metric("🌊 Hydrophobic %", f"{aa_stats['hydrophobic_%']}%")
    col4.metric("💧 Polar %", f"{aa_stats['polar_%']}%")
    col5.metric("⚡ Charged %", f"{aa_stats['charged_%']}%")
    
    st.markdown("</div>", unsafe_allow_html=True)
    

    # PREDICTION    
    st.markdown("<div class='section-heading'>3️⃣ Structure Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='container-box'>", unsafe_allow_html=True)
    
    with st.spinner("🔬 Running BiLSTM model inference..."):
        sst3, probs, processing_time = predict_with_confidence(sequence)
    
    # Calculate metrics
    ss_stats = compute_structure_statistics(sst3)
    avg_conf = get_avg_confidence(probs)
    max_conf = get_max_confidence(probs)
    
    st.success("✅ Prediction complete!")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Avg. Confidence", f"{avg_conf:.2%}")
    col2.metric("📈 Max. Confidence", f"{max_conf:.2%}")
    col3.metric("⏱️ Processing Time", f"{processing_time:.3f}s")
    col4.metric("🧠 Model", "BiLSTM Q3")
    
    st.markdown("</div>", unsafe_allow_html=True)
    

    # RESULTS DISPLAY - TABBED INTERFACE    
    st.markdown("<div class='section-heading'>4️⃣ Results & Visualizations</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗺️ Linear Map",
        "📊 Composition",
        "📈 Confidence",
        "🎯 Domain Segments",
        "📥 Export",
        "🔬 Advanced Analysis",
        "📋 Details"
    ])
    
    # TAB 1: Linear Secondary Structure Map
    with tab1:
        st.markdown("<div class='tab-heading'>📍 Residue-by-Residue Structure Map</div>", unsafe_allow_html=True)
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        
        st.markdown(
            "<div style='font-size: 18px; letter-spacing: 2px; padding: 20px; background-color: rgba(100,100,100,0.15); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 2px 8px rgba(0,0,0,0.2);'>" +
            color_sst3(sst3) +
            "</div>",
            unsafe_allow_html=True
        )
        
        # Breakdown
        st.markdown("<div class='subsection-heading'>📊 Structure Breakdown</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, (code, info) in enumerate(STRUCTURE_COLORS.items()):
            with cols[idx]:
                count = ss_stats[code]["count"]
                pct = ss_stats[code]["percentage"]
                st.markdown(f"""
<div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); padding: 15px; border-left: 4px solid {info['hex']}; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);'>
<b style='color: {info['hex']}'>{info['name']} ({code})</b><br>
<span style='font-size: 0.9em; opacity: 0.8'>{count} residues ({pct}%)</span>
</div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 2: Structure Composition Pie Chart
    with tab2:
        st.markdown("<div class='tab-heading'>🥧 Secondary Structure Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        
        comp_data = {
            "name": [STRUCTURE_COLORS[code]["name"] for code in ["H", "E", "C"]],
            "count": [ss_stats[code]["count"] for code in ["H", "E", "C"]],
            "percentage": [ss_stats[code]["percentage"] for code in ["H", "E", "C"]]
        }
        
        fig = px.pie(
            values=comp_data["count"],
            names=comp_data["name"],
            color=comp_data["name"],
            color_discrete_map={
                STRUCTURE_COLORS["H"]["name"]: STRUCTURE_COLORS["H"]["hex"],
                STRUCTURE_COLORS["E"]["name"]: STRUCTURE_COLORS["E"]["hex"],
                STRUCTURE_COLORS["C"]["name"]: STRUCTURE_COLORS["C"]["hex"]
            },
            hole=0.35
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=500,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 3: Confidence Score Line Plot
    with tab3:
        st.markdown("<div class='tab-heading'>📈 Model Confidence Scores</div>", unsafe_allow_html=True)
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        
        df_prob = pd.DataFrame(
            probs,
            columns=[
                STRUCTURE_COLORS["H"]["name"],
                STRUCTURE_COLORS["E"]["name"],
                STRUCTURE_COLORS["C"]["name"]
            ]
        )
        df_prob["Residue"] = np.arange(1, len(df_prob) + 1)
        
        fig = px.line(
            df_prob,
            x="Residue",
            y=list(df_prob.columns[:-1]),
            height=600,
            color_discrete_map={
                STRUCTURE_COLORS["H"]["name"]: STRUCTURE_COLORS["H"]["hex"],
                STRUCTURE_COLORS["E"]["name"]: STRUCTURE_COLORS["E"]["hex"],
                STRUCTURE_COLORS["C"]["name"]: STRUCTURE_COLORS["C"]["hex"]
            }
        )
        fig.update_layout(
            yaxis_title="Prediction Probability",
            xaxis_title="Residue Position",
            legend_title="Structure Type",
            hovermode="x unified",
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 4: Domain Segmentation
    with tab4:
        st.markdown("<div class='tab-heading'>🧩 Domain Segmentation</div>", unsafe_allow_html=True)
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        
        segments = segment_domains(sst3, window=15)
        
        seg_data = []
        for start, end, structure_type in segments:
            seg_data.append({
                "Start": start + 1,
                "End": end,
                "Length": end - start,
                "Dominant": structure_type,
                "Structure": STRUCTURE_COLORS[structure_type]["name"]
            })
        
        df_segments = pd.DataFrame(seg_data)
        st.dataframe(
            df_segments,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Dominant": st.column_config.TextColumn(disabled=True),
                "Start": st.column_config.NumberColumn(format="%d"),
                "End": st.column_config.NumberColumn(format="%d"),
                "Length": st.column_config.NumberColumn(format="%d")
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 5: Export Results
    with tab5:
        st.markdown("<div class='tab-heading'>💾 Export Results</div>", unsafe_allow_html=True)
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        
        # FASTA Format
        fasta_output = export_results(sequence, sst3, probs)
        
        st.text_area(
            "FASTA Format:",
            fasta_output,
            height=300,
            disabled=True
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download FASTA",
                fasta_output,
                file_name=f"protein_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.fasta",
                mime="text/plain",
                use_container_width=True
            )
        
        # CSV Export
        with col2:
            csv_buffer = io.StringIO()
            csv_buffer.write("Position,Residue,Structure,H_Prob,E_Prob,C_Prob\n")
            for i, (aa, ss) in enumerate(zip(sequence, sst3)):
                csv_buffer.write(
                    f"{i+1},{aa},{ss},{probs[i,0]:.4f},{probs[i,1]:.4f},{probs[i,2]:.4f}\n"
                )
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                "📊 Download CSV",
                csv_data,
                file_name=f"protein_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # JSON Export
        with col3:
            import json
            json_data = {
                "sequence": sequence,
                "prediction": sst3,
                "statistics": ss_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            st.download_button(
                "📄 Download JSON",
                json.dumps(json_data, indent=2),
                file_name=f"protein_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 6: Advanced Visualizations
    with tab6:
        st.markdown("<div class='tab-heading'>🔬 Advanced Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='subsection-heading'>🔥 Residue-wise Heatmap</div>", unsafe_allow_html=True)
            
            sst_map = {"H": 0, "E": 1, "C": 2}
            heat_vals = [sst_map[c] for c in sst3]
            
            fig = px.imshow(
                [heat_vals],
                aspect="auto",
                color_continuous_scale=[
                    STRUCTURE_COLORS["H"]["hex"],
                    STRUCTURE_COLORS["E"]["hex"],
                    STRUCTURE_COLORS["C"]["hex"]
                ],
                labels=dict(x="Residue Index", color="Structure"),
                height=350
            )
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<div class='subsection-heading'>📊 Confidence Distribution</div>", unsafe_allow_html=True)
            
            max_probs = np.max(probs, axis=1)
            fig = px.histogram(
                x=max_probs,
                nbins=30,
                labels={"x": "Confidence Score"},
                height=350
            )
            fig.add_vline(
                x=np.mean(max_probs),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {np.mean(max_probs):.2%}"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div class='subsection-heading'>🎲 3D Probability Space</div>", unsafe_allow_html=True)
        
        fig = go.Figure(
            data=go.Scatter3d(
                x=probs[:, 0],
                y=probs[:, 1],
                z=probs[:, 2],
                mode="lines+markers",
                marker=dict(
                    size=4,
                    color=np.arange(len(probs)),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Position")
                ),
                line=dict(color="rgba(150,150,150,0.5)", width=2)
            )
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f"P({STRUCTURE_COLORS['H']['name']})",
                yaxis_title=f"P({STRUCTURE_COLORS['E']['name']})",
                zaxis_title=f"P({STRUCTURE_COLORS['C']['name']})",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            height=500,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div class='subsection-heading'>🧬 Amino Acid-Structure Association</div>", unsafe_allow_html=True)
        
        df_assoc = pd.DataFrame({
            "Amino Acid": list(sequence),
            "Structure": list(sst3)
        })
        
        pivot = pd.crosstab(df_assoc["Amino Acid"], df_assoc["Structure"])
        
        fig = px.imshow(
            pivot,
            color_continuous_scale="Blues",
            aspect="auto",
            height=400
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 7: Detailed Information
    with tab7:
        st.markdown("<div class='tab-heading'>📋 Comprehensive Details</div>", unsafe_allow_html=True)
        st.markdown("<div class='container-box'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='subsection-heading'>📝 Sequence Information</div>", unsafe_allow_html=True)
            seq_info = {
                "Total Residues": aa_stats["total_residues"],
                "Unique Amino Acids": aa_stats["unique_residues"],
                "Hydrophobic %": f"{aa_stats['hydrophobic_%']}%",
                "Polar %": f"{aa_stats['polar_%']}%",
                "Charged %": f"{aa_stats['charged_%']}%",
                "GC-like Content": f"{(aa_stats.get('composition', {}).get('G', 0) + aa_stats.get('composition', {}).get('C', 0)) / aa_stats['total_residues'] * 100:.2f}%"
            }
            
            for key, value in seq_info.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.markdown("<div class='subsection-heading'>🎯 Prediction Statistics</div>", unsafe_allow_html=True)
            pred_info = {
                "Average Confidence": f"{avg_conf:.4f}",
                "Maximum Confidence": f"{max_conf:.4f}",
                "Minimum Confidence": f"{np.min(probs):.4f}",
                "Std. Deviation": f"{np.std(np.max(probs, axis=1)):.4f}",
                "Prediction Certainty": "High" if avg_conf > 0.7 else "Medium" if avg_conf > 0.5 else "Low"
            }
            
            for key, value in pred_info.items():
                st.write(f"**{key}:** {value}")
        
        st.markdown("<div class='subsection-heading'>📊 Structure Distribution</div>", unsafe_allow_html=True)
        dist_cols = st.columns(3)
        for idx, code in enumerate(["H", "E", "C"]):
            with dist_cols[idx]:
                st.metric(
                    f"{STRUCTURE_COLORS[code]['name']} ({code})",
                    f"{ss_stats[code]['percentage']}%",
                    f"{ss_stats[code]['count']} residues"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👆 Enter a protein sequence above to get started!")
    
    # Display example information
    st.markdown("---")
    st.subheader("📚 How to Use")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
**Step 1: Input**
Enter or paste your protein sequence in standard amino acid codes (A-Z)
        """)
    
    with col2:
        st.markdown("""
**Step 2: Predict**
The BiLSTM model will predict secondary structure in real-time
        """)
    
    with col3:
        st.markdown("""
**Step 3: Analyze**
Explore visualizations and export results
        """)
    
    st.markdown("---")
    st.subheader("📖 Example Sequences")
    
    examples_col1, examples_col2 = st.columns(2)
    
    with examples_col1:
        st.code("MNIDSTKAVLEQLKDLG", language="")
        st.caption("Small example (17 residues)")
    
    with examples_col2:
        st.code("MKVLIVGAGPNASVVIQ", language="")
        st.caption("Another example (17 residues)")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
📊 Protein Secondary Structure Prediction (Q3) | Powered by BiLSTM Neural Networks<br>
🧬 ML Platform for Protein Structure Analysis
</div>
""", unsafe_allow_html=True)
