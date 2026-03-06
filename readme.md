# 🧬 Protein Secondary Structure Prediction System

An **end-to-end machine learning system** for predicting **protein secondary structure (Q3)** directly from amino acid sequences, with interactive visualization and interpretability tools.

This project explores how **deep learning models can learn local structural patterns from protein sequences** and exposes the prediction process through an **interactive web interface**.

---

# 📌 Overview

Determining the **3D structure of proteins experimentally** (via X-ray crystallography, NMR spectroscopy, or Cryo-EM) is expensive and time-consuming.

However, useful insights can often be obtained from **secondary structure prediction**, where each amino acid residue is classified into one of three structural states.

| Label | Structure   |
| ----- | ----------- |
| **H** | Alpha Helix |
| **E** | Beta Sheet  |
| **C** | Coil / Loop |

This project builds a **sequence-to-structure prediction pipeline** that learns structural tendencies directly from protein sequences.

---

# 🧠 Key Idea

The system predicts the **secondary structure for each residue in a protein sequence** using a deep learning model.

### Prediction Pipeline

```
Protein Sequence
      ↓
Integer Encoding
      ↓
Embedding Representation
      ↓
Sliding Window Context Modeling
      ↓
Bidirectional LSTM (BiLSTM)
      ↓
Per-Residue Q3 Structure Prediction
```

Instead of producing only a sequence of labels, the system also provides **probability distributions and structural visualizations**, making predictions easier to interpret.

---

# 🚀 Features

## 🔬 Sequence-Based Structure Prediction

* Predicts secondary structure directly from amino acid sequences
* Does not require evolutionary profiles or multiple sequence alignments

## 🧠 Deep Learning Sequence Model

* Bidirectional LSTM architecture
* Captures local structural dependencies
* Sliding window context modeling

## 📊 Interactive Visualization

The system exposes model predictions through multiple views:

* Linear secondary structure maps
* Helix / Sheet / Coil composition
* Residue-level confidence plots
* Domain-style segmentation
* Residue heatmaps
* 3D probability trajectory visualization

## 🌐 Web Interface

An interactive **Streamlit application** allows users to:

* Input protein sequences
* Visualize predicted secondary structures
* Explore prediction confidence

---

# 🏗 System Architecture

```
Input Sequence
      │
      ▼
Sequence Encoding
      │
      ▼
Sliding Window Feature Extraction
      │
      ▼
BiLSTM Model
      │
      ▼
Softmax Prediction (H / E / C)
      │
      ▼
Visualization & Interactive UI
```

---

# 🧬 Dataset

The model is trained using **protein sequences with DSSP-derived secondary structure labels** extracted from experimentally solved protein structures in the **Protein Data Bank (PDB)**.

Each residue is assigned a **Q3 secondary structure label**:

```
H → Helix
E → Beta Strand
C → Coil
```

---

# 🧠 Machine Learning Model

## Model Architecture

| Layer               | Description                                   |
| ------------------- | --------------------------------------------- |
| **Embedding**       | Converts encoded amino acids to dense vectors |
| **BiLSTM**          | Captures bidirectional sequence dependencies  |
| **Dense + Softmax** | Predicts probability of H / E / C             |

---

## Prediction Type

Per-residue classification:

```
Sequence:   M N I D S T
Prediction: C C H H C C
```

---

## Training Objective

**Loss Function**

```
Sparse Categorical Crossentropy
```

---

## Model Output

For each residue, the model predicts:

```
P(Helix)
P(Sheet)
P(Coil)
```

This allows the system to:

* Identify ambiguous regions
* Detect structural transitions
* Visualize prediction confidence

---

# 📈 Visualization Components

The interactive UI includes several visualization modules.

## Linear Structure Map

Color-coded sequence showing predicted structures.

```
Sequence:  MNIDST
Structure: CCHHCC
```

---

## Composition Analysis

Pie chart showing proportions of:

* Helices
* Sheets
* Coils

---

## Confidence Plot

Residue-wise probability curves for each structure type.

---

## Domain-Style Segmentation

Highlights dominant structure regions in the sequence.

---

## Residue Heatmap

Visual representation of structure states across the sequence.

---

## 3D Probability Trajectory

A 3D visualization of prediction probabilities in **H–E–C space**, showing how model confidence evolves across the protein.

---

# 💻 Tech Stack

## Programming

* Python

## Machine Learning

* TensorFlow
* Keras

## Data Processing

* NumPy
* Pandas

## Visualization

* Plotly
* Matplotlib

## Web Interface

* Streamlit
* HTML
* CSS

---

# ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/protein-secondary-structure-prediction.git
```

Navigate to the project directory:

```bash
cd protein-secondary-structure-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

Then open the **local URL displayed in the terminal**.

---

# 📌 Example Input

```
MNIDSTKAVLEQLKDLG
```

The application will output predicted secondary structures and visualizations.

---

# ⚠ Limitations

* Uses **sequence-only features**
* Long-range interactions may not be fully captured
* Accuracy is lower than predictors using **evolutionary profiles (MSA)**

However, sequence-only models are useful when:

* homologous sequences are unavailable
* rapid approximate prediction is needed

---

# 🎓 Learning Outcomes

This project demonstrates:

* Deep learning for biological sequence modeling
* Per-residue classification systems
* ML interpretability for sequence models
* Building end-to-end ML systems beyond notebooks
* Deploying ML models through interactive interfaces

---