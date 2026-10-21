# 🛰️ TerraCIL / SatReCIL Web Demo

Interactive Streamlit dashboard for **class-incremental satellite land-cover classification** on the EuroSAT RGB dataset, powered by the **SatReCIL** continual learning pipeline.

---

## 🌟 Key Features

- **Class-Incremental Learning Evaluation**:
  - **Stage 1 (3 classes)**: `Forest`, `Residential`, `River` (Base model trained from scratch)
  - **Stage 2 (4 classes)**: `Forest`, `Residential`, `River`, `Highway` (Expands classification head + Replay Buffer)
  - **Stage 3 (5 classes)**: `Forest`, `Residential`, `River`, `Highway`, `Industrial` (Full multi-stage model)
  - **⚡ Side-by-Side Comparison**: Evaluates all 3 stages simultaneously on the same satellite image to observe forward and backward transfer in real-time.
- **Experience Replay Buffer Inspection**:
  - Live inspection of `checkpoints/*.json` replay buffers.
  - Interactive quota breakdown and distribution charts illustrating how balanced exemplars mitigate catastrophic forgetting.
- **EuroSatCompactCNN Architecture**:
  - Compact custom CNN designed for edge / FPGA deployment (~60,512 backbone parameters, ~60,837 total parameters in Stage 3).
  - 64-dimensional latent embedding visualizer.
  - EuroSAT spectral normalization: $\mu = [0.3444, 0.3803, 0.4078]$, $\sigma = [0.2026, 0.1369, 0.1158]$.
- **Built-in Sample Library**:
  - Includes ready-to-test 64×64 satellite image patches in `sample_images/` for instant evaluation, plus custom file upload support.
- **Self-Contained Checkpoints**:
  - All verified PyTorch model weights (`.pt`) and replay buffer configurations (`.json`) are bundled in `checkpoints/`.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <YOUR_REPO_URL>
cd terracil_web_demo
```

### 2. Set Up Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📁 Repository Structure

```
terracil_web_demo/
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── demo_requirements.txt      # Minimal dependency requirements
├── checkpoints/               # Trained weights & replay buffer states
│   ├── stage1_base_seed42.pt
│   ├── stage1_replay_seed42.json
│   ├── stage1_metadata.json
│   ├── stage2_replay_seed42.pt
│   ├── stage2_replay_seed42.json
│   ├── stage2_metadata.json
│   ├── stage3_replay_seed42.pt
│   ├── stage3_replay_seed42.json
│   └── stage3_metadata.json
├── sample_images/             # Preloaded EuroSAT 64x64 test patches
│   ├── sample_forest.png
│   ├── sample_residential.png
│   ├── sample_river.png
│   ├── sample_highway.png
│   └── sample_industrial.png
├── .gitignore
└── README.md
```
