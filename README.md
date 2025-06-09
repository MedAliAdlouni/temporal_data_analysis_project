
# Sunspot Time Series Analysis (1700–1988)

## 📌 Overview
This project analyzes and models the time series of sunspot activity from 1700 to 1988. The goal is to explore trends, clean and visualize the data, and apply statistical models to forecast future sunspot counts.

## 📊 Methods Used
- Time series decomposition (classical and STL)
- Statistical modeling with SARIMA
- Data cleaning and visualization
- Forecast evaluation and validation

## 📁 Project Structure
```
.
├── main.ipynb          # Jupyter notebook containing the full analysis
├── fonctions.py        # Custom helper functions used in the notebook (assumed)
├── venv/               # Virtual environment (not included in version control)
└── README.md           # Project description
```

## 🚀 Getting Started

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate the environment

- **On Windows (PowerShell)**:
```bash
.env\Scripts\Activate.ps1
```

- **On macOS/Linux**:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install pandas matplotlib numpy statsmodels jupyter
```

### 4. Launch the notebook
```bash
jupyter notebook main.ipynb
```

## 📈 Dataset
The dataset consists of yearly sunspot counts between 1700 and 1988, commonly used in time series forecasting tasks.

## 🧠 Author
Mohammed Ali El Adlouni  
Master MALIA – Université Lyon 2
