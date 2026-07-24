# 🧠 Personality Risk & Trait Predictor (Streamlit App)

An interactive Machine Learning web application built with **Streamlit**, **Pandas**, **XGBoost/Scikit-Learn**, and **Plotly** to assess and predict personality traits (Introvert vs. Extrovert) based on daily social habits and psychological indicators.

---

## 🌟 Features
- **Modern UI Design**: Custom dark-mode glassmorphism styling, responsive dual-column layout, and animated badges.
- **Interactive Inputs**: Fine-tune daily habits using sliders, radio toggles, and numeric inputs.
- **Visual Predictions**: Real-time probability chart powered by Plotly and custom behavioral metrics (Solitude Index, Outing Energy, Social Battery).
- **Streamlit Cloud Ready**: Easily deployable to Streamlit Community Cloud directly from GitHub.

---

## 📁 Repository Structure

```text
ML_Project/
├── app.py                      # Main Streamlit web application
├── calibrated_model.joblib     # Pre-trained ML pipeline model
├── requirements.txt            # Python dependencies for deployment
├── .streamlit/
│   └── config.toml             # Streamlit visual theme configuration
├── train.csv                   # Model training dataset
├── code.ipynb                  # Data processing & model training notebook
└── README.md                   # Project documentation
```

---

## 🚀 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   cd YOUR_REPOSITORY_NAME
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deploying to Streamlit Community Cloud via GitHub

Follow these simple steps to host your app online for free:

### Step 1: Push your code to GitHub
1. Initialize git and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Deploy Streamlit Personality Predictor UI"
   ```
2. Create a new repository on [GitHub](https://github.com/new).
3. Connect your repository and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **New app**.
3. Select your GitHub repository (`YOUR_USERNAME/YOUR_REPOSITORY_NAME`), branch (`main`), and set **Main file path** to:
   ```text
   app.py
   ```
4. Click **Deploy!** 🚀
