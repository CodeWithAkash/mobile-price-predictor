# 📱 Mobile Price Range Prediction

This machine learning project predicts the price range of a mobile phone (Low, Medium, High, Very High) based on its features using a classification model.

---

## 🧠 Problem Statement

Build a model that takes in various features of mobile phones like RAM, battery power, camera specs, etc., and predicts the price category:

- `0`: Low Cost  
- `1`: Medium Cost  
- `2`: High Cost  
- `3`: Very High Cost

---

## 📁 Dataset

The dataset used is `mobiledata.csv`, which includes 2000 records with the following 21 columns:

- **Features**:  
  `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`,  
  `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, `wifi`

- **Target**:  
  `price_range` (0–3)

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit (for web app)

---

## 🔍 Model Details

- **Model**: Random Forest Classifier
- **Preprocessing**: Feature scaling using StandardScaler
- **Evaluation**: Accuracy Score, Confusion Matrix, Classification Report

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
