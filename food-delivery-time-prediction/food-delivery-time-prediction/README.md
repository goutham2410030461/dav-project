# Food Delivery Time Prediction — Data Analytics & Visualization

A clean, academic-style web app built with **Python (Flask)** to analyze and predict delivery time based on **Distance**, **Weather**, **Traffic**, and **Order Volume**.

## ✨ Features
- Minimal, easy-to-understand dataset (CSV)
- Visualizations (Plotly): scatter + bar charts
- Simple, interpretable ML pipeline (OneHot + Linear Regression)
- Flask web app with neat & minimal UI
- Ready for GitHub + Render deployment

## 🗂 Project Structure
```
food-delivery-time-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── model.pkl
├── README.md
├── dataset/
│   └── food_delivery.csv
├── templates/
│   ├── index.html
│   ├── visualize.html
│   ├── predict.html
│   └── result.html
└── static/
    └── style.css
```

## 📦 Local Setup (VS Code)
```bash
# 1) (optional) Create and activate a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train the model (creates/updates model.pkl)
python train_model.py

# 4) Run the app
python app.py
# Open: http://127.0.0.1:5000/
```

## 🧠 How the Model Works
We use a very simple and explainable pipeline:
- **OneHotEncoder** for the categorical columns (*Weather*, *Traffic*)
- **Pass-through** for numeric columns (*Distance*, *Order_Volume*)
- **Linear Regression** for prediction

This is ideal for understanding feature effects in viva.

## 📊 Visualizations
- **Scatter:** Delivery Time vs Distance (colored by Traffic)
- **Bar:** Average Delivery Time by Weather
- **Bar:** Average Delivery Time by Traffic

## 🚀 Deploy on Render
1. **Push this folder to GitHub** (create a new repository).
2. Go to **Render.com → New → Web Service**.
3. **Connect your GitHub repo**, pick this project.
4. Use these settings:
   - **Environment:** Python 3.9+
   - **Build Command:**
     ```
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```
     gunicorn app:app
     ```
5. Click **Deploy**. Copy your live URL and share it in your submission.

## ✅ Submission Checklist
- [ ] GitHub repository link
- [ ] Render live website link
- [ ] Screenshots of Visualization & Prediction pages (optional)

---

**Made for academic submission. Keep it simple, clear, and explainable.**
