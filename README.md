# Thiranex_project-4
Real-world Data Project (Finance, Health, or Retail) Work on a domain-specific dataset for applied learning.
# 🏥 Patient Health Records — End-to-End Data Science Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

##  Project Overview

This is a **real-world Health Data Science project** that performs end-to-end analysis on a large-scale patient records dataset. The project covers everything from raw data cleaning to building a machine learning model that predicts whether a hospital visit will be an **emergency or not**.

> **Goal:** Apply data science skills in a real-world healthcare context — clean, explore, engineer features, model, and visualize patient visit data to extract meaningful clinical insights.

---

##  Dataset Description

| Attribute | Details |
|---|---|
|  File | `patients_record.csv` |
|  Total Records | 274,592 hospital visits |
|  Unique Patients | 100,000 |
|  Time Period | January 2018 — December 2024 |
|  Features | 8 raw columns → 15 engineered features |

### Columns in Dataset

| Column | Description |
|---|---|
| `patient_id` | Unique patient identifier |
| `visit_date` | Date of hospital visit |
| `visit_type` | Type of visit (outpatient, emergency, telehealth, inpatient) |
| `primary_diagnosis` | Main diagnosis of the visit |
| `primary_icd10` | ICD-10 code for primary diagnosis |
| `secondary_diagnoses` | Additional diagnoses (pipe-separated) |
| `secondary_icd10s` | ICD-10 codes for secondary diagnoses |
| `provider_specialty` | Medical specialty of the treating doctor |

---

## 🔁 Project Pipeline

```
Raw CSV Data
     │
     ▼
1️⃣  Data Loading & Cleaning
     │   → Parse dates, fill missing values, remove noise
     │
     ▼
2️⃣  Exploratory Data Analysis (EDA)
     │   → Visit types, top diagnoses, specialty breakdown
     │
     ▼
3️⃣  Feature Engineering
     │   → Comorbidity count, visit history, prior emergency flag
     │
     ▼
4️⃣  Predictive Modelling
     │   → Random Forest Classifier (Emergency Visit Risk)
     │
     ▼
5️⃣  Visualizations
     │   → 3 figures, 9 charts total
     │
     ▼
6️⃣  Conclusions & Recommendations
```

---

##  Machine Learning Model

**Task:** Binary Classification — Predict if a visit will be an **Emergency**

**Algorithm:** Random Forest Classifier

| Parameter | Value |
|---|---|
| `n_estimators` | 150 |
| `max_depth` | 10 |
| `class_weight` | balanced |
| `test_size` | 20% |
| `random_state` | 42 |

### Features Used

| Feature | Description |
|---|---|
| `diag_enc` | Primary diagnosis (label encoded) |
| `spec_enc` | Provider specialty (label encoded) |
| `n_secondary` | Number of secondary diagnoses (comorbidities) |
| `month` | Month of visit (seasonality) |
| `year` | Year of visit (temporal trend) |
| `visit_count` | Total visits by patient up to that date |
| `prev_emergency` | Whether patient had a prior emergency visit |

---

## 📈 Key Findings

### 🔹 Disease Burden
- **Hypertension** is the #1 diagnosis with 60,505 visits
- **Obesity** (52,401) and **Hyperlipidemia** (46,315) follow closely
- These 3 conditions account for ~58% of all visits — a clear cardiovascular disease pattern

### 🔹 Visit Type Distribution
| Visit Type | Count | Percentage |
|---|---|---|
| Outpatient | 164,581 | 59.9% |
| Emergency | 41,230 | 15.0% |
| Telehealth | 40,973 | 14.9% |
| Inpatient | 27,808 | 10.1% |

### 🔹 Emergency Risk Factors
- Patients with **2+ secondary diagnoses** have significantly higher emergency rates
- **Depression, hyperlipidemia, and CKD** show the highest emergency visit rates
- **Prior emergency history** is the strongest individual predictor

### 🔹 Temporal Trends
- **Telehealth surged after 2020** and now rivals outpatient in volume
- Emergency visits peak slightly in **winter months (Dec–Jan)**

---

## 📉 Visualizations

### Figure 1 — EDA Dashboard
> Visit type distribution, top 10 diagnoses, yearly trends, emergency rates, specialty mix, monthly seasonality, and comorbidity distribution

### Figure 2 — Model Evaluation
> Confusion matrix, ROC curve (AUC score), and feature importance chart

### Figure 3 — Temporal & Comorbidity Analysis
> Quarterly visit volume trend, emergency rate vs secondary diagnoses, visit-type heatmap by specialty

---

## Recommendations

 **Proactive Care Programs** — Enrol high-comorbidity patients in care-coordination to reduce avoidable ED visits

 **Telehealth Follow-ups** — Use ML model predictions to flag high-risk patients for telehealth check-ins

 **Capacity Planning** — Increase nephrology and cardiology resources — both show disproportionate emergency load

 **Seasonal Staffing** — Bolster ED staffing during November–January peak period

---

##  Project Structure

```
project/
│
├── main.py                      # Main analysis script
├── patients_record.csv          # Dataset (place here before running)
├── README.md                    # Project documentation
│
└── outputs/                     # Auto-created on run
    ├── fig1_eda_dashboard.png
    ├── fig2_model_evaluation.png
    └── fig3_temporal_comorbidity.png
```

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/patient-health-analysis.git
cd patient-health-analysis
```

### 2. Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Add the Dataset
Place `patients_record.csv` in the **same folder** as `main.py`

### 4. Run the Script
```bash
python main.py
```

Output charts will be saved automatically in the `outputs/` folder.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Chart plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Machine learning model & evaluation |

---



---

> ⭐ If you found this project helpful, please give it a star on GitHub!
