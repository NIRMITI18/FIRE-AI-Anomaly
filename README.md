
## 📌 1. Project Overview

This project aims to detect **anomalies in historical sales data** across 7 stores and 10 products using both **statistical** and **machine learning** techniques.
The goal is to identify unusual sales patterns such as sudden spikes, drops, seasonal deviations, or irregular store/product behavior.

The final solution is deployed inside a **Streamlit interactive dashboard** that enables:

* Data exploration
* Visual anomaly detection
* ML-based anomaly flagging
* Store & product-level insights

---

## 🛠️ 2. Tools & Technologies Used

| Category      | Tools                                  |
| ------------- | -------------------------------------- |
| Programming   | Python                                 |
| Libraries     | pandas, numpy, matplotlib, seaborn     |
| ML Model      | Isolation Forest (sklearn)             |
| Preprocessing | StandardScaler                         |
| Visualization | Streamlit Dashboard                    |
| Data          | train.csv (2010–2018), test.csv (2019) |

---

## 🧠 3. Methods & Approach

### A. Data Preprocessing

* Converted date column to datetime
* Extracted Year, Month, Day for ML modeling
* Standardized numeric values using **StandardScaler**
* Handled missing or inconsistent values

---

### B. Exploratory Data Analysis (EDA)

Performed to understand:

* Trends across years
* Seasonality patterns
* Store-wise and product-wise variations
* Visible outliers in sales

Visualizations used:

* Line charts
* Boxplots
* Heatmaps
* Distribution plots

---

### C. Methods Used for Anomaly Detection

#### 1. Statistical Method

* Rolling Mean & Rolling Standard Deviation
* Z-Score (3σ rule)** for anomaly thresholding
  Useful for detecting point anomalies and seasonal deviations.

#### 2. Machine Learning Method – Isolation Forest

Why Isolation Forest?

* Unsupervised
* Scales well on large datasets
* Captures complex multi-feature anomalies
* Works well for non-linear patterns

Input Features:

* Sales
* Store ID
* Product ID
* Year, Month, Day

Output:

* `-1` → Anomaly
* `1` → Normal

#### 3. Hybrid Approach

Both methods’ outputs were compared:

* If both flagged → high confidence anomaly
* If only one flagged → examined using EDA

This improves accuracy and reduces false positives.

---

## 📊 4. Key Observations & Findings

### A. Seasonal Trends

* Sales consistently increase during festive months (Oct–Dec).
* Some products show strong seasonality (product-3, product-7).

### B. Anomalies Detected

* Multiple unexpected drops in 2014, 2016, 2018.
* Certain spikes unrelated to seasonality → possible promotional events or data noise.
* 
### C. Store-Level Patterns

* Store-2 and Store-5: higher irregularity.
* Store-7: most stable performance.

### D. Isolation Forest Insights

* Captured global anomalies effectively.
* Identified multi-dimensional outliers not visible through simple statistics.

---

## 📈 5. Result Summary

* Successfully detected anomalies using statistical + ML methods.
* Combined approach improved reliability of detection.
* Created a Streamlit dashboard for interactive exploration.
* Helps in improving:

  * Forecasting
  * Inventory planning
  * Fraud detection
  * Data quality monitoring

---

## ▶️ 6. How to Run the App

###1. Install Required Libraries

```bash
pip install -r requirements.txt
```

###2. Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 7. Files Included

* `anomaly.py` – Streamlit dashboard
* `train.csv` – Sales data 2010–2018
* `README.md` – Documentation

---

## 🏁 8. Conclusion

This project demonstrates a robust approach to anomaly detection using a hybrid statistical + ML pipeline, supported by data visualization and an interactive interface. The solution is scalable, interpretable, and practical for real-world business use.

