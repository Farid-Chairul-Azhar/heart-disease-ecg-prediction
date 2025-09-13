# Heart Disease Prediction from ECG Signals

This project develops a deep learning system to predict heart disease from **12-lead ECG signals** using the **PTB-XL dataset**.  
The system classifies ECG signals into three categories:
- **Normal (NORM)**
- **Myocardial Infarction (MI)**
- **ST/T Change (STTC)**


## Dataset
- Source: [PTB-XL ECG dataset (PhysioNet)](https://physionet.org/content/ptb-xl/1.0.1/)
- Total records: **21,799**
- Records used in this research: **18,512**
  - NORM: 9,399  
  - MI: 4,821  
  - STTC: 4,292  
- Each record: **12-lead, 10 seconds, 100Hz**


## Model
- Input: ECG signals `(1000 x 12)`
- Architecture: **1D CNN + BiLSTM**
- Training: **Stratified k-Fold Cross Validation**
- Metrics: Accuracy, Precision, Recall, F1-Score


## Results
- **Best Fold Performance (Fold-3)**:
  - Accuracy: **86.97%**
  - Precision: **86.90%**
  - Recall: **87.00%**
  - F1-Score (macro): **86.92%**
- Per class F1-score:
  - NORM: **88.2%**
  - MI: **87.0%**
  - STTC: **85.6%**


## Implementation
### 🔹 Notebook (Training & Evaluation)
- File: [`notebooks/Model CNN Penelitian Ilmiah.ipynb`](notebooks/Model%20CNN%20Penelitian%20Ilmiah.ipynb)  
- Contains: preprocessing, model training, evaluation, and results visualization.

### 🔹 Web App (Deployment)
- File: [`app/app.py`](app/app.py)  
- Built with **Streamlit**  
- Allows users to upload ECG signals and get predictions (Normal, MI, or STTC).


## 🚀 Deployment
### Option 1: Run locally
```bash
# install dependencies
pip install -r requirements.txt

# run app
streamlit run app/app.py
