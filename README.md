# heart-disease-ecg-prediction
Deep learning model (1D-CNN + BiLSTM) for ECG signal classification (Normal, MI, ST/T Change) using PTB-XL dataset.

# Heart Disease Prediction from ECG Signals

This project develops a deep learning model (1D-CNN + BiLSTM) to classify ECG signals into:
- Normal (NORM)
- Myocardial Infarction (MI)
- ST/T Change (STTC)

## Dataset
- PTB-XL ECG dataset from PhysioNet
- 18,512 records (12-lead, 10s, 100Hz)

## Model
- Input: ECG signals (1000x12)
- Architecture: 1D CNN + BiLSTM
- Training: k-fold cross validation

## Results
- Accuracy: 86.9%
- F1-score:
  - NORM: 88.2%
  - MI: 87.0%
  - STTC: 85.6%

## Deployment
- Web app built with Streamlit
- Tested with SUS score of 77.25%

## File
- `Model CNN Penelitian Ilmiah.ipynb` → main notebook with model training and evaluation
