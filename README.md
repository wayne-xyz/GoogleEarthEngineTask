# GoogleEarthEngineTask
Several Tasks about using Google Earth Engine.

## Tif_ML

### Compared_ml_v2_downscal.ipynb

# Fire Detection using Multi-Source Satellite Imagery

This project implements a machine learning approach to detect fires using multi-source satellite imagery, including NICFI, Sentinel, and other remote sensing data.

## Project Overview

The project uses multiple satellite data sources to detect and analyze fire occurrences:
- NICFI (Norway's International Climate and Forest Initiative) imagery
- Sentinel satellite data
- Combined multi-source approach

## Data Pipeline

### 1. Data Preparation
- Downscaling of high-resolution imagery
- Feature extraction from multiple bands
- Data cleaning and preprocessing
- Handling class imbalance using undersampling

### 2. Model Implementation
Three different models are implemented and compared:
- Logistic Regression
- Random Forest
- XGBoost

### 3. Performance Metrics
The models are evaluated using metrics suitable for imbalanced datasets:
- Balanced Accuracy
- Matthews Correlation Coefficient
- ROC AUC Score
- Precision-Recall Curves
- Cohen's Kappa

## Project Structure