This repository provides a two-stage cascade classification pipeline designed for imbalanced material property classification tasks. The framework implements full data processing, adaptive feature engineering, multi-model ensemble training, noise-aware data augmentation, grid search optimization, and standardized model persistence.
The entire workflow is split into modular, maintainable Python files with clear responsibilities, supporting reproducible training, rigorous evaluation, and end-to-end model deployment.
1. Project Overview
Task Definition

2. Project Structure & Module Description
All modules are decoupled for independent debugging and iterative optimization.
config.py
data_pipeline.py
preprocessor.py
model_utils.py
cascade_stage1.py
cascade_stage2.py
evaluate_viz.py
model_saver.py
main.py
3. Environment Dependencies
Install all required packages:
pip install -r requirements.txt
Core dependencies:
- numpy, pandas
- scikit-learn
- lightgbm, xgboost
- imbalanced-learn
- matplotlib, joblib
4. How to Run
1. Place your CSV dataset in the project root directory
2. Check and modify dataset name in config.py
3. Run the full pipeline:
python main.py
5. Output Files
All outputs are saved incascade_models/:
- Model files: All stage1 & stage2 trained models (*.pkl)
- Preprocessor files: Imputer, scaler, variance filter, feature selectors
- Feature logs: Retained feature columns in JSON
- Data arrays: Processed train/test features and labels (*.npy)
- Config: Best threshold, ensemble weights and optimal metrics
- Visualization: Confusion matrix, PR curve figures
6. Pipeline Workflow
Data Loading → Cleaning & Filtering → Preprocessing & Normalization → Dual-stage Feature Selection → Stage1 Binary Training → Stage2 Augmented Fine Classification → Grid Search Optimization → Metric Evaluation & Visualization → Model Persistence
7. Key Features
8. License
This project is open for academic research and non-commercial usage.