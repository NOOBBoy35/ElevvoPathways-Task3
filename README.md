# Forest Cover Type Classification - Level 2 Task 3

## 📋 Project Overview

This project implements a machine learning classification system to predict forest cover types using the UCI Forest Cover Type dataset. The goal is to classify forest areas into different cover types based on various environmental and geographical features.

## 🎯 Objective

Build and evaluate machine learning models to classify forest cover types with high accuracy using ensemble methods and advanced techniques.

## 📊 Dataset

**Dataset**: UCI Forest Cover Type Dataset
- **Source**: UCI Machine Learning Repository
- **Size**: ~75MB (covtype.data)
- **Records**: 581,012 observations
- **Features**: 54 features (10 continuous + 44 categorical)
- **Target**: 7 forest cover types (1-7)

### Feature Categories:
- **Continuous Features** (10):
  - Elevation, Aspect, Slope
  - Horizontal/Vertical Distance to Hydrology
  - Horizontal Distance to Roadways
  - Hillshade (9am, Noon, 3pm)
  - Horizontal Distance to Fire Points

- **Categorical Features** (44):
  - Wilderness Area (4 binary features)
  - Soil Type (40 binary features)

## 🏗️ Project Structure

```
Level2_Task3/
├── Level2_Task3.ipynb          # Main Jupyter notebook
├── Level2_Task3_executed.ipynb # Executed notebook with outputs
├── covertype/                   # Dataset directory
│   ├── covtype.data            # Main dataset file
│   ├── covtype.info            # Dataset information
│   └── old_covtype.info        # Legacy info file
└── README.md                   # This file
```

## 🚀 Implementation Steps

### 1. Data Loading & Preprocessing
- Load the dataset using pandas
- Define column names for all 55 columns (54 features + 1 target)
- Handle missing values and data validation

### 2. Data Cleaning & Feature Engineering
- Separate features (X) and target variable (y)
- Apply StandardScaler for feature normalization
- Split data into training (80%) and testing (20%) sets
- Use stratified sampling to maintain class distribution

### 3. Model Training & Evaluation

#### Primary Model: Random Forest
- **Algorithm**: RandomForestClassifier
- **Parameters**: n_estimators=100, random_state=42
- **Performance**: 95.42% accuracy
- **Features**: 54 input features
- **Output**: 7-class classification

#### Bonus Model: XGBoost
- **Algorithm**: XGBoost (Gradient Boosting)
- **Parameters**: 100 boosting rounds, multi-class objective
- **Performance**: 86.85% accuracy
- **Installation**: Automatic via pip during execution

### 4. Model Analysis & Visualization

#### Confusion Matrix
- Visual representation of model predictions
- Heatmap showing true vs predicted classes
- Identifies misclassification patterns

#### Feature Importance Analysis
- Top 15 most important features
- Bar plot visualization
- Helps understand key predictive factors

### 5. Hyperparameter Tuning
- **Method**: GridSearchCV
- **Parameters**: n_estimators, max_depth, min_samples_split
- **Cross-validation**: 2-fold CV
- **Optimization**: Accuracy scoring

## 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 95.42% | 0.95 | 0.95 | 0.95 |
| XGBoost | 86.85% | 0.87 | 0.87 | 0.87 |

### Class-wise Performance (Random Forest):
- **Class 1**: 96% precision, 94% recall
- **Class 2**: 95% precision, 97% recall  
- **Class 3**: 94% precision, 96% recall
- **Class 4**: 91% precision, 85% recall
- **Class 5**: 95% precision, 78% recall
- **Class 6**: 93% precision, 89% recall
- **Class 7**: 97% precision, 95% recall

## 🛠️ Dependencies

### Required Libraries:
```python
pandas          # Data manipulation
scikit-learn    # Machine learning algorithms
seaborn         # Data visualization
matplotlib      # Plotting
xgboost         # Gradient boosting (installed automatically)
```

### Installation:
```bash
pip install pandas scikit-learn seaborn matplotlib
pip install xgboost  # Installed automatically in notebook
```

## 🎯 Key Findings

1. **Random Forest outperforms XGBoost** in this classification task
2. **High accuracy** (95.42%) achieved with Random Forest
3. **Feature importance analysis** reveals key environmental factors
4. **Stratified sampling** ensures balanced class representation
5. **StandardScaler** improves model performance significantly

## 🔧 Usage

1. **Run the notebook**:
   ```bash
   jupyter notebook Level2_Task3.ipynb
   ```

2. **Execute all cells** to reproduce results

3. **View outputs**:
   - Model performance metrics
   - Confusion matrix visualization
   - Feature importance plots
   - Model comparison results

## 📝 Notes

- The dataset is automatically downloaded and processed
- XGBoost installation happens automatically during execution
- Hyperparameter tuning uses a reduced parameter grid for efficiency
- All visualizations are generated inline in the notebook

## 🤝 Contributing

This project is part of the ElevvoPathways Level 2 curriculum. For questions or improvements, please refer to the course guidelines.

---

**Author**: Abdullah  
**Date**: August 2024  
**Course**: ElevvoPathways Level 2 - Task 3 