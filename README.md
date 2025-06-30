# 🌲 Heart Disease Prediction – Decision Trees & Random Forests

## 📌 Objective
To classify whether a patient has heart disease using tree-based machine learning models — Decision Tree and Random Forest — and evaluate their performance using metrics and cross-validation.

---

## 📁 Dataset
- **Source**: [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **File Used**: `heart.csv`

---

## 🛠 Tools Used
- Python
- Pandas, NumPy
- Scikit-learn (DecisionTree, RandomForest, metrics, cross-validation)
- Matplotlib, Seaborn

---

## ✅ Steps Performed

### 1. Data Cleaning
- Confirmed no missing values
- All features were numeric, no encoding needed

### 2. Train-Test Split
- Split data into training and testing (80/20 ratio)

### 3. Trained a Decision Tree Classifier
- Model: `DecisionTreeClassifier(random_state=42)`
- **Accuracy**: 98.5%
- **Confusion Matrix**:
[[102 0]
[ 3 100]]
- **Classification Report**:
- Precision: 0.97–1.00
- Recall: 0.97–1.00
- F1-Score: **0.99**

### 4. Visualized Decision Tree
- Plotted tree using `plot_tree()` from sklearn
- Showed clear split logic and decision paths

### 5. Controlled Overfitting
- Pruned tree by setting `max_depth=3` (optional tuning step)

### 6. Trained a Random Forest
- Model: `RandomForestClassifier(n_estimators=100)`
- Compared performance to single tree
- Feature importance chart showed top contributing features

### 7. Evaluated with Cross-Validation
- 5-fold cross-validation with `cross_val_score()`
- **Scores**: `[1.0, 1.0, 1.0, 1.0, 0.985]`
- **Average Accuracy**: **99.7%**

---

## 🔍 Feature Importances (Top Predictors)
- `cp` (chest pain type)
- `thalach` (max heart rate)
- `oldpeak` (ST depression)
- `ca`, `thal`, `exang`

---

## 📊 Key Insights
- Random Forest clearly outperformed a single Decision Tree.
- High accuracy + cross-validation mean the model generalizes well.
- Tree depth control helps prevent overfitting in single trees.

---

## 📂 Files Included
- `Task5.ipynb` – Complete notebook with models and plots
- `heart.csv` – Dataset used
- `README.md` – Project documentation

---

## 📝 Submission
This project was completed as part of the **AI & ML Internship**  
**Task 5: Tree-Based Models – Decision Trees & Random Forests**

