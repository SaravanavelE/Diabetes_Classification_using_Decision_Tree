# Diabetes Classification Using Decision Tree

## Project Overview
This project focuses on predicting diabetes in patients using a **Decision Tree Classifier**. The model is trained on medical diagnostic data to classify whether a patient is diabetic or non-diabetic. The project demonstrates a complete machine learning pipeline from data loading to model evaluation and visualization.

---

## Dataset
The dataset consists of medical attributes commonly used for diabetes diagnosis.

### Features
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target
- Outcome  
  - `0` → Non-Diabetic  
  - `1` → Diabetic

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Graphviz
- Google Colab

---

## Workflow
1. Load dataset using Pandas
2. Perform basic data inspection
3. Split dataset into training and testing sets (80:20)
4. Train Decision Tree Classifier
5. Evaluate model accuracy
6. Visualize decision tree

---

## Model Training
```python
from sklearn.tree import DecisionTreeClassifier
```

## Model Evaluation
```from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)
```
## Accuracy
74.67%

## Decision Tree Visualization
```from sklearn.tree import export_graphviz
import graphviz

graphviz.Source(export_graphviz(
    model,
    feature_names=x.columns,
    filled=True
))
```
## Project Structure
├── diabetes.csv
├── diabetes_decision_tree.ipynb
├── README.md

## Future Improvements
Handle zero values using imputation
Hyperparameter tuning
Feature selection
Compare with other models (SVM, Random Forest)
Model deployment using Flask or FastAPI

## Author

Saravanavel E
AI & Data Science Student
GitHub: https://github.com/SaravanavelE

## License
This project is intended for educational and academic use.

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
