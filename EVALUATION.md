# 🤖 Model Evaluation Report
(An additional document for the report on the given task)

This document provides a detailed report on the performance of the planetary classifier Model. The primary measures used for evaluation of the task are **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

---

## 1. Logistic Regression

* **Accuracy:** 75.2%
* **Weighted Avg F1-Score:** 0.75

### Classification Report
precision    recall  f1-score   support

       0       0.86      0.84      0.85      1127
       1       0.82      0.89      0.86      1279
       2       0.74      0.75      0.75      1129
       3       0.71      0.70      0.70      1163
       4       0.73      0.75      0.74      1111
       5       0.65      0.61      0.63      1026
       6       0.83      0.81      0.82      1128
       7       0.82      0.82      0.82      1186
       8       0.69      0.67      0.68      1114
       9       0.65      0.64      0.64      1130

accuracy                           0.75     11393
---

## 2. Naive Bayes

* **Accuracy:** 73.3%
* **Weighted Avg F1-Score:** 0.73

### Classification Report
          precision    recall  f1-score   support

       0       0.89      0.81      0.85      1127
       1       0.79      0.89      0.84      1279
       2       0.78      0.75      0.76      1129
       3       0.68      0.68      0.68      1163
       4       0.68      0.70      0.69      1111
       5       0.59      0.60      0.60      1026
       6       0.84      0.81      0.82      1128
       7       0.81      0.77      0.79      1186
       8       0.68      0.67      0.68      1114
       9       0.59      0.60      0.59      1130

accuracy                           0.73     11393
---

## 3. Random Forest (Final Model)

The Random Forest classifier demonstrated the best performance across all metrics.

* **Accuracy:** 87.1%
* **Weighted Avg F1-Score:** 0.87

### Classification Report
          precision    recall  f1-score   support

       0       0.94      0.92      0.93      1127
       1       0.96      0.96      0.96      1279
       2       0.88      0.91      0.90      1129
       3       0.84      0.82      0.83      1163
       4       0.84      0.83      0.83      1111
       5       0.84      0.83      0.83      1026
       6       0.92      0.91      0.92      1128
       7       0.89      0.90      0.90      1186
       8       0.81      0.83      0.82      1114
       9       0.79      0.78      0.78      1130

accuracy                           0.87     11393
---

## 🏆 Summary

The **Random Forest** model was selected as the final model due to its superior F1-score and accuracy.
