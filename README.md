# Planetary  Classifier 🪐

This is my first machine learning project! It's a program that tries to predict whether a planet could be habitable based on data like its temperature, gravity, and water content.

---

## 🤔 What This Code Does: A Step-by-Step Guide

I wrote a Python script that trains a computer to make predictions. Here’s how it works:

1.  **Load the Data:** The code starts by loading the dataset of planets (`planetary_dataset.csv`).

2.  **Clean the Data:** Real-world data is often messy. The script fills in any missing values. For numbers, it uses the average value, and for categories, it uses the most common one.

3.  **Prepare Data for the Model:** Machine learning models only understand numbers. So, the code converts any text-based columns into numerical ones.

4.  **Split the Data:** The data is split into a "training set" (to teach the model) and a "testing set" (to see how well it learned).

5.  **Train and Compare Models:** I trained three different types of models to see which was best for this task:
    * Logistic Regression
    * Naive Bayes
    * **Random Forest** (This one turned out to be the best!)

6.  **Improve the Best Model:** I used a tool called `GridSearchCV` to automatically fine-tune the Random Forest model's settings. This process made the model even more accurate.

7.  **Check What's Important:** The code generates a chart showing which factors (like 'Surface Temperature' or 'Gravity') were the most important for making a prediction.

8.  **Save the Final Model:** The final, smartest version of the model is saved to a file called `best_random_forest_model.pkl`. This file can be used later to make new predictions without having to retrain everything.

---

## 🚀 How to Run It

1.  **Clone this project:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the necessary libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script:**
    ```bash
    python planetary_classifier.py
    ```
    The script will print out its findings and save the model file.

---

## 📊 Results

The Random Forest model performed the best after being fine-tuned.

* **Best Model:** Random Forest Classifier
* **Best Settings Found:** `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}`
* **Model Accuracy:** `0.8677579801030829`

### Most Important Features

This chart shows which planetary features most influenced the model's predictions. It looks like `[Mention the top 1-2 features from your plot]` were the most important!

![Feature Importance Plot](feature_importance.png)

---
