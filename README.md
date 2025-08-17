# 🏠 Home Credit Default Risk Case Study

## 📌 Project Overview

This project presents a case study on solving a **binary classification problem**: predicting whether a person applying for a home credit will be able to repay their loan.

The model predicts:
- **1** → Client is likely to have payment difficulties (late payment of more than X days on at least one of the first Y installments).
- **0** → Client is expected to repay without major delays.

The main evaluation metric is **[AUC-ROC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es_419)**, meaning the model outputs **probabilities** that a loan will not be repaid.

---

## 📂 Dataset

The original dataset comes with multiple files containing loan-related information.
For this project, the focus is exclusively on:

- `application_train_aai.csv`
- `application_test_aai.csv`

The notebook handles the data download automatically in **Section 1 - Getting the Data**.

---

## 🛠 Tech Stack

The following tools and libraries were used to build and evaluate the model:

- **Python** – Core programming language
- **Pandas** – Data loading and manipulation
- **Scikit-learn** – Feature engineering, model training, and evaluation
- **Matplotlib & Seaborn** – Data visualization
- **Jupyter Notebooks** – Interactive experimentation

---

## 🚀 Installation

To run this project, clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

> 💡 *Tip:* It’s recommended to install dependencies inside a virtual environment.

---

## 🧹 Code Style

To maintain clean and consistent code, **[Black](https://black.readthedocs.io/)** and **[isort](https://pycqa.github.io/isort/)** are used for automatic formatting:

```bash
isort --profile=black . && black --line-length 88 .
```

Further reading on Python code style:

* [The Hitchhiker’s Guide to Python: Code Style](https://docs.python-guide.org/writing/style/)
* [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

## ✅ Testing

The project includes unit tests to ensure correctness. They can be run with:

```bash
pytest tests/
```

More on Python testing:

* [Effective Python Testing With Pytest](https://realpython.com/pytest-python-testing/)
* [The Hitchhiker’s Guide to Python: Testing Your Code](https://docs.python-guide.org/writing/tests/)

---

## 📊 Workflow Summary

1. **Data Loading** – Automatic download via Google Drive links.
2. **Preprocessing** – Handling missing values, encoding categorical variables, scaling features.
3. **Model Training** – Using supervised learning algorithms for binary classification.
4. **Evaluation** – Primary metric: AUC-ROC score.
5. **Testing** – Ensuring functions and preprocessing steps work as intended.

---

💡 This project serves as a case study for applying machine learning to real-world financial risk prediction problems.
