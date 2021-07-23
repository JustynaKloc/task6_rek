from joblib import load
import seaborn as sns
import pandas as pd


model = load("task6model.pkl")
print("Type of harmfulness")

report = load("report.pkl")
print("*******Model scoring********")
print(report)
results = load("results.pkl")
print(results)

