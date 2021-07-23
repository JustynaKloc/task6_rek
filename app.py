from flask import Flask, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():

    report = load("report.pkl")
    results1 = load("results.pkl")
    report1 = pd.DataFrame(report).T
    return render_template('index.html',tables=[report1.to_html(classes='data')], titles=report1.columns.values, result= results1)
    
if __name__ == "__main__":
    app.run(debug=True)