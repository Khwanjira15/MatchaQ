from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

from nlp_project import evaluate_portfolios, results_to_df

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
JD_FOLDER = os.path.join(UPLOAD_FOLDER, "jd")
PF_FOLDER = os.path.join(UPLOAD_FOLDER, "portfolio")

os.makedirs(JD_FOLDER, exist_ok=True)
os.makedirs(PF_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    table_data = []

    if request.method == "POST":
        jd_file = request.files["jd"]
        portfolio_files = request.files.getlist("portfolio")

        jd_path = os.path.join(JD_FOLDER, secure_filename(jd_file.filename))
        jd_file.save(jd_path)

        pf_paths = []
        for file in portfolio_files:
            path = os.path.join(PF_FOLDER, secure_filename(file.filename))
            file.save(path)
            pf_paths.append(path)

        results = evaluate_portfolios(jd_path, pf_paths)

        # 🔥 แปลงเป็น DataFrame
        df, _, _, _, _ = results_to_df(results, jd_path, pf_paths)

        table_data = df.to_dict(orient="records")

    return render_template("index.html", results=results, table_data=table_data)


if __name__ == "__main__":
    app.run(debug=True)