from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "marks_predictor.pkl")
model = joblib.load(model_path)

# Mappings for categorical inputs
parent_map = {
    "high school": 0
}
prep_map = {"none": 0, "completed": 1}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get input values
            hours = float(request.form["hours"])
            mid = float(request.form["mid"])
            weekend = float(request.form["weekend"])
            attendance = float(request.form["attendance"])
            education = parent_map[request.form["parent_edu"]]
            prep = prep_map[request.form["test_prep"]]

            # Prepare input for model
            input_data = np.array([[hours, mid, weekend, attendance, education, prep]])
            prediction = model.predict(input_data)[0]
            percentage = round(prediction, 2)

            # Determine result logic
            if attendance < 60:
                result = "FAIL"
            elif mid < 45:
                result = "FAIL"
            elif weekend < 45:
                result = "FAIL"
            elif percentage < 35:
                result = "FAIL"
            else:
                result = "PASS"

            return render_template("index.html", percentage=percentage, result=result)

        except Exception as e:
            return f"Error occurred: {e}"

    return render_template("index.html", percentage=None, result=None)


if __name__ == "__main__":
    app.run(debug=False)
