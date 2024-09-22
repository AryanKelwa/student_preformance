from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from src.Pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

#Route for the Home Page
@app.route('/')
def index():
    return render_template('index.htm')



@app.route( '/predict',methods=['GET','POST'] )
def predict():
    if request.method == 'GET':
        return render_template('home.htm')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            writing_score = float(request.form.get('writing_score')),
            reading_score = float(request.form.get('reading_score'))            
        )
        
        pred_df = data.get_data_as_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.htm',results = results[0])
        

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
