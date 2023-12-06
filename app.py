from flask import Flask, request, render_template
import pickle
import numpy as np 

app = Flask(__name__)

model=pickle.load(open('models/DModel.pkl', 'rb'))


@app.route('/')
def home(): 
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    insert_data = (object(x) for x in request.form.values())
    
    pred = model.predict(np.array(insert_data, dtype=object))
    return render_template('home.html', predict_text='Predicted Car Price is {}'.format(pred))

if __name__=="__main__":
    app.run(debug=True)