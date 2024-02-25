from flask import Flask,request,jsonify
import pickle
import pandas as pd
with open("logistic_regression_model.pkl","rb") as f:
    logistic_model=pickle.load(f)

with open('preprocessing_module.pkl',"rb") as p:
    preprocessor=pickle.load(p)

app=Flask(__name__)
@app.route("/",methods=['POST'])
def predict():
 
    input_data = request.get_json()
    default_values = {
        'significance': 'Religious',
        'styleofplace': 'Temple',
        'unit-currency': {'amount': '0'},  # Default to 0 if currency is missing
        'time': '1',
        'rating': '4'
    }
    Significance=input_data['queryResult']['parameters']['significance']
    type_of_place=input_data['queryResult']['parameters']['styleofplace']
    unit_currency = input_data['queryResult']['parameters']['unit-currency']
    if isinstance(unit_currency, dict):
        fee = unit_currency.get('amount', '')
    else:
        fee = '0'

    time=input_data['queryResult']['parameters']['number']
    rating=input_data['queryResult']['parameters']['rating']
    if Significance=="":
        Significance="Religious"
    if type_of_place=="":
        type_of_place="Temple"
    
    if time=="":
        time="2"
    
    if rating=="":
        rating="4"
    
    input_df = pd.DataFrame({
        'Type': [type_of_place],
        'time needed to visit in hrs': [time],
        'Google review rating': [rating],
        'Entrance Fee in INR': [fee],
        'Significance': [Significance]
    })
    processed_df=preprocessor.transform(input_df)
   
    predictions = logistic_model.predict(processed_df)
    response = {
        "fulfillmentText": f"The place where you should go is {predictions[0]}"
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)