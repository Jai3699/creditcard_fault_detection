from flask import Flask,render_template,request,app,Response
import pickle
import numpy as np
import pandas as pd



app = Flask(__name__)
rfc=pickle.load(open('C:/Users/Jai Yaduvanshi/OneDrive/Desktop/creditcard_fraud_detection/pickle/credit_fault.pkl','rb'))
scaler=pickle.load(open('C:/Users/Jai Yaduvanshi/OneDrive/Desktop/creditcard_fraud_detection/pickle/scaler.pkl','rb'))


@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def fun():
        
    if request.method=='POST':
        LIMIT_BAL=int(request.form['LIMIT_BAL'])
        SEX=int(request.form['SEX'])
        EDUCATION=int(request.form['EDUCATION'])
        MARRIAGE=int(request.form['MARRIAGE'])
        AGE=int(request.form['AGE'])
        PAY_0=int(request.form['PAY_0'])
        PAY_2=int(request.form['PAY_2'])
        PAY_3=int(request.form['PAY_3'])
        PAY_4=int(request.form['PAY_4'])
        PAY_5=int(request.form['PAY_5'])
        PAY_6=int(request.form['PAY_6'])
        BILL_ATM1=int(request.form['BILL_ATM1'])
        BILL_ATM2=int(request.form['BILL_ATM2'])
        BILL_ATM3=int(request.form['BILL_ATM3'])
        BILL_ATM4=int(request.form['BILL_ATM4'])
        BILL_ATM5=int(request.form['BILL_ATM5'])
        BILL_ATM6=int(request.form['BILL_ATM6'])
        PAY_ATM1=int(request.form['PAY_ATM1'])
        PAY_ATM2=int(request.form['PAY_ATM2'])
        PAY_ATM3=int(request.form['PAY_ATM3'])
        PAY_ATM4=int(request.form['PAY_ATM4'])
        PAY_ATM5=int(request.form['PAY_ATM5'])
        PAY_ATM6=int(request.form['PAY_ATM6'])
       
        
        scaled_data=scaler.transform([[LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_ATM1,BILL_ATM2,BILL_ATM3,BILL_ATM4,BILL_ATM5,BILL_ATM6,PAY_ATM1,PAY_ATM2,PAY_ATM3,PAY_ATM4,PAY_ATM5,PAY_ATM6]])
        outcome=rfc.predict(scaled_data)
        if outcome[0]==1:
            result='Fraud'
            return render_template('output.html',result=result)
        else:
            result='Not Fraud'
            return render_template('output.html',result=result)  
        
    else:
        return render_template('home.html')    

if __name__=="__main__":
    app.run(host="0.0.0.0")