from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the saved model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        disbursed_amount = float(request.form["disbursed_amount"])
        asset_cost = float(request.form["asset_cost"])
        Employment_Type = float(request.form["Employment_Type"])
        aadhar_flag = float(request.form["aadhar_flag"])
        pan_flag = float(request.form["pan_flag"])
        perform_cns_score = float(request.form["perform_cns_score"])
        NEW_ACCTS_IN_LAST_SIX_MONTHS = float(request.form["NEW_ACCTS_IN_LAST_SIX_MONTHS"])
        DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS = float(request.form["DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS"])
        NO_OF_INQUIRIES = float(request.form["NO_OF_INQUIRIES"])
        # new_pri_installment = float(request.form["new_pri_installment"])
        age = float(request.form["age"])
        Average_Acct_Age_Months = float(request.form["AVERAGE_ACCT_AGE"])
        Credit_History_Length_Months = float(request.form["Credit_History_Length_Months"])
        Number_of_0 = float(request.form["Number_of_0"])
        Loan_to_Asset_Ratio = float(request.form["Loan_to_Asset_Ratio"])
        No_of_Accts = float(request.form["No_of_Accts"])
        Tot_Inactive_Accts = float(request.form["Tot_Inactive_Accts"])
        Tot_Overdue_Accts = float(request.form["Tot_Overdue_Accts"])
        Tot_Current_Balance = float(request.form["Tot_Current_Balance"])
        Tot_Sanctioned_Amount = float(request.form["Tot_Sanctioned_Amount"])
        Tot_Disbursed_Amount = float(request.form["Tot_Disbursed_Amount"])
        Tot_Installment = float(request.form["Tot_Installment"])
        Bal_Disburse_Ratio = float(request.form["Bal_Disburse_Ratio"])
        Pri_Tenure = float(request.form["Pri_Tenure"])
        Sec_Tenure = float(request.form["Sec_Tenure"])
        Disburse_to_Sactioned_Ratio = float(request.form["Disburse_to_Sactioned_Ratio"])
        Active_to_Inactive_Acct_Ratio = float(request.form["Active_to_Inactive_Acct_Ratio"])
        Credit_Risk_Label = float(request.form["Credit_Risk_Label"])
        Sub_Risk_Label = float(request.form["Sub_Risk_Label"])



        # Make predictions
        prediction = model.predict([[disbursed_amount, asset_cost, Employment_Type, aadhar_flag, pan_flag, perform_cns_score, NEW_ACCTS_IN_LAST_SIX_MONTHS, DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS, NO_OF_INQUIRIES, age, Average_Acct_Age_Months, Credit_History_Length_Months, Number_of_0, Loan_to_Asset_Ratio, No_of_Accts, Tot_Inactive_Accts, Tot_Overdue_Accts, Tot_Current_Balance, Tot_Sanctioned_Amount, Tot_Disbursed_Amount, Tot_Installment, Bal_Disburse_Ratio, Pri_Tenure, Sec_Tenure, Disburse_to_Sactioned_Ratio, Active_to_Inactive_Acct_Ratio, Credit_Risk_Label, Sub_Risk_Label]])
        loan = prediction

        return render_template("result.html", loan=loan)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
