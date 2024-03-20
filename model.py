import numpy as np
import pandas as pd
import re 
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')

df.columns = df.columns.str.replace('.', '_', regex=False)

df.fillna('Unkown',inplace=True)

def drop_columns(df):
    df.drop(['UniqueID', 'branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID','State_ID', 'Employee_code_ID'], axis=1, inplace=True)
    return df

df = drop_columns(df)

def credit_risk(train):
    d1=[]
    d2=[]
    for i in train:
        a = i.split("-")
        if len(a) == 1:
            d1.append(a[0])
            d2.append('unknown')
        else:
            d1.append(a[1])
            d2.append(a[0])

    return d1,d2

def calc_number_of_ids(row):
    return sum(row[['Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag',
       'Passport_flag']])
def check_pri_installment(row):
    if row['PRIMARY_INSTAL_AMT']<=1:
        return 0
    else:
        return row['PRIMARY_INSTAL_AMT']

risk_map = {'No Bureau History Available':-1, 
              'Not Scored: No Activity seen on the customer (Inactive)':-1,
              'Not Scored: Sufficient History Not Available':-1,
              'Not Scored: No Updates available in last 36 months':-1,
              'Not Scored: Only a Guarantor':-1,
              'Not Scored: More than 50 active Accounts found':-1,
              'Not Scored: Not Enough Info available on the customer':-1,
              'Very Low Risk':4,
              'Low Risk':3,
              'Medium Risk':2, 
              'High Risk':1,
              'Very High Risk':0}

#Have used the grading system in descending order because A is least risky and going forward risk increases
sub_risk = {'unknown':-1, 'I':5, 'L':2, 'A':13, 'D':10, 'M':1, 'B':12, 'C':11, 'E':9, 'H':6, 'F':8, 'K':3,
       'G':7, 'J':4}

#Firstly converting the employment type to numbers:

employment_map = {'Self employed':0, 'Salaried':1, 'Unkown':-1}


def preprocessing_data(df):

#Age and Disbursal time in years
    df.loc[:,'age'] = pd.to_datetime('today').year - pd.to_datetime(df['Date_of_Birth']).dt.year
    df.loc[:,'disbursal_time'] = pd.to_datetime('today').year - pd.to_datetime(df['DisbursalDate']).dt.year

#Now converting AVERAGE.ACCT.AGE into number of months :
    df['Average_Acct_Age_Months'] = df['AVERAGE_ACCT_AGE'].apply(lambda x : int(re.findall(r'\d+',x)[0])*12 + int(re.findall(r'\d+',x)[1]))

# Now Converting CREDIT.HISTORY.LENGTH into number of months:

    df['Credit_History_Length_Months'] = df['CREDIT_HISTORY_LENGTH'].apply(lambda x : int(re.findall(r'\d+',x)[0])*12 + int(re.findall(r'\d+',x)[1]))

#adding a feature of number of zeroes present in a row so that we can count how many zeroes on row has

    df['Number_of_0'] = (df == 0).astype(int).sum(axis=1)
    
#creating additional column to split the PERFORM_CNS.SCORE.DESCRIPTION using credit risk function defined above

    df.loc[:,'Credit_Risk'],df.loc[:,'Credit_Risk_Grade']  = credit_risk(df["PERFORM_CNS_SCORE_DESCRIPTION"])

#adding loan to asset ratio to check which if the clients with default had suufficient assets to repay loan at time of disbursement

    df.loc[:, 'Loan_to_Asset_Ratio'] = df['disbursed_amount'] /df['asset_cost']

#adding total number of accounts feature:

    df.loc[:,'No_of_Accts'] = df['PRI_NO_OF_ACCTS'] + df['SEC_NO_OF_ACCTS']

#Now adding columns carrying total number of  various accounts including the primary and secondary and combing them in one

    df.loc[:,'Pri_Inactive_Accts'] = df['PRI_NO_OF_ACCTS'] - df['PRI_ACTIVE_ACCTS']
    df.loc[:,'Sec_Inactive_Accts'] = df['SEC_NO_OF_ACCTS'] - df['SEC_ACTIVE_ACCTS']
    df.loc[:,'Tot_Inactive_Accts'] = df['Pri_Inactive_Accts'] + df['Sec_Inactive_Accts']
    df.loc[:,'Tot_Overdue_Accts'] = df['PRI_OVERDUE_ACCTS'] + df['SEC_OVERDUE_ACCTS']
    df.loc[:,'Tot_Current_Balance'] = df['PRI_CURRENT_BALANCE'] + df['SEC_CURRENT_BALANCE']
    df.loc[:,'Tot_Sanctioned_Amount'] = df['PRI_SANCTIONED_AMOUNT'] + df['SEC_SANCTIONED_AMOUNT']
    df.loc[:,'Tot_Disbursed_Amount'] = df['PRI_DISBURSED_AMOUNT'] + df['SEC_DISBURSED_AMOUNT']
    df.loc[:,'Tot_Installment'] = df['PRIMARY_INSTAL_AMT'] + df['SEC_INSTAL_AMT']
    df.loc[:,'Bal_Disburse_Ratio'] = np.round((1+df['Tot_Disbursed_Amount'])/(1+df['Tot_Current_Balance']),2)
    df.loc[:,'Pri_Tenure'] = (df['PRI_DISBURSED_AMOUNT']/( df['PRIMARY_INSTAL_AMT']+1)).astype(int)
    df.loc[:,'Sec_Tenure'] = (df['SEC_DISBURSED_AMOUNT']/(df['SEC_INSTAL_AMT']+1)).astype(int)
    df.loc[:,'Disburse_to_Sactioned_Ratio'] =  np.round((df['Tot_Disbursed_Amount']+1)/(1+df['Tot_Sanctioned_Amount']),2)
    df.loc[:,'Active_to_Inactive_Acct_Ratio'] =  np.round((df['No_of_Accts']+1)/(1+df['Tot_Inactive_Accts']),2)
    return df

def map_data(df):
    df.loc[:,'Credit_Risk_Label'] = df['Credit_Risk'].apply(lambda x: risk_map[x])
    df.loc[:,'Sub_Risk_Label'] = df['Credit_Risk_Grade'].apply(lambda x: sub_risk[x])
    df.loc[:,'Employment_Type'] = df['Employment_Type'].apply(lambda x: employment_map[x])

    return df

def data_correction(df):
    #Many customers have invalid date of birth, so immute invalid data with mean age
    df.loc[:,'PRI_CURRENT_BALANCE'] = df['PRI_CURRENT_BALANCE'].apply(lambda x: 0 if x<0 else x)
    df.loc[:,'SEC_CURRENT_BALANCE'] = df['SEC_CURRENT_BALANCE'].apply(lambda x: 0 if x<0 else x)

    #loan that do not have current pricipal outstanding should have 0 primary installment
    df.loc[:,'new_pri_installment']= df.apply(lambda x : check_pri_installment(x),axis=1)
    return df

def preprocessed_data(df):
    df = data_correction(df)
    df = preprocessing_data(df)
    df = map_data(df)

    return df

df = preprocessed_data(df)
df = df[df['Number_of_0']<=25]

features = ['disbursed_amount', 'asset_cost', 'Employment_Type', 'Aadhar_flag', 'PAN_flag', 'PERFORM_CNS_SCORE', 
            'NEW_ACCTS_IN_LAST_SIX_MONTHS', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'NO_OF_INQUIRIES',
            'age', 'Average_Acct_Age_Months', 'Credit_History_Length_Months', 'Number_of_0', 'Loan_to_Asset_Ratio', 
            'No_of_Accts', 'Tot_Inactive_Accts', 'Tot_Overdue_Accts', 'Tot_Current_Balance', 'Tot_Sanctioned_Amount', 
            'Tot_Disbursed_Amount', 'Tot_Installment', 'Bal_Disburse_Ratio', 'Pri_Tenure', 'Sec_Tenure', 
            'Disburse_to_Sactioned_Ratio', 'Active_to_Inactive_Acct_Ratio', 'Credit_Risk_Label', 'Sub_Risk_Label']

# features = ['disbursed_amount', 'asset_cost', 'Aadhar_flag', 'PAN_flag', 'PERFORM_CNS_SCORE', 'NEW_ACCTS_IN_LAST_SIX_MONTHS', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'NO_OF_INQUIRIES', 'age', 'Average_Acct_Age_Months', 'Credit_History_Length_Months']

X = df[features]
y = df['loan_default']

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
# from sklearn.decomposition import PCA
from xgboost import XGBClassifier

steps = [('scaler', RobustScaler()), ('xgb', XGBClassifier())]

pipe = Pipeline(steps)

pipe.fit(X_train, y_train)

print(pipe.score(X_val, y_val))

import pickle
pickle.dump(pipe, open('model.pkl', 'wb'))



