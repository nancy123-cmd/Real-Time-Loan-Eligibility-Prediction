
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("loan_train.csv")

for col in ['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term','Credit_History']:
    df[col].fillna(df[col].mode()[0], inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df.drop('Loan_ID', axis=1, inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open("loan_model.pkl", "wb"))
print("Model trained and saved")
