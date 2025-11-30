# build_scaler.py
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DATA_PATH = "Data/credit_approval.data"   # change if needed

# 1) load dataset
df = pd.read_csv(DATA_PATH, header=None, na_values=["?"], low_memory=False)

# 2) rename columns
d = {0:'Gender',1:'Age',2:'Debt',3:'Married',4:'BankCustomer',5:'EducationLevel',
     6:'Ethnicity',7:'YearsEmployed',8:'PriorDefault',9:'Employed',10:'CreditScore',
     11:'DriversLicense',12:'Citizen',13:'ZipCode',14:'Income',15:'ApprovalStatus'}
df.rename(columns=d, inplace=True)

# 3) convert numeric columns
for col in ['Age','Debt','YearsEmployed','CreditScore','ZipCode','Income']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4) fill & encode categorical columns
for col in df.select_dtypes(include=['object','category']).columns:
    if df[col].isna().any():
        mode = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode[0] if len(mode)>0 else "missing")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 5) selected features (the 6 your model expects)
features = ['Debt','YearsEmployed','PriorDefault','Employed','Income','CreditScore']

# 6) fit scaler
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(df[features])

# 7) save scaler and features order
with open("scaler.sav", "wb") as f:
    pickle.dump(scaler, f)
with open("features_list.pkl", "wb") as f:
    pickle.dump(features, f)

# 8) print min/max for your info
mins = df[features].min().to_dict()
maxs = df[features].max().to_dict()
print("Saved scaler.sav and features_list.pkl")
print("MINS:", mins)
print("MAXS:", maxs)
