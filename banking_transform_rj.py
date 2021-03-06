import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



df = pd.read_csv('data/banking/bank-additional-full.csv', sep=';')
df_onehot = pd.read_csv('data/banking/bank-additional-full.csv', sep=';')

columns = ["job",
           "marital",
           "education",
           "default", "housing",
           "loan",
           "contact",
           "month",
           "day_of_week",
           "poutcome",
           "y"]

for column in columns:
    one_hot = pd.get_dummies(df_onehot[column], prefix=column)
    del df_onehot[column]
    if column=="y":
        df_onehot[column]=LabelEncoder().fit_transform(df[column]) #Use the label encoder for the label.
    else:
        df_onehot = df_onehot.join(one_hot)

for column in columns:
    df[column] = LabelEncoder().fit_transform(df[column])
    
##For CNN, I will use binary labelisation. Using numerical categories does not make any sense. (Although I must
# agree that using CNN on this kind of data does not seem right anyways.)
train, test = train_test_split(df_onehot, test_size=0.2)
train.to_csv("data/banking/bank-additional-full-transformed-bin-train.csv", sep=";", index=False)
test.to_csv("data/banking/bank-additional-full-transformed-bin-test.csv", sep=";", index=False)

# This column should be dropped for a realistic predictive model:
# Please read Attribute Information
# on this page:https://archive.ics.uci.edu/ml/datasets/bank+marketing
df.drop("duration", axis=1)

train, test = train_test_split(df, test_size=0.2)
train.to_csv("data/banking/bank-additional-full-transformed-train.csv", sep=";", index=False)
test.to_csv("data/banking/bank-additional-full-transformed-test.csv", sep=";", index=False)

## 