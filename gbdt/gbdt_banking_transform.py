import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



df = pd.read_csv('../data/banking/bank-additional-full.csv', sep=';')

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
    df[column] = LabelEncoder().fit_transform(df[column])


# This column should be dropped for a realistic predictive model:
# Please read Attribute Information
# on this page:https://archive.ics.uci.edu/ml/datasets/bank+marketing
df.drop("duration", axis=1)

train, test = train_test_split(df, test_size=0.2)
train.to_csv("../data/banking/bank-additional-full-transformed-train.csv", sep=";", index=False)
test.to_csv("../data/banking/bank-additional-full-transformed-test.csv", sep=";", index=False)
