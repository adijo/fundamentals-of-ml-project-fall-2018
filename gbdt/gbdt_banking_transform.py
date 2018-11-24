import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
df.to_csv("../data/banking/bank-additional-full-transformed.csv", sep=";", index=False)
