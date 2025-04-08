import pandas as pd

train = pd.read_csv('data/phone_prices/train.csv')
# test = pd.read_csv('data/phone_prices/test.csv')

y = train["price_range"]
X = train.drop("price_range", axis=1)

# Save the processed data to new CSV files
X.to_csv('data/phone_prices/train.csv', index=False)
y.to_csv('data/phone_prices/y_train.csv', index=False)