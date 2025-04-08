train = pd.read_csv('../data/phone_prices/train.csv')
test = pd.read_csv('../data/phone_prices/train.csv')

y = train["price_range"]
X = train.drop("price_range", axis=1)

y_test = test["price_range"]
X_test = test.drop("price_range", axis=1)