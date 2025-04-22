from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn

class PhonePriceScoreFlow(FlowSpec):

    test_file = Parameter('test_file', default="../data/phoneprices/test.csv", help="Path to the test CSV file")

    @step
    def start(self):
        self.test_df = pd.read_csv(self.test_file)
        # since no ground truth exists on the test dataset we have no y
        self.X_test = self.test_df
        self.y_test = None
        self.next(self.score)

    @step
    def score(self):
        mlflow.set_tracking_uri("http://localhost:5000") 
        # Load the registered model from MLFlow registry. Adjust stage/version as needed.
        self.model = mlflow.sklearn.load_model("models:/PhonePriceModel/Production")
        self.predictions = self.model.predict(self.X_test).tolist()
        self.next(self.end)

    @step
    def end(self):
        print("Predictions:", self.predictions)
        print("Predictions flow completed.")

if __name__ == '__main__':
    PhonePriceScoreFlow()
