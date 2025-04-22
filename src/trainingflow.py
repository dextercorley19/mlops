from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

class PhonePriceTrainFlow(FlowSpec):

    n_estimators = Parameter('n_estimators', default=100, type=int, help="Number of trees for RandomForest")

    @step
    def start(self):
        # Load training dataset
        self.train_df = pd.read_csv("../data/phoneprices/train.csv")
        # Create validation split
        self.train_df, self.validation_df = train_test_split(self.train_df, test_size=0.2, random_state=42)  # added validation split
        self.next(self.train)

    @step
    def train(self):
        X_train = self.train_df.drop(columns=["price_range"])
        y_train = self.train_df["price_range"]
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42) 
        self.model.fit(X_train, y_train)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        # Evaluate model on the validation dataset
        X_val = self.validation_df.drop(columns=["price_range"]) 
        y_val = self.validation_df["price_range"]
        predictions = self.model.predict(X_val)
        self.accuracy = accuracy_score(y_val, predictions)  
        print("Validation Accuracy:", self.accuracy)
        
        # Log model and metrics to MLFlow
        mlflow.set_tracking_uri('http://localhost:5000')  
        mlflow.set_experiment('phone-price-experiment')
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path="model", registered_model_name="PhonePriceModel")
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_metric("accuracy", self.accuracy)
        self.next(self.end)

    @step
    def end(self):
        print("Training flow completed.")
        print("Final Accuracy:", self.accuracy)

if __name__ == '__main__':
    PhonePriceTrainFlow()
