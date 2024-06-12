from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score


class Train:
    def __init__(self, configs):
        self.config = configs
        self.filename = self.config.get('filename')
        self.input_feature = self.config.get('input_feature')
        self.target = self.config.get('target')
        self.stand_scl = self.config.get('stand_scl')
        self.one_hot = self.config.get('one_hot')
        self.features = self.config.get('features')

    def data_frame(self, df, features):
        return pd.DataFrame(df[features])








    def split_df(self, df):
        X = df[self.input_feature]
        y = df[self.target]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def create_pipeline(self,df):
        categorical_transformer = Pipeline(steps=[

            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        numeric_transformer = Pipeline(steps=[

            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.stand_scl),
                ('cat', categorical_transformer, self.one_hot)
            ]
        )
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])
        return model_pipeline

    def train_model(self):
        df = pd.read_csv(f'Input/{self.filename}.csv')
        df = self.data_frame(df, self.features)

        x_train, x_test, y_train, y_test = self.split_df(df)
        model_pipeline = self.create_pipeline(df)

        model_pipeline.fit(x_train, y_train)
        y_pred=model_pipeline.predict(x_test)
        score=accuracy_score(y_test,y_pred)
        print(score)
        return model_pipeline, x_train, x_test, y_train, y_test



