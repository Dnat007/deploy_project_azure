import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from azure.storage.blob import BlobServiceClient


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)







def download_artifacts():

    connection_string = os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING"
    )

    if not connection_string:
        raise Exception(
            "AZURE_STORAGE_CONNECTION_STRING not found"
        )

    os.makedirs("artifacts", exist_ok=True)

    blob_service_client = (
        BlobServiceClient.from_connection_string(
            connection_string
        )
    )

    container_name = "mlopscontainer"

    files = [
        "model.pkl",
        "preprocessor.pkl"
    ]

    for file_name in files:

        local_file = os.path.join(
            "artifacts",
            file_name
        )

        if os.path.exists(local_file):

            print(
                f"{file_name} already exists"
            )

            continue

        print(
            f"Downloading {file_name}"
        )

        blob_client = (
            blob_service_client.get_blob_client(
                container=container_name,
                blob=file_name
            )
        )

        with open(local_file, "wb") as f:

            download_stream = (
                blob_client.download_blob()
            )

            f.write(
                download_stream.readall()
            )

        print(
            f"{file_name} downloaded"
        )