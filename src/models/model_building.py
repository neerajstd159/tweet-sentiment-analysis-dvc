import numpy as np
import pandas as pd
import logging
import os
import json
import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report

def setup_logger(log_file: str) -> logging.Logger:
    try:
        logger = logging.Logger('Feature engineering')
        logger.setLevel(logging.DEBUG)

        # File handler
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setLevel(logging.INFO)

        # Console handler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)

        return logger
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None
    

def load_data(url: str, logger: logging.Logger) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.info("dataset loaded successfully")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"error while loading data: {e}")
        return pd.DataFrame()
    

def calculate_matrix(y_pred: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)
    f1_scor = f1_score(y_pred, y_test)
    classification = classification_report(y_pred, y_test)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "f1_score": f1_scor,
        "classification_report": classification
    }

    return metrics
    

def main():
    try:
        logger = setup_logger( '././log/app.log')
        train_url = os.path.join("./data", "processed", "train_tfidf.csv")
        test_url = os.path.join("./data", "processed", "test_tfidf.csv")
        metrics_path = os.path.join('./metrics', 'metrics.json')
        model_path = os.path.join('./models', 'bernoulliNB.pkl')

        train_data = load_data(train_url, logger)
        test_data = load_data(test_url, logger)
        
        X_train = train_data.drop(columns=['sentiment']).values
        y_train = train_data['sentiment'].values
        X_test = test_data.drop(columns=['sentiment']).values
        y_test = test_data['sentiment'].values

        bnb = BernoulliNB()
        bnb.fit(X_train, y_train)
        y_pred = bnb.predict(X_test)

        metrics = calculate_matrix(y_pred, y_test)
        logger.info(f"metrics: {metrics}")

        # save matrics
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)

        # save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as file:
            pickle.dump(bnb, file)
        logger.info(f"model saved successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()