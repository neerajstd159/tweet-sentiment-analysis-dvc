import numpy as np
import pandas as pd
import logging
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from typing import Tuple
from scipy.sparse import csr_matrix
import os

def setup_logging(log_file: str = '././log/app.log') -> logging.Logger:
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
    
def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.info("Dataset loaded")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Some error occured: {e}")
        return pd.DataFrame()


def apply_tfidf(X_train: list[str], X_test: list[str], max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 1)) -> Tuple[csr_matrix, csr_matrix]:
    try:
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        logger.debug('successfully applied tfidf')
        return X_train_tfidf, X_test_tfidf
    except Exception as e:
        logger.error(f"Some error while applying tfidf: {e}")
        return csr_matrix((0,0)), csr_matrix((0,0))


def dump_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    if train_data.empty or test_data.empty:
        logger.error('Either Dataframe is empty')
        return
    output_dir = os.path.join('./data', 'processed')
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return
    try:
        train_data.to_csv(os.path.join("./data", "processed", "train_tfidf.csv"), index=False)
        test_data.to_csv(os.path.join("./data", "processed", "test_tfidf.csv"), index=False)
        logger.info('Data successfully written processed dir')
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
    except Exception as e:
        logger.error(f'unexpected error occured {e}')

def main():
    try:
        df_url = '././data/interim/pre_processed_df.csv'
        df = load_data(df_url)
        X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['sentiment'], test_size=0.2)
        X_train_tfidf, X_test_tfidf = apply_tfidf(X_train, X_test)

        X_train_dense = pd.DataFrame(X_train_tfidf.toarray())
        X_test_dense = pd.DataFrame(X_test_tfidf.toarray())

        train_data = pd.concat([X_train_dense, y_train.reset_index(drop=True)], axis=1)
        test_data = pd.concat([X_test_dense, y_test.reset_index(drop=True)], axis=1)
        dump_data(train_data, test_data)
    except Exception as e:
        logger.error(f"unexpected error: {e}")


if __name__ == "__main__":
    logger = setup_logging()
    main()