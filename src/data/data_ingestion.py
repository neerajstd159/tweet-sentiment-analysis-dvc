import numpy as np
import pandas as pd
import chardet
import logging
import os

def setup_logging(log_file: str = '././log/app.log') -> logging.Logger:
    logger = logging.getLogger('data_ingestion')
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

    # Add handlers to the logger
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    return logger

logger = setup_logging()

def load_dataset(url: str) -> pd.DataFrame:
    try:
        columns = ['sentiment', 'ids', 'date', 'flag', 'user', 'tweet']
        df = pd.read_csv(url, encoding="ISO-8859-1", names=columns)
        logger.info('data successfully loaded')
        return filter_data(df)
    except FileNotFoundError:
        logger.error('file not found')
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.error('empty file')
        return pd.DataFrame()
    except pd.errors.ParserError:
        logger.error('parsing error')
        return pd.DataFrame()
    except Exception as e:
        logger.error('something went wrong')
        return pd.DataFrame()
    
def filter_data(df: pd.DataFrame, n=800000) -> pd.DataFrame:
    try:
        df = df[['sentiment', 'tweet']]
        positive_class = df[df['sentiment']==4]
        negative_class = df[df['sentiment']==0]

        if len(positive_class) < n:
            logger.warning(f"insufficient positive samples, only {len(positive_class)} available")
            positive_sample = positive_class.sample(n=len(positive_class), random_state=42)
        else:
            positive_sample = positive_class.sample(n=n, random_state=42)
        
        if len(negative_class) < n:
            logger.warning(f"insufficient negative samples, only {len(negative_class)} available")
            negative_sample = negative_class.sample(n=len(negative_class), random_state=42)
        else:
            negative_sample = negative_class.sample(n=n, random_state=42)

        filtered_df = pd.concat([positive_sample, negative_sample])
        filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

        return filtered_df
    except KeyError as e:
        logger.error(f"Keyerror: {e}")
        return pd.DataFrame()
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occured: {e}")
        return pd.DataFrame()
    
def dump_data(df: pd.DataFrame) -> None:
    if df.empty:
        logger.error('Dataframe is empty')
        return
    output_dir = '././data/interim'
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return
    try:
        df.to_csv('././data/raw/filtered_df.csv', index=False)
        logger.info('Data successfully written to ././data/raw/filtered_df.csv')
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
    except Exception as e:
        logger.error(f'unexpected error occured {e}')

df = load_dataset('././data/raw/tweets_sentiment.csv')
dump_data(df)