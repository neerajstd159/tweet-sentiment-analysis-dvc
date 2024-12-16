import numpy as np
import pandas as pd
import os
import logging
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


def setup_logging(log_file: str = '././log/app.log') -> logging.Logger:
    logger = logging.Logger('data preprocessing')
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


def load_dataset(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.info('Dataset loaded')
        return df
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
        return pd.DataFrame()
    except pd.errors.EmptyDataError as e:
        logger.error(f"Dataset is empty: {e}")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"Parse error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"unexpected error occured: {e}")
        return pd.DataFrame()
    

def replace_pattern(pattern: str, text: str) -> str:
    try:
        text = re.sub(pattern, '', text)
        return text
    except re.error as e:
        logger.error(f"Invalid regex pattern: {e}")
        return ''
    except TypeError as e:
        logger.error(f"Type error: {e} the text parameter must be string")
        return ''
    except Exception as e:
        logger.error(f"unexpected error occured: {e}")
        return ''
    
def replacepattern_with_pattern(pattern1: str,pattern2: str, text: str) -> str:
    try:
        text = re.sub(pattern1, pattern2, text)
        return text
    except re.error as e:
        logger.error(f"Invalid regex pattern: {e}")
        return ''
    except TypeError as e:
        logger.error(f"Type error: {e} the text parameter must be string")
        return ''
    except Exception as e:
        logger.error(f"unexpected error occured: {e}")
        return ''
    
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Replace 4 by 1 in sentiment
        df['sentiment'] = df['sentiment'].replace(4,1)

        # Change to lower case
        df['tweet'] = df['tweet'].apply(lambda x: x.lower())

        # Remove urls
        url_pattern = r'https?://[^\s]*|www\.[^\s]*'
        df['tweet'] = df['tweet'].apply(lambda  x: replace_pattern(url_pattern, x))

        # Remove @username
        user_pattern = r'@[^\s]*'
        df['tweet'] = df['tweet'].apply(lambda  x: replace_pattern(user_pattern, x))

        # Replace a letter with 2 occurences if it has more than 2 occurances
        search_pattern = r'(.)\1{2,}'
        replace_with = r'\1\1'
        df['tweet'] = df['tweet'].apply(lambda  x: replacepattern_with_pattern(search_pattern, replace_with, x))

        # Remove non-alpha-numeric letters
        alpha_num_pattern = r'[^a-zA-Z0-9 ]'
        df['tweet'] = df['tweet'].apply(lambda  x: replace_pattern(alpha_num_pattern, x))

        return df
    
    except Exception as e:
        logger.error(f"Unexpected error occured: {e}")
        return pd.DataFrame()
    
def dump_data(df: pd.DataFrame) -> None:
    if df.empty:
        logger.error('Dataframe is empty')
        return
    output_dir = '././data/processed'
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return
    try:
        df.to_csv('././data/interim/pre_processed_df.csv', index=False)
        logger.info('Data successfully written to ././data/interim/pre_processed_df.csv')
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
    except Exception as e:
        logger.error(f'unexpected error occured {e}')


def remove_stopwords(text: str) -> str:
    try:
        stopword = set(stopwords.words('english'))
        newtext = [word for word in text.split() if word not in stopword]
        return " ".join(newtext)
    except Exception as e:
        logger.error(f"Some error occured: {e}")
        return ""
    

def apply_lemmatization(lemmatizer: WordNetLemmatizer, text: str) -> str:
    try:
        newtext = [lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(newtext)
    except TypeError as e:
        logger.error(f"type error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Some error occured: {e}")
        return ""

logger = setup_logging()

df_url = '././data/raw/filtered_df.csv'
df = load_dataset(df_url)

df = preprocessing(df)

# Remove stopwords
df['tweet'] = df['tweet'].apply(remove_stopwords)

# Apply lemmatization
lemmatizer = WordNetLemmatizer()
df['tweet'] = df['tweet'].apply(lambda x: apply_lemmatization(lemmatizer, x))

dump_data(df)