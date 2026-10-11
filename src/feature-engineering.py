import pandas as pd
import os
import pickle
import logging
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, "feature_engineering.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except Exception as e:
        logger.error("Failed to load parameters: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file and fill missing values."""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise


def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
) -> tuple:
    """Fit TF-IDF on training text and transform test text."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        x_train = train_data["text"].astype(str).values
        y_train = train_data["target"].values
        x_test = test_data["text"].astype(str).values
        y_test = test_data["target"].values

        # Fit only on training data to prevent data leakage
        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        # Save the exact fitted vectorizer used for both datasets
        os.makedirs("models", exist_ok=True)
        vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")

        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        logger.info("Fitted TF-IDF vectorizer saved to %s", vectorizer_path)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df["label"] = y_test

        logger.debug("TF-IDF transformation completed successfully")

        return train_df, test_df

    except Exception as e:
        logger.error("TF-IDF transformation failed: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save a DataFrame to a CSV file."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)

    except Exception as e:
        logger.error("Failed to save data: %s", e)
        raise


def main():
    try:
        params = load_params("params.yaml")
        max_features = params["feature_engineering"]["max_features"]

        train_data = load_data("data/interim/train_processed.csv")
        test_data = load_data("data/interim/test_processed.csv")

        train_df, test_df = apply_tfidf(
            train_data, test_data, max_features
        )

        save_data(
            train_df,
            os.path.join("data", "processed", "train_tfidf.csv")
        )
        save_data(
            test_df,
            os.path.join("data", "processed", "test_tfidf.csv")
        )

        logger.info("Feature engineering completed successfully")

    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        raise


if __name__ == "__main__":
    main()