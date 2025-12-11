#!/usr/bin/env python3
"""
Training script for CatBoost fraud detection model.
Run this LOCALLY before building Docker image.

Usage:
    python train_model.py
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from geopy.distance import great_circle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_DIR = Path(__file__).parent
TRAIN_DATA_PATH = PROJECT_DIR / "train_data" / "train.csv"
MODEL_PATH = PROJECT_DIR / "models" / "my_catboost.cbm"

RANDOM_STATE = 42


def add_time_features(df):
    """Extract time features from transaction_time."""
    logger.info('Adding time features...')
    df = df.copy()
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    dt = df['transaction_time'].dt
    df['hour'] = dt.hour
    df['year'] = dt.year
    df['month'] = dt.month
    df['day_of_month'] = dt.day
    df['day_of_week'] = dt.dayofweek
    df.drop(columns='transaction_time', inplace=True)
    return df


def add_distance_features(df):
    """Calculate distance between customer and merchant."""
    logger.info('Calculating distances...')
    df = df.copy()
    df['distance'] = df.apply(
        lambda x: great_circle(
            (x['lat'], x['lon']), 
            (x['merchant_lat'], x['merchant_lon'])
        ).km,
        axis=1
    )
    df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'], inplace=True)
    return df


def prepare_features(df, categorical_cols, n_cats=50):
    """
    Prepare features for training:
    - Category encoding (group rare categories)
    - Time features
    - Distance features
    - Log transformation
    """
    df = df.copy()
    target_col = 'target'
    
    # Drop columns we don't need
    drop_cols = ['name_1', 'name_2', 'street', 'post_code']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Add time features
    df = add_time_features(df)
    
    # Encode categorical columns (group rare categories)
    logger.info('Encoding categorical features...')
    for col in categorical_cols:
        # Get category counts
        temp_df = df.groupby(col, dropna=False)[[target_col]]\
            .count()\
            .sort_values(target_col, ascending=False)\
            .reset_index()\
            .set_axis([col, 'count'], axis=1)\
            .reset_index()
        
        temp_df['index'] = temp_df.apply(
            lambda x: np.nan if pd.isna(x[col]) else x['index'], 
            axis=1
        )
        
        new_col = col + '_cat'
        temp_df[new_col] = [
            'cat_NAN' if pd.isna(x) 
            else 'cat_' + str(int(x)) if x < n_cats 
            else f'cat_{n_cats}+' 
            for x in temp_df['index']
        ]
        
        df = df.merge(temp_df[[col, new_col]], how='left', on=col)
        df.drop(columns=col, inplace=True)
    
    # Add distance features
    df = add_distance_features(df)
    
    # Log transform continuous features
    continuous_cols = ['amount', 'population_city', 'distance']
    for col in continuous_cols:
        df[col + '_log'] = np.log(df[col] + 1)
        df.drop(columns=col, inplace=True)
    
    return df


def train_model():
    """Main training function."""
    logger.info("=" * 50)
    logger.info("Starting model training")
    logger.info("=" * 50)
    
    # Check if training data exists
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_DATA_PATH}\n"
            "Please copy train.csv:\n"
            "  cp ~/Downloads/teta-ml-1-2025/train.csv ./train_data/"
        )
    
    # Load data
    logger.info(f"Loading data from {TRAIN_DATA_PATH}...")
    df = pd.read_csv(TRAIN_DATA_PATH)
    logger.info(f"Loaded {len(df)} rows")
    
    # Define categorical columns
    categorical_cols = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
    
    # Prepare features
    logger.info("Preparing features...")
    df = prepare_features(df, categorical_cols)
    
    # Split features and target
    target_col = 'target'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    # Identify categorical features for CatBoost
    cat_features = [col for col in X.columns if col.endswith('_cat') or col in ['hour', 'year', 'month', 'day_of_month', 'day_of_week']]
    
    # Fill NaN in categorical features
    for col in cat_features:
        X[col] = X[col].fillna('cat_NAN').astype(str)
    
    # Fill NaN in numerical features
    num_features = [col for col in X.columns if col not in cat_features]
    for col in num_features:
        X[col] = X[col].fillna(X[col].median())
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Train CatBoost
    logger.info("Training CatBoost model...")
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=RANDOM_STATE,
        verbose=50,
        early_stopping_rounds=30,
        task_type='CPU',
        cat_features=cat_features
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    # Print results
    best_score = model.get_best_score()
    logger.info(f"Best validation score: {best_score}")
    
    # Save model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    logger.info(f"Model saved to {MODEL_PATH}")
    
    # Print top feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 feature importances:")
    print(feature_importance.head(10).to_string(index=False))
    
    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info("=" * 50)
    
    return model


if __name__ == "__main__":
    train_model()

