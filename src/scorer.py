import json
import os
import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
model.load_model('./models/my_catboost.cbm')

# Define optimal threshold
model_th = 0.98
logger.info('Pretrained model imported successfully...')


def save_feature_importances(output_dir: str) -> str:
    """
    Save top-5 feature importances as JSON file.
    Bonus feature for 8-10 grade.
    """
    feature_names = model.feature_names_
    importances = model.get_feature_importance()
    
    # Sort by importance and get top 5
    importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    feature_importances = {name: float(imp) for name, imp in importance_pairs}
    
    output_path = os.path.join(output_dir, 'feature_importances.json')
    with open(output_path, 'w') as f:
        json.dump(feature_importances, f, indent=2, ensure_ascii=False)
    
    logger.info('Feature importances saved to: %s', output_path)
    logger.info('Top 5 features: %s', feature_importances)
    
    return output_path


def save_score_distribution(probabilities: np.ndarray, output_dir: str, filename_suffix: str = '') -> str:
    """
    Save density plot of predicted scores as PNG.
    Bonus feature for 8-10 grade.
    """
    if filename_suffix:
        output_path = os.path.join(output_dir, f'score_distribution_{filename_suffix}.png')
    else:
        output_path = os.path.join(output_dir, 'score_distribution.png')
    
    plt.figure(figsize=(10, 6))
    
    # Plot density
    sns.kdeplot(probabilities, fill=True, color='steelblue', alpha=0.7)
    
    # Styling
    plt.title('Распределение предсказанных скоров модели', fontsize=14, fontweight='bold')
    plt.xlabel('Вероятность мошенничества', fontsize=12)
    plt.ylabel('Плотность', fontsize=12)
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_score = np.mean(probabilities)
    median_score = np.median(probabilities)
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=1.5, label=f'Среднее: {mean_score:.4f}')
    plt.axvline(median_score, color='green', linestyle='--', linewidth=1.5, label=f'Медиана: {median_score:.4f}')
    plt.axvline(model_th, color='orange', linestyle='-', linewidth=2, label=f'Порог: {model_th}')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info('Score distribution plot saved to: %s', output_path)
    
    return output_path


# Make prediction
def make_pred(dt, path_to_file):
    """
    Make predictions and return submission dataframe + probabilities.
    """
    # Get probabilities
    probabilities = model.predict_proba(dt)[:, 1]
    
    # Make submission dataframe
    submission = pd.DataFrame({
        'index': pd.read_csv(path_to_file).index,
        'prediction': (probabilities > model_th).astype(int)
    })
    logger.info('Prediction complete for file: %s', path_to_file)
    
    # Return both submission and probabilities (for score distribution plot)
    return submission, probabilities
