"""
Quick setup script for Google Colab
Run this first in Colab to set up the environment
"""

# Install packages
import subprocess
import sys

packages = [
    'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
    'lightgbm', 'xgboost', 'catboost', 'optuna', 'imbalanced-learn'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("All packages installed successfully!")
print("\nNext steps:")
print("1. Upload train.csv and test.csv to Colab")
print("2. Run complete_project_pipeline.py or use the notebook")

