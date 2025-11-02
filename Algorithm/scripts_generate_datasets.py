"""
Sample Data Generator
Generates sample datasets for testing the GA feature selection system.
Run this script to create sample CSV files in sample_datasets/ directory.

Usage: python generate_sample_data.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris, 
    load_breast_cancer,
    load_wine,
    make_classification
)
import os

def generate_iris_dataset():
    """
    Generate Iris dataset (4 features, 150 samples)
    Classic dataset for testing - Easy complexity
    """
    print("Generating Iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['target'] = iris.target
    
    # Add metadata as comments in description
    df.attrs['description'] = "Iris Dataset: 4 features, 3 classes, 150 samples"
    
    return df


def generate_breast_cancer_dataset():
    """
    Generate Breast Cancer Wisconsin dataset (30 features, 569 samples)
    Good for demonstrating feature reduction - Medium complexity
    """
    print("Generating Breast Cancer dataset...")
    cancer = load_breast_cancer()
    df = pd.DataFrame(
        data=cancer.data,
        columns=cancer.feature_names
    )
    df['target'] = cancer.target
    
    df.attrs['description'] = "Breast Cancer: 30 features, 2 classes, 569 samples"
    
    return df


def generate_wine_dataset():
    """
    Generate Wine dataset (13 features, 178 samples)
    Good balance of features and samples - Medium complexity
    """
    print("Generating Wine dataset...")
    wine = load_wine()
    df = pd.DataFrame(
        data=wine.data,
        columns=wine.feature_names
    )
    df['target'] = wine.target
    
    df.attrs['description'] = "Wine Dataset: 13 features, 3 classes, 178 samples"
    
    return df


def generate_synthetic_high_dimensional():
    """
    Generate synthetic high-dimensional dataset (50 features, 1000 samples)
    Challenges GA with many irrelevant features - High complexity
    """
    print("Generating Synthetic High-Dimensional dataset...")
    
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=15,      # Only 15 features are actually useful
        n_redundant=10,        # 10 features are redundant
        n_repeated=5,          # 5 features are repeated
        n_classes=2,
        random_state=42,
        shuffle=False
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(50)]
    
    df = pd.DataFrame(data=X, columns=feature_names)
    df['target'] = y
    
    df.attrs['description'] = "Synthetic: 50 features (15 informative), 2 classes, 1000 samples"
    
    return df


def generate_synthetic_multiclass():
    """
    Generate synthetic multiclass dataset (20 features, 800 samples, 4 classes)
    Tests GA with multiple classes - Medium complexity
    """
    print("Generating Synthetic Multiclass dataset...")
    
    X, y = make_classification(
        n_samples=800,
        n_features=20,
        n_informative=12,
        n_redundant=5,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=42
    )
    
    feature_names = [f'feature_{i+1}' for i in range(20)]
    
    df = pd.DataFrame(data=X, columns=feature_names)
    df['target'] = y
    
    df.attrs['description'] = "Synthetic Multiclass: 20 features, 4 classes, 800 samples"
    
    return df


def generate_small_dataset_for_quick_test():
    """
    Generate very small dataset for quick testing (5 features, 100 samples)
    Fast execution for development and debugging
    """
    print("Generating Small Test dataset...")
    
    np.random.seed(42)
    
    # Create 5 features with different characteristics
    n_samples = 100
    
    df = pd.DataFrame({
        'age': np.random.randint(20, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'score': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'flag': np.random.choice([0, 1], n_samples)
    })
    
    # Create target based on simple rule
    df['target'] = ((df['age'] > 50) & (df['income'] > 45000)).astype(int)
    
    df.attrs['description'] = "Small Test: 5 features, 2 classes, 100 samples"
    
    return df


def save_dataset_info():
    """
    Create a summary file with information about all datasets
    """
    info = """SAMPLE DATASETS INFORMATION
===========================

1. iris.csv
   - Features: 4
   - Samples: 150
   - Classes: 3 (Setosa, Versicolor, Virginica)
   - Complexity: Easy
   - Purpose: Initial testing and validation
   - Features: sepal length, sepal width, petal length, petal width

2. breast_cancer.csv
   - Features: 30
   - Samples: 569
   - Classes: 2 (Malignant, Benign)
   - Complexity: Medium
   - Purpose: Demonstrate feature reduction capability
   - Features: Mean, SE, and "worst" values for radius, texture, perimeter, etc.

3. wine.csv
   - Features: 13
   - Samples: 178
   - Classes: 3 (Wine types)
   - Complexity: Medium
   - Purpose: Balanced dataset for general testing
   - Features: Alcohol, Malic acid, Ash, Alkalinity, Magnesium, etc.

4. synthetic_high_dim.csv
   - Features: 50 (only 15 are informative)
   - Samples: 1000
   - Classes: 2
   - Complexity: High
   - Purpose: Test GA with many irrelevant features
   - Challenge: Find the 15 truly useful features among 50

5. synthetic_multiclass.csv
   - Features: 20
   - Samples: 800
   - Classes: 4
   - Complexity: Medium
   - Purpose: Test multiclass classification
   - Challenge: Handle multiple classes effectively

6. small_test.csv
   - Features: 5
   - Samples: 100
   - Classes: 2
   - Complexity: Easy
   - Purpose: Quick testing during development
   - Fast execution for debugging

USAGE RECOMMENDATION:
--------------------
- Start with small_test.csv for quick validation
- Use iris.csv for initial algorithm testing
- Test with breast_cancer.csv to show feature reduction
- Challenge your GA with synthetic_high_dim.csv
- Demo with wine.csv for presentation

NOTE:
-----
All datasets include a 'target' column as the last column.
All other columns are features that can be selected by the GA.
"""
    
    with open('sample_datasets/DATASETS_INFO.txt', 'w') as f:
        f.write(info)
    
    print("\nDataset information saved to DATASETS_INFO.txt")


def main():
    """
    Main function to generate all sample datasets
    """
    # Create sample_datasets directory if it doesn't exist
    os.makedirs('sample_datasets', exist_ok=True)
    
    print("=" * 60)
    print("GENERATING SAMPLE DATASETS FOR GA FEATURE SELECTION")
    print("=" * 60)
    print()
    
    # Generate all datasets
    datasets = {
        'iris.csv': generate_iris_dataset(),
        'breast_cancer.csv': generate_breast_cancer_dataset(),
        'wine.csv': generate_wine_dataset(),
        'synthetic_high_dim.csv': generate_synthetic_high_dimensional(),
        'synthetic_multiclass.csv': generate_synthetic_multiclass(),
        'small_test.csv': generate_small_dataset_for_quick_test()
    }
    
    # Save all datasets
    print("\n" + "=" * 60)
    print("SAVING DATASETS")
    print("=" * 60)
    
    for filename, df in datasets.items():
        filepath = os.path.join('sample_datasets', filename)
        df.to_csv(filepath, index=False)
        print(f"[OK] Saved: {filename} ({len(df)} rows, {len(df.columns)-1} features)")
    
    # Save dataset information
    save_dataset_info()
    
    print("\n" + "=" * 60)
    print("ALL DATASETS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nDatasets location: sample_datasets/")
    print("\nYou can now use these datasets to test your GA feature selection system.")


if __name__ == '__main__':
    main()

