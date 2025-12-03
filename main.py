import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    """
    Load the concrete dataset from Excel file using pandas.
    
    Returns:
        pd.DataFrame: DataFrame containing the concrete data
    """
    df = pd.read_excel("data/concrete_data.xlsx")
    return df


def explore_data(df):
    """
    Explore the dataset using pandas operations.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        str: Name of the feature most correlated with concrete strength
    """
    print("DATA EXPLORATION")
    
    # Print shape
    print(f"\nDataset shape: {df.shape}")
    
    # Print column names
    print("\nColumn names:")
    print(df.columns.tolist())
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Correlation matrix
    corr = df.corr()
    target = 'Concrete Compressive Strength'
    
    # Correlation with target, excluding itself
    corr_target = corr[target].drop(target)
    most_correlated_feature = corr_target.abs().idxmax()
    
    print(f"\nMost correlated feature with strength: {most_correlated_feature}")
    
    return most_correlated_feature


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the data using pandas operations.
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
    """
    target = 'Concrete Compressive Strength'
    
    X = df.drop(columns=[target])
    y = df[target]
    
    feature_names = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics.
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'mse': mse, 'r2': r2}


def get_feature_importance(model, feature_names, top_n=3):
    """
    Return top N features sorted by importance as a pandas DataFrame.
    """
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort descending and select top_n
    top_features_df = importance_df.sort_values(
        by='Importance', ascending=False
    ).head(top_n)
    
    return top_features_df


def main():
    """Main function to run the entire pipeline."""
    print("Concrete Strength Prediction - Assignment 6")
    
    # Load data
    print("\n1. Loading data from Excel file...")
    df = load_data()
    print(f"   Loaded {len(df)} samples from Excel")
    
    # Explore data with pandas
    print("\n2. Exploring data with pandas...")
    most_correlated = explore_data(df)
    
    # Preprocess data
    print("\n3. Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train model
    print("\n4. Training Random Forest model...")
    model = train_model(X_train, y_train)
    print("   Model trained successfully")
    
    # Evaluate model
    print("\n5. Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"   Mean Squared Error: {metrics['mse']:.2f}")
    print(f"   R-squared Score: {metrics['r2']:.4f}")
    
    # Feature importance
    print("\n6. Top 3 Most Important Features (pandas DataFrame):")
    top_features_df = get_feature_importance(model, feature_names)
    print(top_features_df.to_string(index=False))
    

if __name__ == "__main__":
    main()
