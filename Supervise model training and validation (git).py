import numpy as np
import pandas as pd
import time
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from scipy.stats import entropy, gaussian_kde
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


def compute_kl_divergence(real_X, predicted_X):
    """
    Compute KL divergence between real and predicted data distributions
    Args:
        real_X: Ground truth data
        predicted_X: Model predicted data
    Returns:
        KL divergence value
    """
    real_X_flat = real_X.flatten()
    predicted_X_flat = predicted_X.flatten()

    # Estimate probability density functions using kernel density estimation
    kde_real = gaussian_kde(real_X_flat)
    kde_pred = gaussian_kde(predicted_X_flat)

    # Generate sample points for PDF evaluation
    x_samples = np.linspace(min(real_X_flat.min(), predicted_X_flat.min()),
                            max(real_X_flat.max(), predicted_X_flat.max()), 1000)

    p_real = kde_real(x_samples)
    q_pred = kde_pred(x_samples)

    # Add small constant to avoid division by zero
    p_real += 1e-10
    q_pred += 1e-10

    kl_div = entropy(p_real, q_pred)
    return kl_div


def compute_top_k_error(real_values, predicted_values, k=3):
    """
    Compute Top-K matching error between real and predicted values
    Args:
        real_values: Ground truth values
        predicted_values: Model predictions
        k: Number of nearest neighbors to consider
    Returns:
        Average Top-K error
    """
    errors = []
    for pred in predicted_values:
        # Calculate Euclidean distances to all real values
        distances = np.linalg.norm(real_values - pred, axis=1)
        # Select K smallest distances and compute mean
        top_k_errors = np.sort(distances)[:k]
        errors.append(np.mean(top_k_errors))
    return np.mean(errors)


def train_and_save_models(train_file, validation_file, model_file):
    """
    Train forward and inverse models for antenna design and save to file
    Args:
        train_file: Path to training data Excel file
        validation_file: Path to validation data Excel file
        model_file: Output path for saved models
    """
    # Load training data
    df_train = pd.read_excel(train_file, skiprows=1)
    X_train = df_train.iloc[:, :4].values  # Structural parameters (features)
    T_train = df_train.iloc[:, 4:7].values  # Performance parameters (targets)

    # Load validation data
    df_validation = pd.read_excel(validation_file, skiprows=1)
    X_validation = df_validation.iloc[:, :4].values
    T_validation = df_validation.iloc[:, 4:7].values

    # Calculate Min-Max normalization parameters
    min_values_X, max_values_X = X_train.min(axis=0), X_train.max(axis=0)
    min_values_T, max_values_T = T_train.min(axis=0), T_train.max(axis=0)

    # Normalize data to [0, 1] range
    X_normalized_train = (X_train - min_values_X) / (max_values_X - min_values_X)
    X_normalized_validation = (X_validation - min_values_X) / (max_values_X - min_values_X)
    T_normalized_train = (T_train - min_values_T) / (max_values_T - min_values_T)
    T_normalized_validation = (T_validation - min_values_T) / (max_values_T - min_values_T)

    # Train Model 1: Inverse model (Performance → Structure)
    print("Training Model 1: Predicting Structural Parameters from Performance Targets...")
    model1 = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=15, random_state=42, n_jobs=-1)
    multi_output_stacking1 = MultiOutputRegressor(model1)
    multi_output_stacking1.fit(T_normalized_train, X_normalized_train)

    # Train Model 2: Forward model (Structure → Performance)
    print("Training Model 2: Predicting Performance Parameters from Structural Parameters...")
    model2 = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=15, random_state=42, n_jobs=-1)
    multi_output_stacking2 = MultiOutputRegressor(model2)
    multi_output_stacking2.fit(X_normalized_train, T_normalized_train)

    # Evaluate Model 1 (Inverse model)
    X_normalized_pred_model1 = multi_output_stacking1.predict(T_normalized_validation)
    X_pred_model1 = X_normalized_pred_model1 * (max_values_X - min_values_X) + min_values_X  # Denormalize

    rmse_model1 = np.sqrt(mean_squared_error(X_validation, X_pred_model1))
    r2_model1 = r2_score(X_validation, X_pred_model1)
    kl_div_model1 = compute_kl_divergence(X_validation, X_pred_model1)
    top_k_error_model1 = compute_top_k_error(X_validation, X_pred_model1, k=3)
    mse_model1_normalized = mean_squared_error(X_normalized_validation, X_normalized_pred_model1)

    print(f"Inverse Model Performance:")
    print(f"  - RMSE: {rmse_model1:.4f}")
    print(f"  - R²: {r2_model1:.4f}")
    print(f"  - KL Divergence: {kl_div_model1:.4f}")
    print(f"  - Top-K Error: {top_k_error_model1:.4f}")
    print(f"  - Normalized MSE: {mse_model1_normalized:.4f}")

    # Evaluate Model 2 (Forward model)
    T_normalized_pred_model2 = multi_output_stacking2.predict(X_normalized_validation)
    T_pred_model2 = T_normalized_pred_model2 * (max_values_T - min_values_T) + min_values_T  # Denormalize

    rmse_model2 = np.sqrt(mean_squared_error(T_validation, T_pred_model2))
    r2_model2 = r2_score(T_validation, T_pred_model2)
    kl_div_model2 = compute_kl_divergence(T_validation, T_pred_model2)
    top_k_error_model2 = compute_top_k_error(T_validation, T_pred_model2, k=3)
    mse_model2_normalized = mean_squared_error(T_normalized_validation, T_normalized_pred_model2)

    print(f"Forward Model Performance:")
    print(f"  - RMSE: {rmse_model2:.4f}")
    print(f"  - R²: {r2_model2:.4f}")
    print(f"  - KL Divergence: {kl_div_model2:.4f}")
    print(f"  - Top-K Error: {top_k_error_model2:.4f}")
    print(f"  - Normalized MSE: {mse_model2_normalized:.4f}")

    # Save trained models and normalization parameters
    model_data = {
        'model1': multi_output_stacking1,  # Inverse model
        'model2': multi_output_stacking2,  # Forward model
        'min_values_X': min_values_X,
        'max_values_X': max_values_X,
        'min_values_T': min_values_T,
        'max_values_T': max_values_T
    }
    joblib.dump(model_data, model_file)
    print(f"Models successfully saved to {model_file}")


def load_and_predict(model_file, target_file):
    """
    Load trained models and perform prediction on new target data
    Args:
        model_file: Path to saved model file
        target_file: Path to target data Excel file
    """
    # Load trained models and parameters
    model_data = joblib.load(model_file)
    model1 = model_data['model1']  # Inverse model
    model2 = model_data['model2']  # Forward model
    min_values_X = model_data['min_values_X']
    max_values_X = model_data['max_values_X']
    min_values_T = model_data['min_values_T']
    max_values_T = model_data['max_values_T']

    # Load target performance specifications
    target_df = pd.read_excel(target_file)
    T_new = target_df.iloc[:, :3].values  # Target performance parameters

    # Normalize target values
    T_normalized_new = (T_new - min_values_T) / (max_values_T - min_values_T)

    # Predict structural parameters using inverse model
    X_normalized_pred = model1.predict(T_normalized_new)
    X_pred = X_normalized_pred * (max_values_X - min_values_X) + min_values_X  # Denormalize
    print("Predicted Structural Parameters:")
    print(np.round(X_pred, 1))

    # Verify predictions using forward model
    T_normalized_pred = model2.predict(X_normalized_pred)
    T_pred = T_normalized_pred * (max_values_T - min_values_T) + min_values_T  # Denormalize
    print("Predicted Performance Parameters:")
    print(np.round(T_pred, 2))

    # Calculate prediction accuracy
    top_k_error = compute_top_k_error(T_new, T_pred, k=3)
    print(f"Top-K Matching Error: {top_k_error:.4f}")


if __name__ == "__main__":
    # Train and save models
    train_and_save_models('Alldata.xlsx', 'banddata100.xlsx', 'two_models.pkl')

    # Load models and perform predictions
    load_and_predict('two_models.pkl', 'target_data.xlsx')