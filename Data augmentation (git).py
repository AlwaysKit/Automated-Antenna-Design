import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Load original dataset
input_path = "banddata500.xlsx"
df = pd.read_excel(input_path, skiprows=1)

# Extract features (first 4 columns) and targets (next 3 columns)
features = df.iloc[:, :4].values
targets = df.iloc[:, 4:7].values

# Normalize features and targets using Min-Max scaling
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

features_normalized = scaler_X.fit_transform(features)
targets_normalized = scaler_Y.fit_transform(targets)

# Define regression models for each target column
models = {
    0: SVR(kernel='rbf', C=100, gamma=5, epsilon=0.01),  # Model for first target
    1: SVR(kernel='rbf', C=100, gamma=5, epsilon=0.01),  # Model for second target
    2: SVR(kernel='rbf', C=100, gamma=5, epsilon=0.01)  # Model for third target
}

# Train individual model for each target column
for target_idx in range(targets.shape[1]):
    print(f"Training model for target: {target_idx}")
    model = models[target_idx]
    model.fit(features_normalized, targets_normalized[:, target_idx])

    # Evaluate model performance on training data
    predictions = model.predict(features_normalized)
    r2 = r2_score(targets_normalized[:, target_idx], predictions)
    print(f"R² score for target {target_idx}: {r2:.4f}")


def generate_samples(original_features, n_new_samples=50000, noise_scale=0.02):
    """
    Generate new samples by linear interpolation and noise addition
    Args:
        original_features: Original feature matrix
        n_new_samples: Number of new samples to generate
        noise_scale: Scale of Gaussian noise to add
    Returns:
        DataFrame containing generated samples
    """
    n_original = len(original_features)
    new_samples = []

    for _ in range(n_new_samples):
        # Randomly select two samples for interpolation
        idx1, idx2 = np.random.choice(n_original, 2)
        alpha = np.random.uniform(0.2, 0.8)  # Interpolation coefficient

        # Linear interpolation between two samples
        mixed = original_features[idx1] * alpha + original_features[idx2] * (1 - alpha)

        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale * np.abs(mixed).mean())
        new_sample = mixed + noise
        new_samples.append(new_sample)

    # Convert to DataFrame and clip values to original range
    new_df = pd.DataFrame(new_samples)
    for col in new_df.columns:
        col_min = original_features[:, col].min()
        col_max = original_features[:, col].max()
        new_df[col] = new_df[col].clip(col_min, col_max)

    return new_df


# Generate augmented feature samples
new_features = generate_samples(features, n_new_samples=50000)
print(f"Generated features - Rows: {new_features.shape[0]}, Columns: {new_features.shape[1]}")

# Normalize new features using previously fitted scaler
new_features_normalized = scaler_X.transform(new_features)

# Predict target values for new features using trained models
new_targets_normalized = np.zeros((new_features.shape[0], targets.shape[1]))

for target_idx in range(targets.shape[1]):
    print(f"Predicting target: {target_idx}")
    model = models[target_idx]
    new_targets_normalized[:, target_idx] = model.predict(new_features_normalized)

# Denormalize predicted targets to original scale
new_targets = scaler_Y.inverse_transform(new_targets_normalized)
print(f"Predicted targets - Rows: {new_targets.shape[0]}, Columns: {new_targets.shape[1]}")

# Combine generated features and predicted targets
combined_data = np.hstack([new_features, new_targets])
combined_df = pd.DataFrame(combined_data)

# Save augmented dataset to Excel
output_path = "augmented_data1.xlsx"
combined_df.to_excel(output_path, index=False)

print(f"Data augmentation completed! Total samples: {combined_data.shape[0]}")