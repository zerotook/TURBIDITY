import numpy as np
import pandas as pd

# Set the file path to your data file
file_path = 'yourfilepath'

# Load data
data = pd.read_csv(file_path)

# Define noise parameters
mu, sigma = 0, 0.1

# Define the number of augmented samples you want to generate
num_augmented_samples = 200

# Create empty array to store augmented samples
augmented_data = np.empty((0, 2))

# Loop over each row in the original data
for i in range(data.shape[0]):

    # Get the row data
    row_data = data.iloc[i]

    # Repeat the row data to create a matrix
    matrix = np.tile(row_data, (num_augmented_samples, 1))

    # Generate Gaussian noise with the same shape as the matrix
    noise = np.random.normal(mu, sigma, size=matrix.shape)

    # Add the noise to the matrix
    augmented_matrix = matrix + noise

    # Append the augmented matrix to the augmented data array
    augmented_data = np.vstack((augmented_data, augmented_matrix))

    if i == 3:
        break

# Create a new DataFrame with the augmented data
columns = ['Sentinel2 Turbidity', 'in situ turbidity']
augmented_df = pd.DataFrame(data=augmented_data, columns=columns)

# Combine the original data and augmented data into a single DataFrame
combined_df = pd.concat([data, augmented_df], ignore_index=True)

combined_df.to_csv('output.csv', index=False)
