import h5py
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import argparse
import collections

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test Logistic Regression Model')
    parser.add_argument('--h5_file', type=str, default='preprocessed_data.h5', help='Path to the HDF5 file')
    parser.add_argument('--model_path', type=str, default='logistic_model.joblib', help='Path to the trained model file')
    parser.add_argument('--scaler_path', type=str, default='scaler.joblib', help='Path to the scaler file')
    args = parser.parse_args()

    h5_file_path = args.h5_file
    model_path = args.model_path
    scaler_path = args.scaler_path

    print("Loading preprocessed data...")
    # Load the preprocessed data
    with h5py.File(h5_file_path, 'r') as h5f:
        intensity_data = h5f['intensity'][:]  # Shape: (num_samples, num_bins)
        labels = h5f['labels'][:]
        species_labels = h5f['species_labels'][:]
        # Load m/z bin centers if available
        if 'mz_bins' in h5f:
            mz_bins = h5f['mz_bins'][:]  # Shape: (num_bins,)
        else:
            mz_bins = None

    # Decode species labels if they are stored as bytes
    if isinstance(species_labels[0], bytes):
        species_labels = [label.decode('utf-8') for label in species_labels]

    # Map label indices to species names
    label_to_species = {i: species_labels[i] for i in range(len(species_labels))}
    print(f"Species labels mapping loaded.")

    # Check the distribution of labels in the dataset
    unique_labels = set(labels)
    print(f"Unique labels in the dataset: {unique_labels}")

    label_counts = collections.Counter(labels)
    print("Label counts in the dataset:")
    for label, count in label_counts.items():
        species_name = label_to_species.get(label, "Unknown")
        print(f"Label {label} ({species_name}): {count}")

    # Display available spectra indices
    num_spectra = len(labels)
    print(f"Total number of spectra: {num_spectra}")
    print(f"Available indices: 0 to {num_spectra - 1}")

    # Prompt the user to select a spectrum index
    spectrum_index = int(input(f"Enter the index of the spectrum to analyze (0 to {num_spectra - 1}): "))
    if spectrum_index < 0 or spectrum_index >= num_spectra:
        print("Invalid index. Exiting.")
        return

    print("Loading the selected spectrum...")
    # Get the selected spectrum and label
    intensity = intensity_data[spectrum_index]
    true_label = labels[spectrum_index]
    true_species = label_to_species[true_label]

    print(f"True label for index {spectrum_index}: {true_label}")
    print(f"True species for index {spectrum_index}: {true_species}")

    print("Displaying the spectrum plot...")
    plt.figure(figsize=(10, 6))
    if mz_bins is not None:
        plt.plot(mz_bins, intensity)
        plt.xlabel('m/z')
    else:
        plt.plot(intensity)
        plt.xlabel('Bin Index')
    plt.ylabel('Intensity')
    plt.title(f'Spectrum Index: {spectrum_index}, True Species: {true_species}')
    plt.show(block=True)
    # plt.show
    # plt.show(block=False)
    # plt.pause(1)  # Pause to ensure the plot displays
    print("Spectrum plot displayed. Proceeding to preprocess the spectrum...")

    print("Starting spectrum preprocessing...")
    # Preprocess the spectrum
    # Reshape the spectrum to match the expected input shape
    intensity_sample = intensity.reshape(1, -1)  # Shape: (1, num_bins)

    # Load the scaler and standardize the features
    scaler = load(scaler_path)
    intensity_sample_scaled = scaler.transform(intensity_sample)

    print(f"Intensity sample shape after scaling: {intensity_sample_scaled.shape}")

    print("Loading the trained model...")
    model = load(model_path)
    print("Model loaded successfully.")

    print("Making the prediction...")
    # Make the prediction
    predicted_label = model.predict(intensity_sample_scaled)[0]
    predicted_species = label_to_species.get(predicted_label, "Unknown")

    # Display the predicted pathogen
    print(f"Predicted Species Label: {predicted_label}")
    print(f"Predicted Species: {predicted_species}")
    print(f"True Species Label: {true_label}")
    print(f"True Species: {true_species}")

    if predicted_species == true_species:
        print("Prediction is correct!")
    else:
        print("Prediction is incorrect.")

if __name__ == '__main__':
    main()
