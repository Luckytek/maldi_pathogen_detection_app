import streamlit as st
import h5py
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import collections
import io
import os
import gdown  # Install using `pip install gdown`

def download_data():
    h5_file_path = 'preprocessed_data.h5'
    if not os.path.exists(h5_file_path):
        url = 'https://drive.google.com/uc?id=1c4l6_rZXRhmiwwnO2sqwys2HLU6tJTT8'
        print("Downloading data file...")
        gdown.download(url, h5_file_path, quiet=False)
        print("Data file downloaded successfully.")
    else:
        print("Data file already exists locally.")

# Call the download function at the beginning of your script
download_data()

def main():
    st.title("Pathogen Detection from MALDI Spectra")

    # Load data and model
    @st.cache(allow_output_mutation=True)
    def load_data():
        h5_file_path = 'preprocessed_data.h5'
        with h5py.File(h5_file_path, 'r') as h5f:
            intensity_data = h5f['intensity'][:]
            labels = h5f['labels'][:]
            species_labels = h5f['species_labels'][:]
            if 'mz_bins' in h5f:
                mz_bins = h5f['mz_bins'][:]
            else:
                mz_bins = None
        # Decode species labels if needed
        if isinstance(species_labels[0], bytes):
            species_labels = [label.decode('utf-8') for label in species_labels]
        label_to_species = {i: species_labels[i] for i in range(len(species_labels))}
        return intensity_data, labels, mz_bins, label_to_species

    @st.cache(allow_output_mutation=True)
    def load_model():
        model = load('logistic_model.joblib')
        scaler = load('scaler.joblib')
        return model, scaler

    intensity_data, labels, mz_bins, label_to_species = load_data()
    model, scaler = load_model()

    num_spectra = len(labels)
    spectrum_index = st.number_input(
        f"Select Spectrum Index (0 to {num_spectra -1})",
        min_value=0,
        max_value=num_spectra -1,
        value=0,
        step=1
    )
    intensity = intensity_data[spectrum_index]
    true_label = labels[spectrum_index]
    true_species = label_to_species[true_label]

    st.write(f"**True Species:** {true_species}")

    # Plot the spectrum
    fig, ax = plt.subplots()
    if mz_bins is not None:
        ax.plot(mz_bins, intensity)
        ax.set_xlabel('m/z')
    else:
        ax.plot(intensity)
        ax.set_xlabel('Bin Index')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Spectrum Index: {spectrum_index}, True Species: {true_species}')
    st.pyplot(fig)

    # Preprocess and predict
    intensity_sample = intensity.reshape(1, -1)
    intensity_sample_scaled = scaler.transform(intensity_sample)
    predicted_label = model.predict(intensity_sample_scaled)[0]
    predicted_species = label_to_species.get(predicted_label, "Unknown")
    st.write(f"**Predicted Species:** {predicted_species}")

    if predicted_species == true_species:
        st.success("Prediction is correct!")
    else:
        st.error("Prediction is incorrect.")

if __name__ == '__main__':
    main()
