import h5py
import numpy as np

def load_data(h5_file_path):
    """
    Load the MALDI-TOF mass spectra data from an HDF5 file.

    Parameters:
    - h5_file_path: str
        Path to the HDF5 file containing the preprocessed data.

    Returns:
    - intensity: numpy.ndarray
        2D array of intensity values, shape (num_samples, num_bins).
    - labels: numpy.ndarray
        1D array of label indices, shape (num_samples,).
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        intensity = h5_file['intensity'][:]
        labels = h5_file['labels'][:]
    return intensity, labels
