import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from dataset_logistic_regression import load_data

def get_data(h5_file_path):
    """
    Load the data from the HDF5 file, remove singleton classes, and perform a stratified train-test split.

    Parameters:
    - h5_file_path: str
        Path to the HDF5 file containing the preprocessed data.

    Returns:
    - X_train: numpy.ndarray
        Training data, shape (num_train_samples, num_bins).
    - X_test: numpy.ndarray
        Test data, shape (num_test_samples, num_bins).
    - y_train: numpy.ndarray
        Training labels, shape (num_train_samples,).
    - y_test: numpy.ndarray
        Test labels, shape (num_test_samples,).
    - class_weights: numpy.ndarray
        Class weights for balancing, shape (num_classes,).
    """
    # Load the intensity data and labels
    intensity, labels = load_data(h5_file_path)
    labels = np.array(labels)
    label_counts = Counter(labels)
    print("Original Class Distribution:", label_counts)

    # Identify singleton classes (classes with only one instance)
    singleton_classes = [cls for cls, count in label_counts.items() if count == 1]
    if singleton_classes:
        print(f"Singleton Classes (only 1 instance): {singleton_classes}")
    else:
        print("No singleton classes detected.")

    # Remove singleton classes from the dataset
    if singleton_classes:
        # Indices of samples NOT in singleton classes
        non_singleton_indices = np.where(~np.isin(labels, singleton_classes))[0]
        filtered_labels = labels[non_singleton_indices]
        filtered_intensity = intensity[non_singleton_indices]
    else:
        # Use all data if no singleton classes
        filtered_intensity = intensity
        filtered_labels = labels

    # Stratified train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            filtered_intensity,
            filtered_labels,
            test_size=0.2,
            stratify=filtered_labels,
            random_state=42
        )
    except ValueError as e:
        print("Error during train_test_split:", e)
        print("Ensure that all classes have at least two instances after removing singletons.")
        raise e

    # Compute class weights based on the training data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = np.array(class_weights, dtype=np.float32)

    # Display new class distributions
    print("New Training Class Distribution:", Counter(y_train))
    print("New Validation Class Distribution:", Counter(y_test))

    return X_train, X_test, y_train, y_test, class_weights
