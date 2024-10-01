import numpy as np
from data_loader_logistic_regression import get_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump  # Import dump to save models

def train_model(
    h5_file_path,
    model_save_path='logistic_model.joblib',
    scaler_save_path='scaler.joblib',
    max_iter=1000,
    solver='saga',
    penalty='l2',
    C=1.0
):
    """
    Train and save a Logistic Regression model using scikit-learn.

    Parameters:
    - h5_file_path: str
        Path to the HDF5 file containing the preprocessed data.
    - model_save_path: str
        Path to save the trained Logistic Regression model.
    - scaler_save_path: str
        Path to save the StandardScaler.
    - max_iter: int
        Maximum number of iterations for the solver.
    - solver: str
        Algorithm to use in the optimization problem.
    - penalty: str
        Norm used in the penalization ('l1', 'l2', or 'elasticnet').
    - C: float
        Inverse of regularization strength; must be a positive float.
    """
    # Get the data
    X_train, X_test, y_train, y_test, class_weights = get_data(h5_file_path)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Compute class weights dictionary
    classes = np.unique(y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

    # Create the Logistic Regression model
    model = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        penalty=penalty,
        C=C,
        class_weight=class_weight_dict,
        tol=1e-3,  # Increased tolerance from implicit 1e-4, means the difference between stopping at a change of 0.001 versus 0.0001
        n_jobs=-1,
        verbose=1
    )

    # Train the model
    print("Training the Logistic Regression model...")
    model.fit(X_train, y_train)

    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f'Training Accuracy: {train_acc:.4f}')

    # Evaluate on test data
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_acc:.4f}')

    # Print classification report
    print('Classification Report:')
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # Save the model and scaler
    dump(model, model_save_path)
    dump(scaler, scaler_save_path)
    print(f'Model saved to {model_save_path}')
    print(f'Scaler saved to {scaler_save_path}')

if __name__ == '__main__':
    # Entry point of the script

    h5_file_path = 'preprocessed_data.h5'  # Path to your preprocessed data HDF5 file

    # Call the training function with specified parameters
    train_model(
        h5_file_path,
        model_save_path='logistic_model.joblib',
        scaler_save_path='scaler.joblib',
        max_iter=1000,
        solver='saga',  # 'saga' supports l1 penalty and is suitable for large datasets
        penalty='l2',
        C=1.0
    )
