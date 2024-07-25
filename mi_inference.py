import pickle
import numpy as np 
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

dico = { 0 : "left_hand", 1 : "right_hand"}

# https://github.com/NeuroTechX/moabb/blob/a840f7d16b155b46d4f2d6482b9c6edc12aceafa/examples/load_model.py#L46

# Example: Load a model from pickle for real-time inference
def load_model(model_name):
    with open(f"./models/Models_CrossSubject/BNCI2014-004/2/AM+SVM/fitted_model_1.pkl", "rb") as f: #open(f"{model_name}_model.pkl", "rb") as f:
        return pickle.load(f)


# Example of real-time inference (assuming X_new is the new 3D data)
# Note: For real-time inference, data needs to be preprocessed similarly to training data
# X_new should be in the shape [n_samples, n_channels, n_times]

# Load a model for real-time inference



loaded_model = load_model("AM+SVM")



grid_search = loaded_model.named_steps.get('gridsearchcv')
# Access the best model from GridSearchCV
best_model = loaded_model.named_steps['gridsearchcv'].best_estimator_


def inference(X = np.random.rand(1, 3, 1000)):
    # Use the loaded pipeline to transform and predict
    X_new_processed = loaded_model.named_steps['logvariance'].transform(X)  # Apply LogVariance transformation
    return dico[best_model.predict(X_new_processed)[0]]
