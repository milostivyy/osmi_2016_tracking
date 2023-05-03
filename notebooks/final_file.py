import pickle

# Load the pickled model
with open('/workspaces/codespaces-jupyter/notebooks/svm_pickle_model.pkl', 'rb') as f:
    svm_dict = pickle.load(f)

# Extract the SVM model and feature names from the dictionary
svm_model = svm_dict["model"]
feature_names = svm_dict["feature_names"]

# Create a new data point with 39 features
new_data_point = [[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]]

# Make predictions on the new data point
predictions = svm_model.predict(new_data_point)


# Print out the feature names and predictions
print("Feature Names:")
for feature in feature_names:
    print(feature)
print("\nPredictions:")
print(predictions)
