import pickle
import json
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load evaluation data
def load_evaluation_data(data_path):
    with open(data_path, 'r') as file:
        data_list = json.load(file)  # data is a list of dicts
    
    X = []
    y = []
    for item in data_list:
        X.append(item['question'])
        y.append(item['answer'])
        
    return X, y

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

if __name__ == "__main__":
    # Paths
    model_path = "trained_model.pkl"
    evaluation_data_path = "db/questions/cot_questions_clean.json"

    # Load model and data
    model = load_model(model_path)
    X_test, y_test = load_evaluation_data(evaluation_data_path)

    # Evaluate
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Print results
    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)