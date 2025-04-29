import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    print("Initial Data Overview:")
    print(df.info())
    print(df.head())

    selected_cols = ['y', 'job', 'marital', 'default', 'housing', 'poutcome']
    df_subset = df[selected_cols]
    print("Subset of Data:")
    print(df_subset.head())

    df_encoded = pd.get_dummies(df_subset, columns=selected_cols[1:])
    df_encoded['y'] = df_encoded['y'].map({'yes': 1, 'no': 0})

    return df_encoded

def plot_correlation_heatmap(data):
    plt.figure(figsize=(16, 12))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.show()

def evaluate_model(model, X_test, y_test, title, cmap):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"Confusion Matrix ({title}):\n", cm)
    print(f"Accuracy ({title}): {acc:.2f}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main():
    data = load_and_preprocess_data('bank.csv')
    plot_correlation_heatmap(data)

    X = data.drop('y', axis=1)
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    evaluate_model(log_model, X_test, y_test, 'Logistic Regression', 'Blues')

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    evaluate_model(knn_model, X_test, y_test, 'KNN (k=3)', 'Greens')

if __name__ == "__main__":
    main()
