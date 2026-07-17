import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
df = pd.read_csv(
	'iris.CSV',
	header=None,
	names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
)

X = df.iloc[:, :4]
y = df.iloc[:, 4]

feature_names = df.columns[:4]
class_names = y.unique()

print(X.shape)
print(y.shape)
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nClass Distribution:")
print(df['class'].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())
# Histograms of Features
# df.hist(figsize=(10,8), bins=20)
# plt.suptitle("Feature Distributions")
# plt.show()
# # Pairplot of Features
# sns.pairplot(df, hue='class', markers=["o", "s", "D"])
# plt.show()
# # Correlation Heatmap
# plt.figure(figsize=(8,6))
# sns.heatmap(df.iloc[:,:4].corr(),annot=True,cmap='coolwarm')
# plt.title("Feature Correlation Heatmap")
# plt.show()
# Encode Labels
class_to_int = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
}

y_encoded = y.map(class_to_int)
print(y_encoded.head())

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

y_train_onehot = one_hot_encode(y_train.values, 3)
y_val_onehot = one_hot_encode(y_val.values, 3)
y_test_onehot = one_hot_encode(y_test.values, 3)

print(y_train_onehot[:5])

def initialize_parameters():
    np.random.seed(42)

    parameters = {
        "W1": np.random.randn(4, 8) * np.sqrt(1 / 4),
        "b1": np.zeros((1, 8)),

        "W2": np.random.randn(8, 6) * np.sqrt(1 / 8),
        "b2": np.zeros((1, 6)),

        "W3": np.random.randn(6, 3) * np.sqrt(1 / 6),
        "b3": np.zeros((1, 3))
    }

    return parameters

parameters = initialize_parameters()

for key, value in parameters.items():
    print(f"{key}: {value.shape}")
    
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    return A * (1 - A)

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)   # Numerical stability
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

sample = np.array([[2, 3, 1]])

print(sigmoid(sample))
print(softmax(sample))

# forwarded propagation

def forward_propagation(X, parameters):
    

    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    # Hidden Layer 1
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    # Hidden Layer 2
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Output Layer
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

    # Store values for backpropagation
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3
    }

    return A3, cache

parameters = initialize_parameters()

A3, cache = forward_propagation(X_train, parameters)

print("Output Shape:", A3.shape)
print("First Prediction:")
print(A3[0])

print("Sum of probabilities:", np.sum(A3[0]))

def compute_loss(y_true, y_pred):
   

    m = y_true.shape[0]

    # Prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.sum(y_true * np.log(y_pred)) / m

    return loss
A3, cache = forward_propagation(X_train, parameters)

loss = compute_loss(y_train_onehot, A3)

print("Initial Loss:", loss)


def backward_propagation(X, y_true, parameters, cache):

    m = X.shape[0]

    W2 = parameters["W2"]
    W3 = parameters["W3"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    # ---------- Output Layer ----------
    dZ3 = A3 - y_true
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    # ---------- Hidden Layer 2 ----------
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * sigmoid_derivative(A2)

    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # ---------- Hidden Layer 1 ----------
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)

    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3
    }

    return gradients

def update_parameters(parameters, gradients, learning_rate):

    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]

    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]

    parameters["W3"] -= learning_rate * gradients["dW3"]
    parameters["b3"] -= learning_rate * gradients["db3"]

    return parameters
parameters = initialize_parameters()

# Forward pass
A3, cache = forward_propagation(X_train, parameters)

# Compute gradients
gradients = backward_propagation(
    X_train,
    y_train_onehot,
    parameters,
    cache
)

# Save old weights
old_W1 = parameters["W1"].copy()

# Update parameters
parameters = update_parameters(parameters, gradients, learning_rate=0.01)

# Check whether weights changed
print(np.array_equal(old_W1, parameters["W1"]))

def create_mini_batches(X, y, batch_size=16):
   
    m = X.shape[0]

    indices = np.random.permutation(m)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        mini_batches.append((X_batch, y_batch))

    return mini_batches

def predict(X, parameters):
    """
    Predict class labels.
    """

    A3, _ = forward_propagation(X, parameters)

    predictions = np.argmax(A3, axis=1)

    return predictions

def compute_accuracy(X, y_true, parameters):
    """
    Compute classification accuracy.

    Parameters:
        X : Input features
        y_true : One-hot labels
        parameters : Network parameters
    """

    predictions = predict(X, parameters)

    true_labels = np.argmax(y_true, axis=1)

    accuracy = np.mean(predictions == true_labels)

    return accuracy

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=1000,
    learning_rate=0.01,
    batch_size=16,
    patience=10
):
    """
    Train the neural network using mini-batch gradient descent.
    """

    parameters = initialize_parameters()

    training_loss = []
    validation_accuracy = []

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):

        mini_batches = create_mini_batches(
            X_train,
            y_train,
            batch_size
        )

        # Train on each mini-batch
        for X_batch, y_batch in mini_batches:

            A3, cache = forward_propagation(
                X_batch,
                parameters
            )

            gradients = backward_propagation(
                X_batch,
                y_batch,
                parameters,
                cache
            )

            parameters = update_parameters(
                parameters,
                gradients,
                learning_rate
            )

        # Every 50 epochs
        if epoch % 50 == 0:

            train_output, _ = forward_propagation(
                X_train,
                parameters
            )

            loss = compute_loss(
                y_train,
                train_output
            )

            val_acc = compute_accuracy(
                X_val,
                y_val,
                parameters
            )

            training_loss.append(loss)
            validation_accuracy.append(val_acc)

            print(
                f"Epoch {epoch:4d} | "
                f"Loss = {loss:.4f} | "
                f"Validation Accuracy = {val_acc:.4f}"
            )

            # ---------- Early Stopping ----------
            if val_acc > best_val_acc:

                best_val_acc = val_acc
                best_parameters = {
                    k: v.copy()
                    for k, v in parameters.items()
                }

                patience_counter = 0

            else:

                patience_counter += 1

                if patience_counter >= patience:

                    print("\nEarly stopping triggered!")

                    return (
                        best_parameters,
                        training_loss,
                        validation_accuracy
                    )

    return (
        parameters,
        training_loss,
        validation_accuracy
    )
    
parameters, loss_history, val_accuracy = train_model(
    X_train,
    y_train_onehot,
    X_val,
    y_val_onehot,
    epochs=1000,
    learning_rate=0.01,
    batch_size=16
)

# Predict on test data
test_predictions = predict(X_test, parameters)

# Convert one-hot labels back to class indices
y_test_labels = np.argmax(y_test_onehot, axis=1)

# Calculate accuracy
test_accuracy = accuracy_score(y_test_labels, test_predictions)

print(f"Test Accuracy: {test_accuracy:.4f}")

cm = confusion_matrix(y_test_labels, test_predictions)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=["Setosa","Versicolor","Virginica"],
    yticklabels=["Setosa","Versicolor","Virginica"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(
    y_test_labels,
    test_predictions,
    target_names=[
        "Setosa",
        "Versicolor",
        "Virginica"
    ]
))

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

def initialize_parameters_pca():
    np.random.seed(42)

    return {
        "W1": np.random.randn(2,8)*np.sqrt(1/2),
        "b1": np.zeros((1,8)),
        "W2": np.random.randn(8,6)*np.sqrt(1/8),
        "b2": np.zeros((1,6)),
        "W3": np.random.randn(6,3)*np.sqrt(1/6),
        "b3": np.zeros((1,3))
    }

parameters_pca = initialize_parameters_pca()
    
h = 0.02

x_min, x_max = X_train_pca[:,0].min()-1, X_train_pca[:,0].max()+1
y_min, y_max = X_train_pca[:,1].min()-1, X_train_pca[:,1].max()+1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

grid = np.c_[xx.ravel(), yy.ravel()]

pred = predict(grid, parameters_pca)
pred = pred.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(xx, yy, pred, alpha=0.3)

plt.scatter(
    X_train_pca[:,0],
    X_train_pca[:,1],
    c=np.argmax(y_train_onehot, axis=1),
    edgecolor='k'
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Decision Boundary (PCA Projection)")
plt.show()

mlp = MLPClassifier(
    hidden_layer_sizes=(8,6),
    activation='logistic',
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, np.argmax(y_train_onehot, axis=1))

sk_predictions = mlp.predict(X_test)

sk_accuracy = accuracy_score(
    y_test_labels,
    sk_predictions
)

print("Scikit-learn Accuracy:", sk_accuracy)

print(classification_report(
    y_test_labels,
    sk_predictions
))

comparison = pd.DataFrame({
    "Model": [
        "NumPy MLP",
        "Sklearn MLP"
    ],
    "Accuracy": [
        test_accuracy,
        sk_accuracy
    ]
})

print(comparison)

learning_rates = [0.001, 0.01, 0.1]

results_lr = []

for lr in learning_rates:

    parameters, loss_history, val_history = train_model(
        X_train,
        y_train_onehot,
        X_val,
        y_val_onehot,
        epochs=1000,
        learning_rate=lr,
        batch_size=16
    )

    predictions = predict(X_test, parameters)

    accuracy = accuracy_score(
        np.argmax(y_test_onehot, axis=1),
        predictions
    )

    results_lr.append([lr, accuracy])

    print(f"Learning Rate: {lr}  Accuracy: {accuracy:.4f}")
    
lr_results = pd.DataFrame(
    results_lr,
    columns=["Learning Rate","Test Accuracy"]
)

print(lr_results)
