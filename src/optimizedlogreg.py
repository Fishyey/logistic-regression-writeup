from os import path
import numpy as np
# import kagglehub


# path = kagglehub.dataset_download("nareshbhat/wine-quality-binary-classification")
script_dir = path.dirname(path.abspath(__file__))
file_path = path.join(script_dir, "..", "data", "uncleaned_wine_data")

df = np.genfromtxt(file_path, delimiter=",", dtype=None)

# np.savetxt("data", df, delimiter=",", fmt="%s")
features = df[0, :]
df = df[1:, :]

rng = np.random.default_rng(seed=42)
rng.shuffle(df)

training_set = df[:1200, :]
testing_set = df[1200:, :]

training_x = training_set[:, :-1].astype(float)
training_y = training_set[:, -1]
for i in range(len(training_y)):
    if training_y[i] == "good":
        training_y[i] = 1
    else:
        training_y[i] = 0
training_y = training_y.astype(float)

testing_x = testing_set[:, :-1].astype(float)
testing_y = testing_set[:, -1]
for i in range(len(testing_y)):
    if testing_y[i] == "good":
        testing_y[i] = 1
    else:
        testing_y[i] = 0
testing_y = testing_y.astype(float)

m, n = training_x.shape
wb = np.zeros(n + 1) # Includes bias, which will be the 0th term

### NORMALIZING
min_values = training_x.min(axis=0)
max_values = training_x.max(axis=0)
scale_values = 1/(max_values - min_values)

training_x = (training_x - min_values) * scale_values
testing_x = (testing_x - min_values) * scale_values

print(training_x[:5])
print(testing_x[:5])

### OPERATIONS
def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def prediction(w: np.ndarray, x: np.ndarray) -> float:
    return sigmoid(np.dot(w, x))

def compute_loss(w: np.ndarray, x: np.ndarray, y: float) -> float:
    return -(y * np.log(prediction(w, x)) + (1 - y) * np.log(1 - prediction(w, x)))

def compute_gradient(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    gradient_term = np.empty((m, 1))
    for i in range(m):
        a = prediction(w, x[i])
        gradient_term[i] = a - y[i].item() # (y_hat - y)
    return 1/m * np.dot(x.T, gradient_term)

training_x_bias = np.hstack((np.ones((m, 1)), training_x)) # Necessary so that our bias (0th in wb) has a term to multiply with, which is 1

num_correct_before = 0
for i in range(m):
   if training_y[i] == round(prediction(wb, training_x_bias[i])):
        num_correct_before += 1
print("Initial Accuracy: ", num_correct_before / m)

### TRAINING
epochs = 10000
learning_rate = 0.03

for i in range(epochs):
    gradient = compute_gradient(wb, training_x_bias, training_y)
    delta = learning_rate * gradient
    wb = (wb - delta.T).squeeze() # Ensures wb remains 1d
    if i%1000 == 0:
        loss = 0.0
        for j in range(len(training_x)):
            loss += compute_loss(wb, training_x_bias[j], training_y[j])
        print(f"Cost function for {i} epochs: {loss/len(training_x)}")
        raw_preds = [prediction(wb, training_x_bias[j]) for j in range(m)]
        print(f"Epoch {i} - Min Pred: {min(raw_preds):.3f}, Max Pred: {max(raw_preds):.3f}, Avg Pred: {np.mean(raw_preds):.3f}")


### TESTING
MSE = 0
num_correct = 0
testing_x_bias = np.hstack((np.ones((len(testing_x), 1)), testing_x))
for i in range(len(testing_x)):
    MSE += (testing_y[i] - prediction(wb, testing_x_bias[i])) ** 2
    if testing_y[i] == round(prediction(wb, testing_x_bias[i])):
        num_correct += 1

print("MSE: ", MSE / len(testing_x))
print("Accuracy: ", num_correct / len(testing_x))
print("Weights: ", wb[1:], "Bias: ", wb[0])

#for i in range(m):

    


