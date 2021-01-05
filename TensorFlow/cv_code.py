# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_Project 2: MLP Implementation_-_-_-_-_-_-_-_-_-_-_-__-_-_-_-_-_-_-
                                                                                            # Alonso Vega
                                                                                         # Alazar Alwajih
                                                                                                # 4/19/20

# Objective: Gain experience with Python and TensorFlow/Keras API. We will implement
#                 a multi-layer NN training model using the popular MNIST clothing data set.
# _-_-_-_-_-_-_-_-_-_-_-__-_-_-_-_-_-_-_-_-_-_-__-_-_-_-_-_-_-_-_-_-_-__-_-_-_-_-_-_-_-_-_-_-__-_-_-_-_-_-

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_data, y_train_data), (X_test, y_test) = fashion_mnist.load_data()

print("Feature matrix dimension: ", X_train_data.shape, "\n")
print("Feature matrix data type: ", X_train_data.dtype, "\n")

n_img = X_train_data.shape[2]
n_neu_hid1 = 300
n_neu_hid2 = 100

n_split = 5000
X_valid, X_train = X_train_data[:n_split] / 255.0, X_train_data[n_split:] / 255.0
y_valid, y_train = y_train_data[:n_split], y_train_data[n_split:]

class_id = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Classification MLP: 2 Hidden Layers
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[n_img, n_img]))              # vectorizing
model.add(keras.layers.Dense(n_neu_hid1, activation="relu"))             # n_neu_hid1 neurons/ ReLU act. function
model.add(keras.layers.Dense(n_neu_hid2, activation="relu"))             # n_neu_hid2 neurons/ ReLU act. function
model.add(keras.layers.Dense(n_neu_hid2, activation="relu"))             # n_neu_hid3 neurons/ ReLU act. function
model.add(keras.layers.Dense(len(class_id), activation="softmax"))       # output layer

print(model.summary(), "\n")

# Train
n_epoch = 10
n_batch = 50

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=0.05), metrics=["accuracy"])    # stochastic gradient descent

print("\n---------------------------------------------Training---------------------------------------------")
history = model.fit(X_train, y_train, epochs=n_epoch,
                    validation_data=(X_valid, y_valid), batch_size=n_batch)     # training

# Plot
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("Learning Curves")
plt.xlabel('epoch')
plt.show()


# Test
print("\n---------------------------------------------Testing----------------------------------------------")
model.evaluate(X_test, y_test)

# 10 Random samples from test set
n_rnd = np.random.randint(0, X_test.shape[0], 10)
X_hat = X_test[n_rnd, :, :]

n_hat = X_hat.shape[0]

# Predictions in [0,1]
y_hat = model.predict(X_hat)

print("\n", "Probabilistic Prediction Vector: ", "\n", y_hat.round(3))

# Predictions in Class Set
y_hat_class = model.predict_classes(X_hat)
class_hat = np.array(class_id)[y_hat_class]

# Plot
n_row = 2
n_col = 5
fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(2.5*n_col, 4*n_row))
for j in range(0, n_hat):
    axis = axes[j//n_col, j%n_col]
    axis.imshow(X_hat[j], cmap='gray')
    axis.set_title('Label: {}'.format(class_hat[j]))
plt.tight_layout()
plt.show()

# Weights
hidden = model.layers

print("-----------------------------------------Parameters of 2nd layer:--------------------------------------- \n")
print(hidden[1].get_weights())





