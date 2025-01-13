import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize et

# Modelleri tanımla ve eğit
def build_model(seed):
    tf.keras.utils.set_random_seed(seed)
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


seeds = [42, 99, 123, 7, 2023]  # Farklı ilk değerler
models = []

for seed in seeds:
    model = build_model(seed)
    model.fit(x_train, y_train, epochs=1, validation_split=0.2, verbose=0)
    models.append(model)

# Test başarımlarını ölç
def evaluate_models(models, x_test, y_test):
    return [model.evaluate(x_test, y_test, verbose=0)[1] for model in models]

test_accuracies = evaluate_models(models, x_test, y_test)
print("Test başarımları:", test_accuracies)

# İki model arasında interpolasyon yapma fonksiyonu
def interpolate_between_models(model1, model2, alphas):
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    interpolated_accuracies = []

    for alpha in alphas:
        interpolated_weights = [(1 - alpha) * w1 + alpha * w2 for w1, w2 in zip(weights1, weights2)]
        model1.set_weights(interpolated_weights)
        acc = model1.evaluate(x_test, y_test, verbose=0)[1]
        interpolated_accuracies.append(acc)

    return interpolated_accuracies

# Tüm model çiftleri için interpolasyon yap ve görselleştir
def visualize_all_model_interpolations(models, test_accuracies, alphas):
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                interpolated_accuracies = interpolate_between_models(models[i], models[j], alphas)

                plt.figure(figsize=(10, 6))
                plt.plot(alphas, interpolated_accuracies, label=f"Interpolated Accuracy: Model {i+1} -> Model {j+1}", color="blue")
                plt.axhline(test_accuracies[i], linestyle="--", color="green", label=f"Model {i+1} Accuracy")
                plt.axhline(test_accuracies[j], linestyle="--", color="red", label=f"Model {j+1} Accuracy")

                plt.xlabel("Alpha (Interpolation)")
                plt.ylabel("Test Accuracy")
                plt.title(f"Interpolation Between Model {i+1} and Model {j+1}")
                plt.legend()
                plt.grid(True)
                plt.show()

# Tüm model kombinasyonları için görselleştirme
alphas = np.linspace(0, 1, 100)
visualize_all_model_interpolations(models, test_accuracies, alphas)
