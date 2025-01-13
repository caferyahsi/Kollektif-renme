# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:25:23 2025

@author: cafer
"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# CIFAR-10 veri kümesini yükle ve normalize et
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Modelleri tanımla ve eğit
def build_model(seed):
    tf.keras.utils.set_random_seed(seed)
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

seeds = [42, 99, 123, 7, 2023]
models = []

for seed in seeds:
    model = build_model(seed)
    model.fit(x_train, y_train, epochs=1, validation_split=0.2, verbose=0)
    models.append(model)

# Test başarımlarını ölç
def evaluate_models(models, x_test, y_test):
    results = []
    for model in models:
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        results.append((loss, acc))
    return results

test_results = evaluate_models(models, x_test, y_test)
test_losses = [res[0] for res in test_results]
test_accuracies = [res[1] for res in test_results]

# İki model arasında interpolasyon yapma fonksiyonu
def interpolate_between_models(model1, model2, alphas):
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    interpolated_accuracies = []
    interpolated_losses = []

    for alpha in alphas:
        interpolated_weights = [(1 - alpha) * w1 + alpha * w2 for w1, w2 in zip(weights1, weights2)]
        model1.set_weights(interpolated_weights)
        loss, acc = model1.evaluate(x_test, y_test, verbose=0)
        interpolated_losses.append(loss)
        interpolated_accuracies.append(acc)

    return interpolated_losses, interpolated_accuracies

# Geodesic Interpolation fonksiyonu
def geodesic_interpolation(model1, model2, alphas):
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    interpolated_accuracies = []
    interpolated_losses = []

    for alpha in alphas:
        # Geodesic interpolation: sqrt(1-alpha) * weights1 + sqrt(alpha) * weights2
        interpolated_weights = [
            np.sqrt(1 - alpha) * w1 + np.sqrt(alpha) * w2 for w1, w2 in zip(weights1, weights2)
        ]
        model1.set_weights(interpolated_weights)
        loss, acc = model1.evaluate(x_test, y_test, verbose=0)
        interpolated_losses.append(loss)
        interpolated_accuracies.append(acc)

    return interpolated_losses, interpolated_accuracies

# Tüm model çiftleri için interpolasyon ve görselleştirme
def visualize_all_model_interpolations(models, test_losses, test_accuracies, alphas):
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                interpolated_losses, interpolated_accuracies = interpolate_between_models(models[i], models[j], alphas)

                # Accuracy grafiği
                plt.figure(figsize=(10, 6))
                plt.plot(alphas, interpolated_accuracies, label=f"Interpolated Accuracy: Model {i+1} -> Model {j+1}", color="blue")
                plt.axhline(test_accuracies[i], linestyle="--", color="green", label=f"Model {i+1} Accuracy")
                plt.axhline(test_accuracies[j], linestyle="--", color="red", label=f"Model {j+1} Accuracy")
                plt.xlabel("Alpha (Interpolation)")
                plt.ylabel("Test Accuracy")
                plt.title(f"Interpolation Between Model {i+1} and Model {j+1} (Accuracy)")
                plt.legend()
                plt.grid(True)
                plt.show()

                # Loss grafiği
                plt.figure(figsize=(10, 6))
                plt.plot(alphas, interpolated_losses, label=f"Interpolated Loss: Model {i+1} -> Model {j+1}", color="orange")
                plt.axhline(test_losses[i], linestyle="--", color="green", label=f"Model {i+1} Loss")
                plt.axhline(test_losses[j], linestyle="--", color="red", label=f"Model {j+1} Loss")
                plt.xlabel("Alpha (Interpolation)")
                plt.ylabel("Test Loss")
                plt.title(f"Interpolation Between Model {i+1} and Model {j+1} (Loss)")
                plt.legend()
                plt.grid(True)
                plt.show()

# Geodesic Interpolation görselleştirme fonksiyonu
def visualize_geodesic_interpolations(models, test_losses, test_accuracies, alphas):
    for i in range(len(models)):
        if i==4:
            break
        for j in range(len(models)):
            if i != j:
                interpolated_losses, interpolated_accuracies = geodesic_interpolation(models[i], models[j], alphas)

                # Geodesic Accuracy grafiği
                plt.figure(figsize=(10, 6))
                plt.plot(alphas, interpolated_accuracies, label=f"Geodesic Accuracy: Model {i+1} -> Model {j+1}", color="purple")
                plt.axhline(test_accuracies[i], linestyle="--", color="green", label=f"Model {i+1} Accuracy")
                plt.axhline(test_accuracies[j], linestyle="--", color="red", label=f"Model {j+1} Accuracy")
                plt.xlabel("Alpha (Geodesic Interpolation)")
                plt.ylabel("Test Accuracy")
                plt.title(f"Geodesic Interpolation Between Model {i+1} and Model {j+1} (Accuracy)")
                plt.legend()
                plt.grid(True)
                plt.show()

               

# Ağırlık uzayında interpolasyon (PCA ile)
def visualize_weight_space(models, alphas):
    weights = [model.get_weights() for model in models]
    flat_weights = [np.concatenate([w.flatten() for w in model_weights]) for model_weights in weights]

    # Interpolasyon ağırlıklarını hesapla
    interpolated_weights = []
    for i in range(len(flat_weights)):
        for j in range(len(flat_weights)):
            if i < j:
                for alpha in alphas:
                    interpolated_weights.append((1 - alpha) * flat_weights[i] + alpha * flat_weights[j])

    # PCA ile indirgeme
    pca = PCA(n_components=2)
    reduced_weights = pca.fit_transform(interpolated_weights)

    # Görselleştirme
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], c=np.tile(alphas, len(models) * (len(models) - 1) // 2), cmap="viridis", label="Interpolated Points")
    plt.colorbar(label="Alpha (Interpolation)")
    plt.title("Weight Space Interpolation (PCA)")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.grid(True)
    plt.show()

# Isı haritası ile ağırlık değişimi
def visualize_weight_heatmap(models, alphas):
    model1_weights = models[0].get_weights()
    model2_weights = models[1].get_weights()

    # Sadece ilk katman ağırlıkları (örnek)
    w1 = model1_weights[0]
    w2 = model2_weights[0]

    interpolated_weights = [(1 - alpha) * w1 + alpha * w2 for alpha in alphas]
    
    # Isı haritası
    plt.figure(figsize=(10, 6))
    sns.heatmap(np.array(interpolated_weights)[:, :, 0], cmap="coolwarm", cbar=True)
    plt.xlabel("Feature Index")
    plt.ylabel("Interpolation Step")
    plt.title("Weight Changes During Interpolation (Heatmap)")
    plt.show()

# Görselleştirme çağrıları
alphas = np.linspace(0, 1, 50)

visualize_geodesic_interpolations(models, test_losses, test_accuracies, alphas)

