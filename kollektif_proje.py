import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Veri setini yükle
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize et

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

seeds = [42, 99, 123, 7, 2023]  # Farklı ilk değerler
models = []
histories = []

for seed in seeds:
    model = build_model(seed)
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=0)
    models.append(model)
    histories.append(history)

# Test başarımlarını ölç
test_accuracies = [model.evaluate(x_test, y_test, verbose=0)[1] for model in models]
print("Test başarımları:", test_accuracies)

# Yol bulma (interpolasyon)
def interpolate_weights(model1, model2, alpha):
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    interpolated_weights = [(1 - alpha) * w1 + alpha * w2 for w1, w2 in zip(weights1, weights2)]
    return interpolated_weights

alphas = np.linspace(1, 100, 100) / 100
interpolated_accuracies = []

for i in range(len(models) - 1):
    model1, model2 = models[i], models[i + 1]
    accs = []
    for alpha in alphas:
        interpolated_weights = interpolate_weights(model1, model2, alpha)
        model1.set_weights(interpolated_weights)
        acc = model1.evaluate(x_test, y_test, verbose=0)[1]
        accs.append(acc)
    interpolated_accuracies.append(accs)

# Ensemble oluşturma
def ensemble_predictions(models, x):
    predictions = [model.predict(x, verbose=0) for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

ensemble_accuracy = np.mean(ensemble_predictions(models, x_test) == y_test)
print("Ensemble doğruluğu:", ensemble_accuracy)

# Görselleştirme
plt.figure(figsize=(10, 6))
for i, accs in enumerate(interpolated_accuracies):
    plt.plot(alphas, accs, label=f"Model {i} -> {i+1}")
plt.xlabel("Alpha (Interpolation)")
plt.ylabel("Accuracy")
plt.title("Interpolation Accuracy Between Models")
plt.legend()
plt.show()


def multi_model_interpolation(models, alphas):
    """
    Çoklu model arasında interpolasyon yapar.
    """
    # İlk modeli temel al
    interpolated_weights = models[0].get_weights()

    for i in range(1, len(models)):
        # Mevcut model ile interpolasyon yap
        weights_current = models[i].get_weights()
        interpolated_weights = [
            (1 - alphas[i - 1]) * w_prev + alphas[i - 1] * w_curr
            for w_prev, w_curr in zip(interpolated_weights, weights_current)
        ]

    return interpolated_weights

# Alphas: Her model için interpolasyon katsayıları
alphas = np.linspace(1, 100, 100) / 100  # Örneğin, 5 model arasında ağırlıklandırma

# Beş model arasında interpolasyon
final_weights = multi_model_interpolation(models, alphas)

# Sonuçları yeni bir modele yükle
models[0].set_weights(final_weights)

# Performansı ölç
final_accuracy = models[0].evaluate(x_test, y_test, verbose=0)[1]
print("Son interpolasyon doğruluğu:", final_accuracy)

# Ardışık interpolasyonun doğruluğunu görselleştir
alphas_all = np.linspace(1, 100, 100) / 100
multi_model_accuracies = []

for alpha in alphas_all:
    weights = multi_model_interpolation(models, [alpha] * 4)
    models[0].set_weights(weights)
    acc = models[0].evaluate(x_test, y_test, verbose=0)[1]
    multi_model_accuracies.append(acc)

# Grafik oluştur
plt.plot(alphas_all, multi_model_accuracies, marker='o')
plt.xlabel("Interpolation Alpha")
plt.ylabel("Test Accuracy")
plt.title("Multi-Model Interpolation Accuracy")
plt.show()




def multi_model_interpolation(models, alphas):
    """
    Çoklu model arasında interpolasyon yapar.
    """
    # İlk modeli temel al
    interpolated_weights = models[0].get_weights()

    for i in range(1, len(models)):
        # Mevcut model ile interpolasyon yap
        weights_current = models[i].get_weights()
        interpolated_weights = [
            (1 - alphas[i - 1]) * w_prev + alphas[i - 1] * w_curr
            for w_prev, w_curr in zip(interpolated_weights, weights_current)
        ]

    return interpolated_weights

# Alphas: Her model için interpolasyon katsayıları
alphas = np.linspace(1, 100, 100) / 100  # Örneğin, 5 model arasında ağırlıklandırma

# Beş model arasında interpolasyon
final_weights = multi_model_interpolation(models, alphas)

# Sonuçları yeni bir modele yükle
models[0].set_weights(final_weights)

# Performans ölçümü ve görselleştirme
alphas = np.linspace(1, 100, 100) / 100
multi_model_train_accuracies = []  # Eğitim doğrulukları
multi_model_test_accuracies = []   # Test doğrulukları

for alpha in alphas_all:
    weights = multi_model_interpolation(models, [alpha] * 4)
    models[0].set_weights(weights)
    
    # Eğitim doğruluğunu ölç
    train_acc = models[0].evaluate(x_train, y_train, verbose=0)[1]
    multi_model_train_accuracies.append(train_acc)
    
    # Test doğruluğunu ölç
    test_acc = models[0].evaluate(x_test, y_test, verbose=0)[1]
    multi_model_test_accuracies.append(test_acc)

# Grafik oluştur
plt.figure(figsize=(10, 6))
plt.plot(alphas_all, multi_model_train_accuracies, marker='o', label="Train Accuracy", color='blue')
plt.plot(alphas_all, multi_model_test_accuracies, marker='x', label="Test Accuracy", color='red')
plt.xlabel("Interpolation Alpha")
plt.ylabel("Accuracy")
plt.title("Train & Test Accuracy During Multi-Model Interpolation")
plt.legend()
plt.grid(True)
plt.show()



def multi_model_extrapolation(models, alphas):
    """
    Çoklu model arasında ekstrapolasyon yapar.
    """
    # İlk modeli temel al
    extrapolated_weights = models[0].get_weights()

    for i in range(1, len(models)):
        # Mevcut model ile ekstrapolasyon yap
        weights_current = models[i].get_weights()
        extrapolated_weights = [
            (1 - alphas[i - 1]) * w_prev + alphas[i - 1] * w_curr
            for w_prev, w_curr in zip(extrapolated_weights, weights_current)
        ]

    return extrapolated_weights

# Alphas: -5 ile 5 arasında 100 eşit adım
alphas = np.linspace(1, 100, 100) / 100

# Performans ölçümü ve görselleştirme
multi_model_train_accuracies = []  # Eğitim doğrulukları
multi_model_test_accuracies = []   # Test doğrulukları

for alpha in alphas:
    weights = multi_model_extrapolation(models, [alpha] * (len(models) - 1))  # Tüm adımlar için alpha kullan
    models[0].set_weights(weights)
    
    # Eğitim doğruluğunu ölç
    train_acc = models[0].evaluate(x_train, y_train, verbose=0)[1]
    multi_model_train_accuracies.append(train_acc)
    
    # Test doğruluğunu ölç
    test_acc = models[0].evaluate(x_test, y_test, verbose=0)[1]
    multi_model_test_accuracies.append(test_acc)

# Grafik oluştur
plt.figure(figsize=(10, 6))
plt.plot(alphas, multi_model_train_accuracies, marker='o', label="Train Accuracy", color='blue', markersize=4)
plt.plot(alphas, multi_model_test_accuracies, marker='x', label="Test Accuracy", color='red', markersize=4)
plt.xlabel("Alpha (-5 to 5)")
plt.ylabel("Accuracy")
plt.title("Train & Test Accuracy with Extrapolation")
plt.legend()
plt.grid(True)
plt.show()



def interpolate_between_models(model1, model2, alphas, x_train, y_train, x_test, y_test):
    """
    İki model arasında interpolasyon yapar ve doğrulukları döndürür.
    """
    train_accuracies = []
    test_accuracies = []

    for alpha in alphas:
        # Ağırlıkları interpolate et
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        interpolated_weights = [
            (1 - alpha) * w1 + alpha * w2
            for w1, w2 in zip(weights1, weights2)
        ]
        model1.set_weights(interpolated_weights)

        # Eğitim ve test doğruluklarını hesapla
        train_acc = model1.evaluate(x_train, y_train, verbose=0)[1]
        test_acc = model1.evaluate(x_test, y_test, verbose=0)[1]

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    return train_accuracies, test_accuracies

# Alphas: -5 ile 5 arasında 100 eşit adım
alphas = np.linspace(1, 100, 100) / 100


# Görselleştirme için tüm model çiftlerini hesapla
for i in range(len(models) - 1):
    train_acc, test_acc = interpolate_between_models(
        models[i], models[i + 1], alphas, x_train, y_train, x_test, y_test
    )

    # Grafik oluştur
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, train_acc, marker='o', label=f"Train Accuracy (Model {i+1} -> Model {i+2})", color='blue', markersize=4)
    plt.plot(alphas, test_acc, marker='x', label=f"Test Accuracy (Model {i+1} -> Model {i+2})", color='red', markersize=4)
    plt.xlabel("Alpha (0 to 1)")
    plt.ylabel("Accuracy")
    plt.title(f"Interpolation Between Model {i+1} and Model {i+2}")
    plt.legend()
    plt.grid(True)
    plt.show()