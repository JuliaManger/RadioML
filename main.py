import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping



# LOAD DATASET
with open("RML2016.10a_dict.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

snrs = sorted(set(k[1] for k in data.keys()))
mods = sorted(set(k[0] for k in data.keys()))

X, labels, snr_labels = [], [], []
for mod in mods:
    for snr in snrs:
        samples = data[(mod, snr)]
        X.append(samples)
        labels     += [mods.index(mod)] * len(samples)
        snr_labels += [snr] * len(samples)

X = np.vstack(X).astype(np.float32) 

y = np.array(labels)
snr_labels = np.array(snr_labels)


# SPLIT DATASET

idx = np.random.permutation(len(X))
split = int(0.7 * len(X))
train_idx, test_idx = idx[:split], idx[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]
snr_test         = snr_labels[test_idx]

X_train = X_train[:, :, :, np.newaxis]  
X_test  = X_test[:, :, :, np.newaxis]

y_train_cat = tf.keras.utils.to_categorical(y_train, len(mods))
y_test_cat  = tf.keras.utils.to_categorical(y_test,  len(mods))


# MODEL FROM THE PAPER (CNN2)
model = models.Sequential([
    layers.Input(shape=(2, 128, 1)),
    layers.Conv2D(256, (1, 3), activation='relu', padding='same'),
    layers.Conv2D(80, (2, 3), activation='relu', padding='valid'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(len(mods), activation='softmax'),
])



model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()



# TRAINING
history = model.fit(
    X_train, y_train_cat,
    batch_size=1024,
    epochs=2,
    validation_split=0.1
)



# PLOTTING

# Accuracy vs SNR
acc_per_snr = {}
for snr in snrs:
    mask = snr_test == snr
    loss, acc = model.evaluate(X_test[mask], y_test_cat[mask], verbose=0)
    acc_per_snr[snr] = acc

plt.figure()
plt.plot(snrs, [acc_per_snr[s] for s in snrs], marker='o')
plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy")
plt.title("Accuracy vs SNR"); plt.grid(True)
plt.tight_layout(); plt.savefig("acc_vs_snr.png"); plt.show()



# Confusion Matrix

def plot_confusion_matrix(snr_value, cmap='Blues'):
    mask = snr_test == snr_value

    preds = model.predict(X_test[mask]).argmax(axis=1)

    cm = confusion_matrix(y_test[mask], preds)

    # Normalise row-wise
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))

    plt.imshow(cm_norm, vmin=0, vmax=1, cmap=cmap)

    plt.xticks(range(len(mods)), mods, rotation=45, ha='right')
    plt.yticks(range(len(mods)), mods)

    cbar = plt.colorbar()
    cbar.set_label("Normalized Accuracy")

    plt.title(f"Confusion Matrix at {snr_value} dB SNR")

    plt.tight_layout()

    plt.savefig(f"confusion_matrix_{snr_value}dB.png")

    plt.show()


plot_confusion_matrix(18, cmap='Blues')

plot_confusion_matrix(-6, cmap='Blues')
