import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def load_quickdraw_data(path, num_samples_per_class=5000, img_size=(28, 28)):
    classes = sorted(os.listdir(path))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    X, y = [], []

    for cls in classes:
        cls_path = os.path.join(path, cls)
        data = np.load(cls_path)
        if len(data) > num_samples_per_class:
            data = data[:num_samples_per_class]
        X.append(data)
        y.extend([class_to_idx[cls]] * len(data))

    X = np.vstack(X).reshape(-1, *img_size, 1) / 255.0
    y = np.array(y)
    return X, y, classes


data_path = './dataset'
num_samples_per_class = 10000
X, y, class_names = load_quickdraw_data(data_path, num_samples_per_class=num_samples_per_class)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

def build_resnet(input_shape=(28, 28, 1), num_classes=345):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(2):
        shortcut = x
        x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)

    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(2):
        shortcut = x
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(2):
        shortcut = x
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_resnet()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

batch_size = 128
epochs = 200

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    steps_per_epoch=len(X_train) // batch_size,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save('resnet_quickdraw_model.h5')
print("Finish!")
