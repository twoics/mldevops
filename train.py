import numpy as np
import tensorflow as tf
from json import dump


def main():
    train_labels = np.load('prepared/train_labels')
    train_images = np.load('prepared/train_images')
    valid_labels = np.load('prepared/valid_labels')
    valid_images = np.load('prepared/valid_images')
    test_labels = np.load('prepared/test_labels')
    test_images = np.load('prepared/test_images')
    model1 = tf.keras.Sequential(
        (tf.keras.layers.Conv2D(24, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(48, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(96, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'))
    )
    model2 = tf.keras.Sequential(
        (tf.keras.layers.Conv2D(24, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(48, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(96, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(800, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'))
    )
    model3 = tf.keras.Sequential(
        (tf.keras.layers.Conv2D(24, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(96, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(192, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(800, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'))
    )
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()])
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()])
    model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()])
    results1valid = []
    results1train = []
    results1test = []
    results2valid = []
    results2train = []
    results2test = []
    results3valid = []
    results3train = []
    results3test = []
    for i in range(15):
        print(f'Model 1 Epoch {i + 1}')
        model1.fit(train_images, train_labels, 64)
        results1valid.append(model1.evaluate(valid_images, valid_labels, 32)[1])
        results1test.append(model1.evaluate(test_images, test_labels, 32)[1])
        results1train.append(model1.evaluate(train_images, train_labels, 32)[1])
    for i in range(15):
        print(f'Model 2 Epoch {i + 1}')
        model2.fit(train_images, train_labels, 64)
        results2valid.append(model2.evaluate(valid_images, valid_labels, 32)[1])
        results2test.append(model2.evaluate(test_images, test_labels, 32)[1])
        results2train.append(model2.evaluate(train_images, train_labels, 32)[1])
    for i in range(15):
        print(f'Model 3 Epoch {i + 1}')
        model3.fit(train_images, train_labels, 64)
        results3valid.append(model3.evaluate(valid_images, valid_labels, 32)[1])
        results3test.append(model3.evaluate(test_images, test_labels, 32)[1])
        results3train.append(model3.evaluate(train_images, train_labels, 32)[1])
    with open('./summary.json', 'w') as file:
        dump(
            {'model1': max(results1test), 'model2': max(results2test), 'model3': max(results3test)},
            file
            )

if __name__ == '__main__':
    main()
