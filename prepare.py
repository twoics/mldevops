import os
from PIL import Image
import tqdm
import numpy as np
import random


def load_all_data(path):
    food = []
    non_food = []
    for folder in tqdm.tqdm(os.listdir(path)):
        for filename in os.listdir(f'{path}/{folder}/food'):
            if random.random() < 0.4:
                food.append(Image.open(f'{path}/{folder}/food/{filename}'))
        for filename in os.listdir(f'{path}/{folder}/non_food'):
            if random.random() < 0.4:
                non_food.append(Image.open(f'{path}/{folder}/non_food/{filename}'))
    return food, non_food


def to_square(image: Image.Image, size=256):
    '''Функция для приведения изображения к квадрату заданого размера, с помощью добавления черных полос'''
    width, height = image.size
    if width == height:
        return image.resize((size, size))
    elif width > height:
        result = Image.new(image.mode, (width, width), (0, 0, 0))
        result.paste(image, (0, (width - height) // 2))
        return result.resize((size, size))
    else:
        result = Image.new(image.mode, (height, height), (0, 0, 0))
        result.paste(image, ((height - width) // 2, 0))
        return result.resize((size, size))
    

def main():
    random.seed(0)

    food, non_food = load_all_data('data')
    non_food = [(image if image.mode == 'RGB' else image.convert('RGB')) for image in non_food]

    food_square = [to_square(image, size=100) for image in tqdm.tqdm(food)]
    non_food_square = [to_square(image, size=100) for image in tqdm.tqdm(non_food)]

    objects = np.array([np.array(image) for image in food_square + non_food_square])
    labels = np.array([1] * len(food_square) + [0] * len(non_food_square))

    RATIO = (60, 20, 20) # train/test/valid
    RATIO = [sum(RATIO[:i + 1]) for i in range(len(RATIO))]
    length = len(labels)
    all_index = np.array(range(length))
    np.random.shuffle(all_index)
    train_indexes = all_index[:length * RATIO[0] // 100]
    test_indexes = all_index[length * RATIO[0] // 100:length * RATIO[1] // 100]
    valid_indexes = all_index[length * RATIO[1] // 100:]

    print('Размеры тренировочного, тестового, валидационного набора')
    len(train_indexes), len(test_indexes), len(valid_indexes)
    print(f'Форма объектов {objects.shape[1:]}')

    train_labels = labels[train_indexes]
    train_images = objects[train_indexes]
    valid_labels = labels[valid_indexes]
    valid_images = objects[valid_indexes]
    test_labels = labels[test_indexes]
    test_images = objects[test_indexes]

    os.mkdir('prepared')


    with open('prepared/train_labels', 'wb') as file:
        np.save(file, train_labels)
    with open('prepared/train_images', 'wb') as file:
        np.save(file, train_images)
    with open('prepared/valid_labels', 'wb') as file:
        np.save(file, valid_labels)
    with open('prepared/valid_images', 'wb') as file:
        np.save(file, valid_images)
    with open('prepared/test_labels', 'wb') as file:
        np.save(file, test_labels)
    with open('prepared/test_images', 'wb') as file:
        np.save(file, test_images)


if __name__ == '__main__':
    main()
