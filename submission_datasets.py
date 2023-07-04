import numpy as np
import os

DATA_DIR = 'data'
os.mkdir(DATA_DIR) if not os.path.isdir(DATA_DIR) else None
# === LINEAR === #
def loadLinRegDataset(submission):
    X, Y = np.load(os.path.join(DATA_DIR,f'dataset_{submission}1.npy'))
    return X, Y

# === IMAGES === #
def loadImagesData(name):
    train_images = np.load(os.path.join(DATA_DIR,f'images_{name}_train.npy'))
    test_images = np.load(os.path.join(DATA_DIR,f'images_{name}_test.npy'))
    train_labels = np.load(os.path.join(DATA_DIR,f'labels_{name}_train.npy'))
    testing_labels = np.load(os.path.join(DATA_DIR,f'labels_{name}_test.npy'))
    # print(train_images.shape)
    # print(test_images.shape)
    # print(train_labels.shape)
    # print(testing_labels.shape)
    return (train_images, train_labels), (test_images, testing_labels)


# === TEXTS === #
def loadTextsData(name):
    train_texts = np.load(os.path.join(DATA_DIR,f'texts_{name}_train.npy'))
    test_text = np.load(os.path.join(DATA_DIR,f'texts_{name}_test.npy'))
    train_labels = np.load(os.path.join(DATA_DIR,f'labels_{name}_train.npy'))
    testing_labels = np.load(os.path.join(DATA_DIR,f'labels_{name}_test.npy'))
    # print(train_texts.shape)
    # print(test_text.shape)
    # print(train_labels.shape)
    # print(testing_labels.shape)
    return train_texts, train_labels, test_text, testing_labels


# === TIME SERIES === #

# def loadTimeSeries(name):
#     train_set = np.load(os.path.join(DATA_DIR,f'windows_{name}_train.npy'), allow_pickle=True)
#     test_set = np.load(os.path.join(DATA_DIR,f'windows_{name}_test.npy'), allow_pickle=True)
#     train_labels = np.load(os.path.join(DATA_DIR,f'labels_{name}_train.npy'), allow_pickle=True)
#     testing_labels = np.load(os.path.join(DATA_DIR,f'labels_{name}_test.npy'), allow_pickle=True)
#     # print(train_set.shape)
#     # print(test_set.shape)
#     # print(train_labels.shape)
#     # print(testing_labels.shape)
#     return train_set, train_labels, test_set, testing_labels

# loadTimeSeries('sunspots')