import os.path
import unittest
import submission_datasets as sbds
from tensorflow import keras

# change this to your models directory, move all your model h5 there
MODEL_DIR = 'Submission_B'  # make sure this directory contain all the h5 models
assert os.path.isdir(MODEL_DIR), "Model directory doesn't exist"

# Loading all datasets
X, Y = sbds.loadLinRegDataset('B')
(train_images_b3, train_labels_b3), (test_images_b3, testing_labels_b3) = sbds.loadImagesData('RPS')
(train_images_b2, train_labels_b2), (test_images_b2, testing_labels_b2) = sbds.loadImagesData('fashionMNIST')
training_padded, training_labels, test_padded, test_labels = sbds.loadTextsData('bbc')
# train_set, test_set = sbds.loadDailyMaxTemp()

class TestSubmissionB(unittest.TestCase):
    def testB1(self):
        global X, Y
        model_B1 = keras.models.load_model(os.path.join(MODEL_DIR,'model_B1.h5'))
        try:
            result = model_B1.evaluate(X, Y, verbose=0)
            print(f'result B1 : loss = {result}')
            self.assertTrue(result < 1e-3)
        except TypeError:
            result = model_B1.evaluate(X, Y, verbose=0)[0]
            print(f'result B1 : loss = {result}')
            self.assertTrue(result < 1e-3)

    def testB2(self):
        global train_images_b2, train_labels_b2, test_images_b2, testing_labels_b2
        model_B2 = keras.models.load_model(os.path.join(MODEL_DIR,'model_B2.h5'))
        _, result_train = model_B2.evaluate(train_images_b2, train_labels_b2, verbose=0)
        _, result_test = model_B2.evaluate(test_images_b2, testing_labels_b2, verbose=0)
        print(f'result B2 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test > 0.83 and result_train > 0.83)

    def testB3(self):
        global train_images_b3, train_labels_b3, test_images_b3, testing_labels_b3
        model_B3 = keras.models.load_model(os.path.join(MODEL_DIR,'model_B3.h5'))
        _, result_train = model_B3.evaluate(train_images_b3, train_labels_b3, verbose=0)
        _, result_test = model_B3.evaluate(test_images_b3, testing_labels_b3, verbose=0)
        print(f'result B3 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test > 0.83 and result_train > 0.83)

    def testB4(self):
        global training_padded, training_labels, test_padded, test_labels
        model_B4 = keras.models.load_model(os.path.join(MODEL_DIR,'model_B4.h5'))
        _, result_train = model_B4.evaluate(training_padded, training_labels, verbose=0)
        _, result_test = model_B4.evaluate(test_padded, test_labels, verbose=0)
        print(f'result B4 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test > 0.91 and result_train > 0.91)

    # def testB5(self):
    #     global train_set,test_set
    #     model_B5 = keras.models.load_model(os.path.join(MODEL_DIR,'model_B5.h5'))
    #     _, result_train = model_B5.evaluate(train_set, verbose=0)
    #     _, result_test = model_B5.evaluate(train_set, verbose=0)
    #     print(f'result B5 : val_accuracy = {result_test}, train_accuracy={result_train}')
    #     self.assertTrue(result_test < 0.2 and result_train < 0.2)

if __name__ == '__main__':
    unittest.main()