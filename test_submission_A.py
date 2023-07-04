import os.path
import unittest
import submission_datasets as sbds
from tensorflow import keras

# change this to your models directory, move all your model h5 there
MODEL_DIR = 'Submission_A'  # make sure this directory contain all the h5 models
assert os.path.isdir(MODEL_DIR), "Model directory doesn't exist"

# Loading all datasets
X, Y = sbds.loadLinRegDataset('A')
(train_images, train_labels), (test_images, testing_labels) = sbds.loadImagesData('horse_human')
training_padded, training_labels, test_padded, test_labels = sbds.loadTextsData('imdb')
# train_set, train_labels, test_set, testing_labels = sbds.loadTimeSeries('sunspots')

class TestSubmissionA(unittest.TestCase):
    def testA1(self):
        global X, Y
        model_A1 = keras.models.load_model(os.path.join(MODEL_DIR,'model_A1.h5'))
        try:
            result = model_A1.evaluate(X, Y, verbose=0)
            print(f'result A1 : loss = {result}')
            self.assertTrue(result < 1e-4)
        except TypeError:
            result = model_A1.evaluate(X, Y, verbose=0)[0]
            print(f'result A1 : loss = {result}')
            self.assertTrue(result < 1e-4)

    def testA2(self):
        global train_images, train_labels, test_images, testing_labels
        model_A2 = keras.models.load_model(os.path.join(MODEL_DIR,'model_A2.h5'))
        _, result_train = model_A2.evaluate(train_images, train_labels, verbose=0)
        _, result_test = model_A2.evaluate(test_images, testing_labels, verbose=0)
        print(f'result A2 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test > 0.83 and result_train > 0.83)

    def testA3(self):
        global train_images, train_labels, test_images, testing_labels
        model_A3 = keras.models.load_model(os.path.join(MODEL_DIR,'model_A3.h5'))
        _, result_train = model_A3.evaluate(train_images, train_labels, verbose=0)
        _, result_test = model_A3.evaluate(test_images, testing_labels, verbose=0)
        print(f'result A3 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test > 0.97 and result_train > 0.97)

    def testA4(self):
        global training_padded, training_labels, test_padded, test_labels
        model_A4 = keras.models.load_model(os.path.join(MODEL_DIR,'model_A4.h5'))
        _, result_train = model_A4.evaluate(training_padded, training_labels, verbose=0)
        _, result_test = model_A4.evaluate(test_padded, test_labels, verbose=0)
        print(f'result A4 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test > 0.83 and result_train > 0.83)

    # def testA5(self):
    #     global train_set, train_labels,test_set, testing_labels
    #     model_A5 = keras.models.load_model(os.path.join(MODEL_DIR,'model_A5.h5'))
    #     _, result_train = model_A5.evaluate(train_set, train_labels, verbose=0)
    #     _, result_test = model_A5.evaluate(test_set, testing_labels, verbose=0)
    #     print(f'result A5 : val_accuracy = {result_test}, train_accuracy={result_train}')
    #     self.assertTrue(result_test < 0.15 and result_train < 0.15)

if __name__ == '__main__':
    unittest.main()
