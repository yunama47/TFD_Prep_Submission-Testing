import os.path
import unittest
# import submission_datasets_copy as dl
import datasets_loader as dl
from tensorflow import keras

# change this to your models directory, move all your model h5 there
MODEL_DIR = 'Submission_C'  # make sure this directory contain all the h5 models
assert os.path.isdir(MODEL_DIR), "Model directory doesn't exist"

# Loading all datasets
X, Y = dl.loadLinRegDataset('C')
(train_images_c2, train_labels_c2), (test_images_c2, testing_labels_c2) = dl.loadMNIST()
train_generator, validation_generator = dl.loadCatsAndDogs(localDataDir='data', download=False)

training_padded, training_labels, test_padded, test_labels = dl.loadSarcasm(localJson='data/sarcasm.json', download=False)
train_set, test_set = dl.loadDailyMinTemp(local_csv='data/daily-min-temperatures.csv', download=False)

class TestSubmissionC(unittest.TestCase):
    def testC1(self):
        global X, Y
        model_C1 = keras.models.load_model(os.path.join(MODEL_DIR,'model_C1.h5'))
        try:
            result = model_C1.evaluate(X, Y, verbose=0)
            print(f'result C1 : loss = {result}')
            self.assertTrue(result < 1e-4)
        except TypeError:
            result = model_C1.evaluate(X, Y, verbose=0)[0]
            print(f'result C1 : loss = {result}')
            self.assertTrue(result < 1e-4)

    def testC2(self):
        global train_images_c2, train_labels_c2, test_images_c2, testing_labels_c2
        model_C2 = keras.models.load_model(os.path.join(MODEL_DIR,'model_C2.h5'))
        _, result_train = model_C2.evaluate(train_images_c2, train_labels_c2, verbose=0)
        _, result_test = model_C2.evaluate(test_images_c2, testing_labels_c2, verbose=0)
        print(f'result C2 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test > 0.91 and result_train > 0.91)

    def testC3(self): # cats dogs
        global train_generator, validation_generator 
        model_C3 = keras.models.load_model(os.path.join(MODEL_DIR,'model_C3.h5'))
        _, result_train = model_C3.evaluate(train_generator, verbose=0)
        _, result_test = model_C3.evaluate(validation_generator, verbose=0)
        print(f'result C3 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test > 0.72 and result_train > 0.72)

    def testC4(self):
        global training_padded, training_labels, test_padded, test_labels
        model_C4 = keras.models.load_model(os.path.join(MODEL_DIR,'model_C4.h5'))
        _, result_train = model_C4.evaluate(training_padded, training_labels, verbose=0)
        _, result_test = model_C4.evaluate(test_padded, test_labels, verbose=0)
        print(f'result C4 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test > 0.75 and result_train > 0.75)

    def testC5(self):
        global train_set,test_set
        model_C5 = keras.models.load_model(os.path.join(MODEL_DIR,'model_C5.h5'))
        _, result_train = model_C5.evaluate(train_set, verbose=0)
        _, result_test = model_C5.evaluate(train_set, verbose=0)
        print(f'result C5 : val_accuracy = {result_test}, train_accuracy={result_train}')
        self.assertTrue(result_test < 0.19 and result_train < 0.19)

if __name__ == '__main__':
    unittest.main()