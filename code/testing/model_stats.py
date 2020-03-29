# Designed to be ran from colab.google.com however can be ran on command line.
# This program can take muliple models and tests it against a dataset.
# The results are plotted and shown on a confusion matrix.
# It also includes a custom lost function.
# This program dowsnt include noise in the dataset.

from sklearn.metrics import confusion_matrix
import joblib, os, sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import keras.backend as K

DATADIR = sys.argv[1]
os.chdir(DATADIR)

# Custom loss
def custom_loss(y_true, y_predict):
    loss_variable = calculate_new_loss_variable(y_true, y_predict)
    return K.categorical_crossentropy(y_true, y_predict)  + loss_variable

def calculate_new_loss_variable(y_true, y_pred):
    y = K.argmax(y_true, axis=1)     # Array of all the indexes of max values for true values
    y_hat = K.argmax(y_pred, axis=1) # Array of all the indexes of max values for predictions
    
    tmp = K.cast(K.sum(K.abs(y - y_hat)), dtype='float32')
    loss =  tmp * 0.0001 # [0, 0.506]
    return loss

# Loading model / CNN
model_1 = load_model('model_1_even.h5')
model_2 = load_model('model_2_unchanged.h5')
model_3 = load_model('model_3_oversampled.h5')
model_4 = load_model('model_4_custom_loss_even.h5', custom_objects={'custom_loss': custom_loss})
models = [model_1, model_2, model_3, model_4]

# take images location as input
os.chdir(DATADIR + '/../Killian_test')

CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y"]  # Removed J, X and Z as they involve motion

INT_LABELS = [i for i in range(23)]

y_true = []
y_pred = []

wrong_predictions = [([0] * 23), ([0] * 23), ([0] * 23), ([0] * 23)]
right_predictions = [([0] * 23), ([0] * 23), ([0] * 23), ([0] * 23)]

images = os.listdir()
print(images)
counter = 0
for model in models:
    print("Testing: Model_{:}".format(counter))
    for image in images:
        # What the label actually is
        true = CATEGORIES.index(image.split(".")[0][0])
        y_true.append(true)

        # Image processing
        img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img_array = img_array/255.0
        img_array = cv2.resize(img_array, (150, 150))
        new_array =  np.array(img_array).reshape(-1, 150, 150, 1)

        # Predicted label array
        pred = model.predict(new_array, verbose = 0)
        pred = INT_LABELS[np.argmax(pred)]
        y_pred.append(pred)

        if true != pred:
            # Prediction was wrong
            wrong_predictions[counter][true] += 1
        
        else:
            # Prediction right
            right_predictions[counter][true] += 1
    
    # Plot results
    plt.plot(CATEGORIES, wrong_predictions[counter])
    plt.plot(CATEGORIES, right_predictions[counter])
    plt.show()

    # Confusion matrix plotting
    plot = plt.imshow(confusion_matrix(y_true, y_pred))
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.xticks([i for i in range(23)])
    plt.yticks([i for i in range(23)])
    plt.show()
    
    counter += 1
    # print(wrong_predictions)
    

# Plot total results
wrong = [0] * 23
right = [0] * 23
for i in range(23):
  wrong[i] = wrong_predictions[0][i] + wrong_predictions[1][i] + wrong_predictions[2][i] + wrong_predictions[3][i]
  right[i] = right_predictions[0][i] + right_predictions[1][i] + right_predictions[2][i] + right_predictions[3][i]

#Plotting predictions on a graph
plt.plot(CATEGORIES, wrong)
plt.plot(CATEGORIES, right)
plt.show()

print("------------")
print("Statistics:")
print("------------")
print("Number of correct predictions: {:}".format(sum(right)))
print("Number of incorrect predictions: {:}".format(sum(wrong)))
print("Letters that need oversampling ( wrong over 50% of the time ):")

for i in range(23):
    num_wrongs = wrong[i] # For that letter
    num_predictions = (len(images) // 23) * len(models)  # How many times was this letter predicted in the above code
    if num_wrongs > (num_predictions * 0.5): # If wrong over 50% of the time
        print(CATEGORIES[i])
