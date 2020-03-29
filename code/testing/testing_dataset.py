# This file can test any model against any data, as long as the location of models and dataset is added in the lines of code below

from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Loading model / CNN
model1 = load_model('/content/drive/My Drive/Pickles/files/changed_data/final_models/model_1_over.h5')
model2 = load_model('/content/drive/My Drive/Pickles/files/changed_data/final_models/model_over_04_03_2020.h5')

# take images location as input

#Load in pickle files
pickle_in = open("/content/drive/My Drive/Pickles/files/changed_data/final_pickles/X_test_even.pickle","rb")
X = joblib.load(pickle_in)
print("JOBLIB X done")
# X.shape should be (8864, 154, 154, 1)


pickle_in = open("/content/drive/My Drive/Pickles/files/changed_data/final_pickles/y_test_even.pickle","rb")
y = joblib.load(pickle_in)
print("JOBLIB y done")


test_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y", "Noise"]

X = X/255.0

# Graph to visualise the correct and incorrect prediction on each letter
wrong_predictions = [([0] * 24), ([0] * 24)]
right_predictions = [([0] * 24), ([0] * 24)]
INT_LABELS = [i for i in range(24)]

CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y", "Noise"]  # Removed J, X and Z as they involve motion

# Graph for confusion matrix
y_true = []
y_pred = []

models  = [model1, model2]
model_scores = [0] * len(models)    # score the models on the number of predictions theyy get right --> determines quickly which models are superior than others
num_pred = X.shape[0]               # Number of predictions in each iteration

counter = 0
for model in models:
    for i in tqdm(range(0, len(X))):
        # What the label actually is
        label = y[i]
        ind = list(label).index(1)
        y_true.append(ind)
        img_array = X[i]
        new_array =  np.array(img_array).reshape(-1, 150, 150, 1) # Ensure X[i] is the right shape for model
        
        # Predicted label array
        pred = model.predict(new_array, verbose = 0)
        pred = INT_LABELS[np.argmax(pred)]
        y_pred.append(pred)

        if ind != pred:
            # Prediction was wrong
            wrong_predictions[counter][ind] += 1
        
        else:
            # Prediction right
            right_predictions[counter][ind] += 1
            model_scores[counter] += 1

    # Graph of right and wrong predictions --> Orange is right, blue is wrong on google colab
    plt.plot(CATEGORIES, wrong_predictions[counter])
    plt.plot(CATEGORIES, right_predictions[counter])
    plt.show()

    # Score the model achieved
    print("The score of this model is {:} out of {:}".format(model_scores[counter], num_pred))
    print("Therefore the model was correct {:}% of the time on this testing dataset".format(int((model_scores[counter]/num_pred)*100)))

    # Confusion matrix
    plot = plt.imshow(confusion_matrix(y_true, y_pred))

    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.xticks([i for i in range(24)])
    plt.yticks([i for i in range(24)])
    plt.show()
    counter += 1

wrong = [0] * 24
right = [0] * 24
for i in range(24):
  wrong[i] = wrong_predictions[0][i] + wrong_predictions[1][i]
  right[i] = right_predictions[0][i] + right_predictions[1][i]

plt.plot(CATEGORIES, wrong)
plt.plot(CATEGORIES, right)
plt.show()

print("The best scoring model was {:}".format(models[model_scores.index(max(model_scores))])) 
