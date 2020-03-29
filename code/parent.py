# Program which calls all programs concurrently i.e. parent program

import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Import home made programs
from text_to_audio.txtToAudio import txtToAud
from text_to_audio.spell_checker import check_spelling, checker
from input_preprocessing.split_frames import split_into_frames
from input_preprocessing.check_dup import check_dup

oversample_letters = {"A":1, "F":1, "N":1, "T":1, "U":1, "Noise":1} #[0, 1, 3, 5, 11, 12, 17, 18, 19]
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y", " "]


def confidence_level(pred1, pred2):#, pred3):
    pred1 = (np.array(pred1)*100)/2
    pred2 = (np.array(pred2)*100)/2
    #pred3 = (np.array(pred3)*100)/3
    max_letter_pred1 = LABELS[np.argmax(pred1)]
    max_letter_pred2 = LABELS[np.argmax(pred2)]
    #max_letter_pred3 = LABELS[np.argmax(pred3)]
    max_precentage_1 = np.amax(pred1) # Out of 33%
    max_precentage_2 = np.amax(pred2)
    #max_precentage_3 = np.amax(pred3)
    pred1 = np.append(pred1, [0])
    combine_of_all = LABELS[np.argmax(pred1 + pred2)]# + pred3)]
    '''
    Since model 3 is speci alised in oversampled letters
    if they are all different and the third model is picking up 
    on one of the overfitted letters the final prediction should be 
    solely this model
    '''
    
    strong_1 = ["A", "B", "C", "G", "I", "N", "W", "E", "U", "T"] 
    strong_2 = ["C", "D", "I", "M", "O", "Q", "R", "Y", "L", "K"] 
    if max_letter_pred1 in strong_1:
        final_pred = max_letter_pred1
    elif max_letter_pred2 in strong_2:
        final_pred = max_letter_pred2
    else:
        final_pred = combine_of_all

#    if max_precentage_1 < 20:
#        final_pred = ""
#    else:
#        final_pred = max_letter_pred1
#    final_pred = "?"
#    if max_letter_pred1 != max_letter_pred2 != max_letter_pred3:
#        if (max_letter_pred3 in oversample_letters) and (max_precentage_3 > 25):
#            final_pred = max_letter_pred3
#
#        else:
#            final_pred = combine_of_all #add them and hope for the best because the models havent a clue
#
#    elif max_letter_pred1 == max_letter_pred2: #if the first two models guess the same then that is the answer
#        final_pred = LABELS[np.argmax(pred1 + pred2)]
#
#    else:
#        final_pred = combine_of_all

    #for seeing the outputs of each model   
    pred1 = pred1.reshape(24)
    pred2 = pred2.reshape(24)
    #pred3 = pred3.reshape(24)
    plt.plot(LABELS, pred1)
    #plt.show()
    plt.plot(LABELS, pred2)
    #plt.show()
    #plt.plot(LABELS, pred3)
    #plt.show()
    return final_pred

def main():
    # Current working directory
    DATADIR = os.getcwd()

    # Get image location off the commandline
    VIDEO_LOCATION = sys.argv[1]

    # Feed video to preprocessing program
    img_array = split_into_frames(VIDEO_LOCATION, sys.argv[2])

    # Loading models / CNN
    #MODEL_LOCATION_1 = os.path.join(DATADIR, "CNN", "models", "final_models", "model_1_even.h5")
    #MODEL_LOCATION_2 = os.path.join(DATADIR, "CNN", "models", "final_models", "model_1_over.h5")
    #MODEL_LOCATION_3 = os.path.join(DATADIR, "CNN", "models", "final_models", "model_3_oversampled.h5")

    model1 = load_model("/home/killian/Downloads/model_1_even.h5")
    model2 = load_model("/home/killian/Downloads/model_even_24.h5")
    #model3 = load_model(MODEL_LOCATION_3)
    #model2 = load_model("MODEL_NAME_for_2")
    #model3 = load_model("MODEL_NAME_for_3")

    # Possible output of the model
    output = ""

    # Feed images to the models
    for image in range(0,len(img_array)-1):

        curr_image = img_array[image]
        next_image = img_array[image + 1]

        print(check_dup(curr_image, next_image))

        #If the image is so similar to the next image then its a duplicate and continue with loop
        if check_dup(curr_image, next_image) < 0.03:
            continue

        curr_image = curr_image/255.0
        next_image = next_image/255.0

        imgplot = plt.imshow(curr_image)
        #plt.show()

        curr_image = np.array(curr_image).reshape(-1, 150, 150, 1)
        next_image = np.array(next_image).reshape(-1, 150, 150, 1)

        prediction1 = model1.predict(curr_image, verbose=0)
        prediction2 = model2.predict(curr_image, verbose=0)
        #prediction3 = model3.predict(curr_image, verbose=0)
        final_predict = confidence_level(prediction1, prediction2)#, prediction3)
        #If the model is getting 3 letters in a row, too many frames was takin in of the same letter
        if len(output) > 2 and final_predict == output[-1] and final_predict == output[-2]:
            continue
        output += final_predict

    print("Output from the models : {}".format(output))
    output = check_spelling(output)
    print("Output from the spell checking algorithm : {}".format(output))
    #txtToAud(output)

if __name__ == '__main__':
    main()