# Technical Manual

## 0. Table of contents
1 Introduction
-1.1 Overview
-1.2 Glossary

2 System Architecture 
-2.1 Video Processing 
-2.1.1 Split video into multiple frames
-2.1.2 Detect position of hands in images
-2.2 Convoluted Neural Network
-2.2.1 Applying filters to a frame
-2.2.2 Classify the image vector
-2.3 Accumulation of text output
-2.4 Converting to audio

3 High Level Design

4 Problems and Resolutions
-4.1 Overfitting
-4.2 Multiple frames issue
-4.3 Vanishing Gradient Problem
-4.3 Hardware issue

5 Testing

6 Installation Guide

## 1. Introduction
### 1.1 Overview

This system utilises Convoluted Neural Networks (CNN) and computer vision libraries to translate a video of an individual conversing in Irish Sign Language (ISL) into audio. The program runs from the command line and takes the absolute file location of the video from the command line. This project, in essence, is to produce a program which allows for free flowing communication between a person without the ability of speech, but with the ability to perform sign language, and a person with the ability of speech, but without the ability to understand or communicate in sign language.

The functions of the system are as follows:
- Converts a video into frames, applying image processing to convert the frames into multi-dimensional arrays and feed them to a trained CNN model.
- Accumulates the output of the CNN models to plain text.
- Output the text produced by the CNN into audio, and save the audio file into the current working directory.

The program is compatible with both Windows 10 and Ubuntu, a Linux distro. Python 3 and the appropriate libraries installed are required to execute the program, as discussed in the installation guide.

### 1.2 Glossary
CNN - Convoluted Neural Network.
ISL - Irish Sign Language 
CPU - Central Processing Unit
GPU - Graphic Processing Unit

## 2. System Architecture

### 2.1 Video Processing

#### 2.1.1 Split video into multiple frames
The inputted video is split into multiple frames based on the number of frames per second the system requires. The Open-CV library was utilized to split the frames.

#### 2.1.2 Applying filters to a frame
The image is converted to grayscale and resized into a (150 x 150) image, as this is the size of the input the CNN has been trained on and therefore requires. The image is padded if necessary. For instance, if the original ratio of the image is not a square, then it is padded to make it so before resizing. Open-CV was used to convert the images into grayscale and resizing them.

#### 2.1.3 Converting the image to a numpy array
Numpy is a multidimensional array library supported by python. The image is converted into a numpy array. This array will have the shape (150, 150, 1) i.e. a single 2-D array of 150 rows and 150 columns of pixels.

### 2.2 Convoluted Neural Network
#### 2.2.1 Prediction of the image vector
The image array is fed into the neural network. The neural network makes a prediction on the image array based on the predictions of two models. These models are built using keras API and tensorflow as backend. These models also make use of Keras' pretrained model Xception. This is a model which has been trained on the ImageNet dataset and can be incorporated into models to improve accuracy.

#### 2.2.2 Prediction Confidence Level
A prediction is made on the image vector by two models trained separately. One of the models is stronger in predicting some letters compared to the other model and vice versa. If the model does not perform strongly on the input and the other model does, the final prediction is obtained from the stronger model.

### 2.3 Accumulation of text output
The duplicate checking function checks the difference between the current frame and the frame after. If the difference is under a certain amount then the two frames are the same. This utilises a built-in method in numpy called numpy.linalg.norm() that calculates the euclidean distance between two image arrays.
If this method does not catch the unnecessary duplicates and thus the model outputs three of the same letters in a row, then the third letter will not be accepted. The output of the CNN is then collected and accumulated.

### 2.4 Run spell checking algorithm on outputted values
Irrelevant frames cause the models to predict duplicates and unneeded predictions, which are accumulated along with wanted predictions. To make up for this, a spell checking algorithm is needed, to rectify the output. Hence, the output from the CNN is fed into a spell checking algorithm. This spell checking algorithm is an imported library called SpellChecker.

### 2.5 Converting to audio
The accumulated text is then passed to a text-to-speech algorithm, which outputs the text as audio to the user. The gTTS library uses Google's text-to-speech API. Finally, the audio is saved to a file.

![](https://i.imgur.com/S5fwpUA.png)
System architecture

## 3. High-Level Design
This section should set out the high-level design of the system. It should include system models showing the relationship between system components and the systems and its environment. These might be object-models, DFD, etc. Unlike the design in the Functional Specification - this description must reflect the design of the system as it is demonstrated.

![](https://i.imgur.com/jCSB3Mw.png) Low-Level Convolution Neural Network Architecture

![](https://i.imgur.com/2op3fr1.png)
High-Level Convolution Neural Network Architecture

![](https://i.imgur.com/BMcxg2g.png)
Dataflow diagram for creating and testing a model


## 4. Problems and Resolution
This section should include a description of any major problems encountered during the design and implementation of the system and the actions that were taken to resolve them.

During the course of this project many problems were encountered.

### 4.1 Overfitting & Underfitting
The first major problem encountered was getting the models to not overfit. Overfitting occurs when a model is fitting too close to the training data. This resulted in a training accuracy significantly larger than the validation accuracy, and a training loss significantly lower than the validation loss. This was especially noticeable when running tests on supposed >80% accurate models but getting poor results more in the region of 5% - 15%. Overfitting is a major issue in the training of neural networks, and thus needed immediate rectification. This problem was solved by the implementation of a multitude of methods and tests.

Firstly, the amount of data fed to our neural network was increased in order to force it to generalise more and not memorise images. Secondly, we decided on tuning all of the hyper parameters within the architecture. This included the amount of convolution layers, dense layers, filters and window sizes. Adjusting these hyperparameters outputted a model that surpassed the accuracy of the previous model. Once we completed the tuning of these parameters we achieved an accuracy of 85%. However, the validation accuracy on unseen data was approximately 10%, resulting in the conclusion that the models were still overfitting.

Regularisation techniques were also implemented. This is a term which encompasses a range of techniques and modifications to improve the generalisation of the model. The first regularisation method that was adopted into the model was the inclusion of Dropout layers. This is a layer included in the Keras API which is generally placed between two fully connected layers. This layer takes a rate between, and including, 0 and 1, and randomly sets that fraction of connections to zero at each update during training. This helps to improve generalisation by reducing the model's over reliance on some connections, as these connections won't always be present during training. This allows the model to use all connections between layers. The right dropout rate to prevent overfitting is dependent on the model. A solution to this was to write a program which would test the same model with various dropout rates between 0 and 1, in order to determine the best rate for that architecture. Too low of a dropout rate would result in a model which still significantly overfits and too high of a dropout would prevent a model learning and result in the model underfitting the data.

Moreover, l2 regularisation was also used to prevent overfitting. This means that a penalty was applied along with the loss function during optimisation, in order to reduce the size of weights, thus reducing the complexity of the model. This penalty is calculated by multiplying the l2 value by the sum of the squared coefficients of a layer. These penalties are applied on a layer by layer basis, and with the Keras API the l2 value is defined in the kernel_regularizer parameter of a layer. Similar to selecting dropout rates, if too large of a l2 value is used, then the coefficients of that layer will be too small to learn features. Inversely, if the l2 value is too small then overfitting will still occur. Therefore, multiple l2 values were tested on various layers in order to find the best values to minimise overfitting.

### 4.2 Vanish Gradient Problem
After the implementation of the outlined regularisation methods, the model built from scratch had a severe reduction in overfitting. The training and validation accuracies were now two in the same. However, another problem was encountered. While both the training and validation accuracies would increment relatively in tandem after each epoch, they consistently reached a stage, in training, where reductions in loss would be so small that the weights in the model would not update. This caused the accuracy of the model to stagnate, preventing further learning. The resulting model would consistently reach around 60% training accuracy and a validation accuracy of around 54% percent. After some research, it was discovered that this is a common problem for neural networks to encounter, and is known as the vanishing gradient problem. At this stage the time remaining to complete the project was dwindling, so the decision was made to use transfer learning solutions in an attempt to boost model accuracy quickly.

Furthermore, Keras' API includes multiple pretrained models which have primarily been trained on the ImageNet database, a massive image database of around 14 million images. These models are accessible through keras.applictions, and come with pretrained weights. Each pretrained model has different architecture, each of which are very large and deep networks, so the right model for the task at hand had to be deduced by means of testing. The conclusion after testing was that Xception an InceptionResNetV2 were the best models at predicting hand signs, however we went with Xception as it had a significantly smaller time per epoch than InceptionResNetV2.

However, despite the introduction of pretrained weights the validation accuracy of the model was just surpassing 70%. After running visualising tests and calculating statistics, it was found that the model was consistently mispredicting a certain number of letters / signs. To fix this, the code which serialised the datasets was changed drastically. A function was added that added random solid grey background noise to the images in order to improve the generalisation of the model. The serialised training and testing datasets were made to contain an even number of images for each label / letter, in hopes that the model did not overfit certain letters to reduce loss. As well as that, separate serialised datasets were created in which the letters which were mispredicted had been oversampled. This means that more images of these labels were added to the dataset, in hopes that the model would correctly predict these labels. This resulted in two models, one trained on even data and another trained on oversampled data. A combination of both these models is used in the final program.

Moreover, a custom loss function was also implemented in hopes of increasing the accuracy of the models. This loss function added a penalty onto the loss function we were already using called categorical_crossentrophy, a loss function used for multi-labelled models with one-hot encoded labels i.e. array of all 0's except a single 1 in the index of the value to be predicted. The added penalty was calculated based on a predicted labels perceived distance from the true label. For instance, if the true value was an "A", but the predicted value was a "R", then the loss function would add a larger penalty than if the predicted value was a "C". The thinking behind this is that the penalty will push the predictions closer to the true labels. However, this loss function did not succeed in increasing the accuracy of the model.

### 4.3 Multiple frames issue
One of the major problems with the implementation of our project was the task of discarding irrelevant frames, frames of a sign transitioning into another sign. Also, how long a particular sign lasts. If the rate of sign is slow then our program will take in multiple frames of the same sign. 

To combat this problem we implemented a few features to improve the final output of our system. Firstly, we created a function which checks the current frame and the next. We utilise a function in numpy which outputs a number of how similar the arrays are. If the similarity is below a certain number we discard the current frame.
Since hands are constantly moving, this function does not catch all the duplications. So, secondly, we do not accept more than two of the same letters from the predictions of the inputs. 
Lastly, the current output of our program might still contain duplicates so we decided on implementing a spell checking algorithm as our final solution for this problem.

### 4.4 Hardware issue
Throughout this project we were constantly on the search for a better way to run our neural network, since our personal computers do not contain the optimum architecture for running machine learning programs. Google Colab was used to tackle this problem. Google Colab to utilise Googles cloud services while leveraging the powered hardware, such as CPUs and GPUs.

## 5. Testing

We created many programs that test different combinations of convoluted layers, dropout layers and their rates, kernel regularizations(L1,L2 as mentioned above), kernel initialization and other hyper parameters such as number of convolutional filters, window size, activations and optimizers.

From our testing we concluded the following:
1. Three convolution layers are the optimum amount of layers
2. After these convolutions follow an activation layer. We deduced that the 'relu' activation is most suitable for the inner layers of neural network. We also concluded that a softmax activation works best in the output layer as it normalises the output array into a probability distribution.
3. Dropout rate was tested on a range between 0 and 1 on the model from scratch. It was concluded that the best dropout rate to prevent both underfitting and overfitting on this model architecture was a dropout rate of 0.6. However, with subsequent pretrained models, the dropout rate which yielded the best results was a dropout rate of between 0.45 and 0.5. This was a single dropout layer after the output from the Xception had been through a MaxPooling2D layer.
4. l2 kernel regularization parameters were also tested. We concluded that a value of 0.0001 produced the most satisfactory results.
5. The kernel initialization
6. The window size parameter within each convolution, decreases for each convolution, starting with a 7 by 7 window and for the final convolution a 3 by 3 window size.
7. Finally, the first convolution should have an input size of 16 and increase quadratically with the final convolution at 64 input size.

Note: All testing code is available on the gitlab repository.

Each one of the models was tested using seen and unseen datasets to convey the accuracy and loss. We created programs to test all the models that we trained called TensorBoard. TensorBoard graphs the performance of accuracy, validation accuracy, loss and validation loss. The logs of the performance are contained in our repositories. We also created datasets of pictures that we created ourselves using images that we had taken.

After the testing of these models, with the optimum hyper parameters and amount of layers, the model obtained an accuracy of 85% and validation accuracy of 10%. Unsatisfied with these results, we searched for better solutions. The concept of transfer learning involves the process of storing knowledge gained while solving a problem and applying it to a related problem. Pretrained models refers to models with a pre made architecture and pre trained weights. We training models using several Keras transfer learning models to achieve the most precise model, these models were also graphed in TensorBoard. The model that accomplished the superior results was called Xception. 

![](https://i.imgur.com/FV2ADoF.png)
Our final model received an accuracy of 97% and validation accuracy of 88%.
![](https://i.imgur.com/GKtYD0A.png)
TensorBoard showing validation accuracy, accuracy, validation loss and loss per epoch for models with changes in the amount of convolution layers, the rate of the dense layer and number of nodes regarding the input sizes for the convolutions. 
![](https://i.imgur.com/6ZQ07pI.png) 
TensorBoard showing validation accuracy, accuracy, validation loss and loss per epoch for Keras pretrained models.

## 6. Installation Guide

**Linux - Ubuntu 18.04**

Ubuntu is more suited for running the project, therefore we recommend using this OS. This is because the installation of necessary files and libraries is much more straightforward. Moreover, executing the following lines of code is needed before the executing our program. Type or copy & paste the following code into your command prompt to set up the required libraries and dependencies.

* Python3 (version 3.6.9):
`sudo apt-get install python3.6`
This is the computer language that will run our programs. This language comes pre installed in Ubuntu already. It is included if it has been uninstalled.

* Pip (version 9.0.1):
`sudo apt install python3-pip`
Pip is for installation of certain python libraries. For example:
![](https://i.imgur.com/Pu4AImi.png)

* Tensorflow (version 2.1.0): 
`pip install tensorflow`
Tensorflow library is a machine learning library for neural networks.

* Numpy (version 1.18.1):
`pip apt install python-numpy`
This library represents images as a multidimensional array.

* Cv2 (version 3.2.0):
`pip install opencv-python`
Cv2 is an image processing library.

* SpellChecker (version 0.5.4):
`pip install pyspellchecker `
This library is for spell correction.

* gTTs (version 2.1.1):
`pip install gTTS`
Text to audio library.

**Windows 10**

Best practice for Windows users is to download all the necessary libraries within a virtual programming environment such as Anaconda. This is to ensure that all the packages and libraries required to run the programs are all installed and accessible.

Navigate to https://www.anaconda.com/distribution/ and download the Python 3.7 version of Anaconda.

Once the software has been downloaded and the installation complete, type “Anaconda Prompt” in the Windows 10 Start menu and press enter. The anaconda prompt should open. Once inside the anaconda prompt, it is good practice to create a conda environment in which to install all the necessary libraries to run the project code. A conda environment is a directory that contains a specific collection of conda packages that you have installed.

For ease of installation, our recommended method for creating this conda environment and installing the necessary libraries is to use the file titled environment_droplet.yaml provided along with the project’s gitlab repository (https://gitlab.computing.dcu.ie/connok27/2020-ca326-kconnolly-signlanguagetranslator/blob/master/code/environment_droplet.yml). If the installation section of this document, disclosed above, has been correctly followed then this document should be already downloaded along with the remainder of the project files. This file can be found in the base directory of the project files alongside` parent.py`

Once this file has been downloaded, it can be used to build the conda environment needed to run the programs contained in the project. To do this, use the cd command in the conda prompt to move into the directory the downloaded project is contained in.

Once in the project directory, run the following command to build the conda environment and install the libraries / dependencies.

`conda env create -f environment_droplet.yml`

The environment can also be built from outside this directory by replacing the environment file name with the path to the file and the file name, as shown below

`conda env create -f <path_to_yml_file_including_file_name>`

where `<path_to_yaml_file_including_file_name>` is the absolute file path to the .yml file.

To use this environment it must be activated. To activate an environment in conda prompt, type the following line

`conda activate myenv`

where myenv is the name of the environment. If the environment was built using the .yml file then the name of the environment will be CA326-signlanguagetranslator.

The environment can also be built without using the provided .yml file, by manually installing the necessary libraries.

To create the environment, open up conda prompt and type the following

`conda create -n myenv python=3.6.10`
 
where myenv is the preferred name of the environment. The environment must then be activated to ensure that the libraries are installed in the correct location. The line to accomplish this is the same as the activation line above.

To install the necessary libraries type the following

`conda install keras opencv python==3.6.10`

This will install keras, opencv, tensorflow and numpy, along with other libraries. These are all libraries which are necessary to run `parent.py`. Installing all these libraries in the one line helps reduce dependency conflicts.

There are two more libraries that need to be installed into this environment in order to run the parent program. These are installed by way of pip, which is python’s package management system. These libraries are pyspellchecker, which is a spell checker python library used by `parent.py` to spell check the output of the models. The other library is gTTS, which is a python library which utilises Google’s text-to-speech API, and is used by `parent.py` to transform the text output of the models to audio. To install these libraries, type both the following lines into conda prompt respectively.

`pip install pyspellchecker`

`pip install gTTS`

All package versions for the Windows 10 installation are included in the environment_droplet.yml file, located in the gitlab code repository.


