# **School of Computing**
# **CA326 Year 3 Project Proposal Form**

**Project Title:**  Building a Neural Network to recognise and translate sign language to audio in real time.

**Student 1:**  **Name** William John O’Hanlon         **ID Number** 17477494  
**Student 2:**  **Name** Killian Connolly              **ID Number** 17303116  
**Staff Member Consulted:** Alistair Sutherland  

### **Description**

We wish to train neural networks, using American Sign Language (ASL) and Irish Sign Language (ISL) data sets, to identify and translate sign language into audio in real time, using a camera as a means of obtaining live feed. We are using an ISL dataset provided to us by Alistair Sutherland and also a dataset on ASL obtained from kaggle.com (publicly available datasets). We plan on training our neural network ISL first, then training the neural network ASL since there are similarities in both languages and datasets available on each. The alphabet of both languages will be the first datasets to train the algorithm with, and after this, the algorithm will be trained to recognise words as well. Words in sign language are more complicated, because there are moving parts like arms and facial gestures, but this is something we wish to implement in the latter stages of the project. The output of the neural network will be a text representation of the sign performed by the user, fed to the algorithm by means of a camera. Once the algorithm is trained to identify sign language and produce text, this text will then be fed to another algorithm which will convert it into audio and play it back to the user in real time.
The objective of this project in essence is to make a program which will allow for free flowing communication between a person without the ability to speech and a person with the ability to speak, but without the ability to understand or communicate in sign language. 


## **Division of Work**

**William:**  
Training the neural network to recognise the ISL alphabet.
Training the network to recognise specific words and phrases in ISL.
Taking in live video from the webcam and feeding it to the neural network i.e. image processing and computer vision aspect.

**Killian:**  
Training the neural network to recognise the ASL alphabet.
Training the network to recognise specific words and phrases in ASL.
Converting the text produced by the neural network into audio in real time.

We think we have divided the work fairly, however some components of the project are critical and require understanding from both of us. Therefore the implementation  of the neural network will require both us to program together.  

### **Programming Languages:**

Python


### **Programming Tools:**

Dataset provided by Alistair Sutherland of Irish Sign Language (ISL). 
Publically available datasets downloadable on Kaggle containing American Sign Language (ASL) images.


### **Learning Challenges:**

We will have to learn how to code and implement neural networks / machine learning algorithms using open source libraries such as PyTorch and TensorFlow. 
We will also have to learn image processing and computer vision using open source software, and how to implement these techniques in real time.
Another challenge we will have to learn to overcome is converting text into audio in real time.
As well as that we will have to familiarise ourselves with both Irish Sign Language (ISL) and American Sign Language (ASL), as neither of us are literate in these languages.

### **Hardware / Software Platforms:**


We will develop our project in Windows 10.
We will also use our webcams from our laptops, and an external webcam for the lab machines, to provide a live camera feed to the neural network.
