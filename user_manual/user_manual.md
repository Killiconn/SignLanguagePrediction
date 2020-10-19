# USER MANUAL

## Table of Contents:
1 Installation

2.1. Preliminary
2.1.1 Recording the video
2.1.2 Letters accepted

3.1 Operating Systems

4.1 Downloading Libraries
4.1.1 Linux Ubuntu 18.04
4.1.2 Windows 10

5.1 Running the program
5.1.1 Linux Ubuntu 18.04
5.1.2 Windows 10
5.1.3 Google Colab

## 1 Installation
Login to gitlab.computing.dcu.ie
Go to the following link: https://gitlab.computing.dcu.ie/connok27/2020-ca326-kconnolly-signlanguagetranslator
Go into the second repository, the ‘code’ repository.

Press the download button as shown above.
Extract the file into the appropriate directory.

When in the file, go to CNN/models/final_models/model_location.md
Follow the link in this file and download the models located in this link. The models are too large for us to place on our gitlab repository.

## 2.1 Preliminary
### 2.1.1 Recording the video
Record a video of a person performing Irish Sign Language. The placement of the arm should be similar to below. The hand should be centre to the video. The video can be of any quality above 480p. 

### 2.1.2 Letters accepted
Our program can detect the following Irish sign language letters:

![](https://i.imgur.com/vseEuhH.png)
![](https://i.imgur.com/WAouXHv.png)


These are alphabetic letters, respectfully, ABCDEFGHIJKLMNOPQRSTUVWXY.
The letters J, X, and Z are not included since they are motions.
owever it is the most popular operating system.


## 3.1 How to download the libraries:

#### 3.1.1 Linux - Ubuntu

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

#### 3.1.2 Windows 10

Best practice for Windows users is to download all the necessary libraries within a virtual programming environment such as Anaconda. This is to ensure that all the packages and libraries required to run the programs are all installed and accessible.

Navigate to https://www.anaconda.com/distribution/ and download the Python 3.7 version of Anaconda.

![](https://i.imgur.com/2LhJCXI.png)

Once the software has been downloaded and the installation complete, type “Anaconda Prompt” in the Windows 10 Start menu and press enter.

![](https://i.imgur.com/U1MkkyM.png)

The anaconda prompt should open. Once inside the anaconda prompt, it is good practice to create a conda environment in which to install all the necessary libraries to run the project code. A conda environment is a directory that contains a specific collection of conda packages that you have installed.

For ease of installation, our recommended method for creating this conda environment and installing the necessary libraries is to use the file titled environment_droplet.yaml provided along with the project’s gitlab repository (https://gitlab.computing.dcu.ie/connok27/2020-ca326-kconnolly-signlanguagetranslator/blob/master/code/environment_droplet.yml). If the installation section of this document, disclosed above, has been correctly followed then this document should be already downloaded along with the remainder of the project files. This file can be found in the base directory of the project files alongside `parent.py`.

Once this file has been downloaded, it can be used to build the conda environment needed to run the programs contained in the project. To do this, use the cd command in the conda prompt to move into the directory the downloaded project is contained in. For more information on how to use the cd command to change directories see below in the “How to Run the Program” section.

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

This will install keras, opencv, tensorflow and numpy, along with other libraries. These are all libraries which are necessary to run `parent.py` . Installing all these libraries in the one line helps reduce dependency conflicts.

There are two more libraries that need to be installed into this environment in order to run the parent program. These are installed by way of pip, which is python’s package management system. These libraries are pyspellchecker, which is a spell checker python library used by `parent.py` to spell check the output of the models. The other library is gTTS, which is a python library which utilises Google’s text-to-speech API, and is used by `parent.py` to transform the text output of the models to audio. To install these libraries, type both the following lines into conda prompt respectively.

`pip install pyspellchecker`

`pip install gTTS`


## 4 How to Run the Program

#### 4.1.1 Ubuntu
Open the command prompt by searching command prompt or pressing Ctrl+Alt+T.
Change the directory into the directory that contains the parent program by means of `cd` `<name_of_directory>`
Type the following command to run the program.
`python parent.py <absolute_file_path_to_video>`

#### 4.1.2 Windows 10

Open conda prompt.
Activate the environment which contains the installed libraries.
Change the directory to the directory the project is contained in, by using the cd command.
Type the following command to run the program.
`python parent.py <absolute_file_path_to_video>`

#### 4.1.3 Google Colab

Our programs require quite a lot of computing power, so if you feel your computer architecture is not capable of running our program, you can utilise Google’s open source software.
Google Colab is a free cloud service that offers the use of google’s high powered GPUs.
To use this.
- Download the files as usual and transfer the files into your google drive account into any file of your choosing.
- Mount your drive.
- Save the video that you would like to translate into your google drive account.
- Change line 82 in the code from `VIDEO_LOCATION = sys.argv[1]` to the location of the video in your Google Drive.
- Run the cell.

Note : The code will not output the resulting text as audio, but will print the text result after the program has finished running.
