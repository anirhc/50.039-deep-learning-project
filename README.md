# 50.039 Deep Learning Project 
# Predicting Mental Health Treatment Needs
## Group 6

### Members
-   Swastik Majumdar 1005802
-   Aditya Vishwanath  1004281
-   Harikrishnan Chalapathy Anirudh 1005501







###  Please refer to the report for a more detailed breakdown of the project.

### Introduction

Our team decided to tackle the issue of mental health and well-being in the workplace, recognizing that it is a significant aspect of healthcare that directly correlates to one's quality of life and overall well-being. We believe that mental health is equally, if not more, important than other facets of physical health.

We understand that the mental well-being of employees plays a critical role in their productivity, engagement, and job satisfaction. However, mental health issues are often stigmatized and not openly addressed in the workplace, resulting in many individuals not receiving the help they need, and employers lacking a robust way to assess the mental well-being of their employees and provide necessary support.

To address this issue, we developed a deep learning model that can predict whether an employee might need mental health support. This model can enable companies to monitor the mental health levels of their employees and provide them with the necessary support, such as counseling, therapy, or more advanced forms of mental health assistance. By leveraging technology and data-driven approaches, we aim to create a tool that can contribute to improving the mental well-being of employees in the workplace, ultimately benefiting their overall health and well-being, as well as the productivity and success of the organization as a whole.

### Dataset
The dataset can be downloaded using the following [link.](https://www.kaggle.com/datasets/ron2112/mental-health-data/download?datasetVersionNumber=1)

The mental health dataset used for this project was collected through a survey conducted in various countries around the world. The dataset contains various features related to mental health, including demographic
information, mental health conditions, and various factors that may contribute to mental health issues.

### Setting up the Python Environment

1. Install Python 3 if you haven't already.

2. Open a terminal or command prompt in the project directory.

3. Create a virtual environment using `venv`:
    ```bash
    python3 -m venv myenv
    ```

4. Activate the virtual environment:
    - On Windows:
    ```bash
    myenv\Scripts\activate.bat
    ```
    - On Unix or Linux:
    ```bash
    source myenv/bin/activate
    ```

5. Install the required packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    This may take a while.

### Training the model

1. Make sure the virtual environment is activated.

2. Train the model:
    ```bash
    python train.py
    ```
    This may take a while.

3. Your model should now be training with the required packages installed.

### Deactivating the Virtual Environment

To deactivate the virtual environment, simply run the following command in the terminal or command prompt:
```bash
deactivate
```
### Files
- `Notebook.ipynb`: Jupyter Notebook containing documented code for this project. This includes
    -   Packages required
    -   Data preprocessing
    -   Data Visualization
    -   Train Test Validation Split
    -   Model Architecture
    -   Accuracy and Loss curves
    -   Evaluation
    -   Loading the trained model
    -   Comparison of model performance against some state-of-the-art ones

-  `data_preprocessing.py`: Contains steps used for cleaning, transforming, and preparing the data for model training.

-  `model.py`: This file contains the implementation of the deep learning model used in this project. It includes the architecture of the neural network, as well as functions used for training and testing the model.

-  `train.py`: This file is used for training the deep learning model. It includes initializing the model, training with specified hyperparameters, and saving the trained model weights for later use.

-  `outputs`: This folder contains various plots or visualizations related to the project.

-  `weights`: This folder contains the saved model parameter values, which can be loaded later.

- `convert_to_onnx.py`: This file is used to convert the saved model to the ONNX format, which is a standard format for representing deep learning models that can be used across different platforms and frameworks.

- `app`: This folder is for the web application. Includes server-side code written in Node.js for hosting and serving the trained model. Also includes frontend code for creating a user interface.

- `requirements.txt`: This file lists the required packages or dependencies for the project.

### Web application: WellCheck

A video demonstration:

https://user-images.githubusercontent.com/73923291/231261638-51e12c2b-9cae-46f4-af1b-d6413dd90b75.mp4

Follow the instructions below to run the app.

#### Prerequisites
- Node.js and npm (Node Package Manager) is installed on your machine.

#### Configuration
- Download the `.env` file from [this link](https://drive.google.com/file/d/1bSEvlCB9L0h22w9dKrSIfAyhnkTt4rdy/view?usp=sharing) 
- Create a `.env` file in the `app` folder. Copy the contents of the downloaded `env` file to `.env`.
- `.env` should now contain a username and password. The final path for `.env` should be `app/.env`.

#### Installation
- Navigate to the `app` folder in the project directory using the command line or terminal: `cd app`.
- Install the required dependencies by running the following command: `npm install`.

#### Running the App
1. Once the dependencies are installed, you can start the app by running the following command: `npm start` or `node server.js`.
2. The app will now be running and hosted on `http://localhost:3000` in your web browser.

#### Terminate
- To stop the app, simply terminate the Node.js process by pressing `Ctrl + C` in the command line or terminal.



