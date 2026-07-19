# 🐦 Bird Vocalization Detection System

## 📖 Overview

The Bird Vocalization Detection System is an end-to-end Machine Learning pipeline designed to classify and identify bird species based on their acoustic vocalizations.

The system processes raw audio files, extracts meaningful acoustic features, and uses machine learning models to predict the bird species. It also features a user-friendly web interface built with Django, allowing users to easily interact with the trained models and get real-time predictions.

## 📸 Web Interface

(Note to developer: Upload a screenshot of your running app to an assets/ folder in your repo and update the image path below!)

## 🚀 Key Features

- 🎵 Audio Feature Extraction: Automated processing of raw audio files into machine-readable datasets (via file_preparator.py).

- 🧠 Pre-trained Models: Comes with serialized, ready-to-use Support Vector Machine (svm.sav) and Naive Bayes (naive.sav) models.

- 🌐 Interactive Web App: A Django-based frontend where users can upload bird sounds and get real-time species predictions.

- 📓 Jupyter Notebook Integration: Includes complete, transparent workflows for model training (svm_model.ipynb) and evaluation (Predict.ipynb).

## 🤖 Models & Techniques

The system employs robust machine learning algorithms to classify audio signals based on extracted acoustic features.

##### 1. Audio Feature Extraction

Raw bird vocalizations are complex time-series data. Using the file_preparator.py script, the raw audio files in the /songs directory are processed to extract key machine-readable acoustic features. These features (often including MFCCs, Spectral Centroids, and Zero-Crossing Rates) quantify the texture, pitch, and spectral shape of the bird calls. The extracted tabular data is then saved into train.csv and test.csv.

##### 2. Support Vector Machine (SVM)

The primary predictive model used is a Support Vector Machine (svm.sav). SVMs are highly effective for audio classification tasks because they perform exceptionally well in high-dimensional feature spaces. The model maps the acoustic features and finds the optimal hyperplane that clearly separates the distinct bird species classes.

##### 3. Naive Bayes

A Naive Bayes classifier (naive.sav) is also included in the pipeline. This probabilistic model serves as a strong baseline, offering incredibly fast training and inference times by assuming conditional independence between the extracted acoustic audio features.

## ⚙️ How It Works

Here is a visual representation of the project's data flow, from raw audio to final web prediction:
```
graph TD;
    A[Raw Bird Audio /songs] -->|Processed by| B(file_preparator.py);
    B -->|Extracts Features| C[(train.csv & test.csv)];
    C -->|Trains| D[svm_model.ipynb];
    D -->|Exports| E[svm.sav & naive.sav];
    E --> F{Django Web App};
    G[User Uploads Audio File] --> F;
    F -->|Inference| H([Species Prediction Output]);
```

## 🗂️ Repository Structure
```
📦 Bird-Vocalization-Detection-System
 ┣ 📂 Django/                  # Django web application (bird_voice)
 ┣ 📂 songs/                   # Directory containing bird audio datasets
 ┣ 📜 birdsong_metadata.csv    # Metadata linking audio files to bird species
 ┣ 📜 train.csv                # Extracted feature dataset for training
 ┣ 📜 test.csv                 # Extracted feature dataset for testing
 ┣ 📜 file_preparator.py       # Python script for audio feature extraction
 ┣ 📜 svm_model.ipynb          # Model training notebook (SVM focus)
 ┣ 📜 Predict.ipynb            # Model evaluation & prediction notebook
 ┣ 📜 svm.sav                  # Saved Support Vector Machine model
 ┣ 📜 naive.sav                # Saved Naive Bayes model
 ┣ 📜 requirements.txt         # Project dependencies
 ┗ 📜 ReadMe.md                # Project documentation
```

## 🛠️ Setup & Installation

### Prerequisites

Make sure you have Python 3.12.5 installed on your machine.

### 1. Clone the repository

```cmd
git clone https://github.com/k-samarth/Bird-Vocalization-Detection-System.git
cd Bird-Vocalization-Detection-System
```

2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to prevent dependency conflicts.

# Create a virtual environment named 'env'
```cmd
python -m venv env

# Activate the virtual environment (Windows)
env\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source env/bin/activate
```

3. Install Dependencies
```cmd
pip install -r requirements.txt
```

## 💻 Usage Instructions

You can run this project in two different ways depending on your goals:

### Option A: Run the Web Application Directly

If you want to skip training and just test the web interface using the pre-trained models (svm.sav or naive.sav):

##### 1. Navigate to the Django project directory:
```cmd
cd Django\bird_voice
```

##### 2. Apply initial database migrations (required for the first run):
```cmd
python manage.py migrate
```

##### 3. Start the Django development server:
```cmd
python manage.py runserver
```

##### 4. Open your web browser and go to http://127.0.0.1:8000 to use the detection system.

### Option B: Retrain Models & Run the Web App

If you wish to preprocess the audio data, extract features, and train the models from scratch:

##### 1. Prepare the Data: Run the data preparation script to process the audio in the /songs folder and generate fresh train.csv and test.csv files.
```cmd
python file_preparator.py
```

##### 2. Train the SVM Model: Open svm_model.ipynb in Jupyter Notebook/Lab and run all the cells to train and save the new model.

##### 3. Test Predictions: Open Predict.ipynb and run all cells to verify model accuracy and inference.

##### 4. Launch the Web App: Once training is verified, navigate to the web directory, migrate, and start the server:
```cmd
cd Django\bird_voice
python manage.py migrate
python manage.py runserver
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to check out the [Issues page](https://github.com/k-samarth/Bird-Vocalization-Detection-System/issues) if you want to contribute to the code.

## 📜 License
This project is open-source and available under the MIT License.

## 📬 Contact
Samarth Kamath - [GitHub Profile](https://github.com/k-samarth)
