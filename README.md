# OCR GUI Application

This repository contains a comprehensive Optical Character Recognition (OCR) GUI application developed using Python. The project leverages PyQt5 for the graphical user interface, Keras and TensorFlow for a neural network model, scikit-learn for a random forest model, and OpenCV for image processing. The application allows users to draw characters on a canvas, predict them using pre-trained machine learning models, and display the results. The project includes training scripts, model files, and supporting documentation, making it a complete solution for handwritten character recognition.

## Project Overview
The OCR GUI application is designed to recognize handwritten characters based on the EMNIST (Extended MNIST) dataset. It provides an interactive interface where users can draw characters, select between two pre-trained models (neural network and random forest), and receive real-time predictions. The project is structured to support both the application runtime and the training process, with all necessary files included or referenced for ease of use.

## Project Structure
- `archive.rar`: A compressed archive containing additional project files or backups.
- `guiy_FINALipynb.ipynb`: The main application script, converted to a Jupyter notebook, containing the GUI implementation and prediction logic using PyQt5.
- `ocr-full-using-anns.ipynb`: A Jupyter notebook script for training and evaluating a neural network model using Keras and TensorFlow, saved as `ocr-full.h5`.
- `ocr-full.h5`: A pre-trained neural network model file trained on the EMNIST dataset.
- `rf.ipynb`: A Jupyter notebook script for training and evaluating a random forest model using scikit-learn, intended to generate `rf_model.pkl` (note: `rf_model.pkl` is not included and must be generated).
- `emnist-balanced-mapping.txt`: A mapping file that converts EMNIST label indices to ASCII characters.

## Features
- **Interactive Canvas**: Users can draw characters on a 400x400 pixel canvas using the mouse.
- **Model Selection**: A dropdown menu allows switching between the neural network model (`ocr-full.h5`) and the random forest model (`rf_model.pkl`).
- **Real-Time Prediction**: Pressing the Enter key triggers the prediction of drawn characters, with results displayed in a text box.
- **Clear Canvas**: A button to reset the canvas for new drawings.
- **ASCII Mapping**: Predictions are mapped to readable characters using the EMNIST mapping file.

## Prerequisites
To run or extend this project, the following are required:
- **Python 3.6 or higher**
- **Required Libraries**:
  - `PyQt5` (for GUI)
  - `numpy` (for numerical operations)
  - `opencv-python` (for image processing)
  - `tensorflow` (for neural network model)
  - `scikit-learn` (for random forest model)
  - `matplotlib` (for visualization during training)
- **Dataset**: EMNIST balanced train and test CSV files (`emnist-balanced-train.csv`, `emnist-balanced-test.csv`) for training (not included).
- **Mapping File**: `emnist-balanced-mapping.txt` must be present in the project directory.

Install dependencies using:
```bash
pip install PyQt5 numpy opencv-python tensorflow scikit-learn matplotlib
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ocr-gui-application.git
   cd ocr-gui-application
   ```
2. Extract `archive.rar` to access additional files or backups.
3. Ensure `emnist-balanced-mapping.txt` is in the project directory. Download the EMNIST dataset if training from scratch.
4. Launch the application:
   ```bash
   jupyter notebook guiy_FINALipynb.ipynb
   ```

## Usage
1. Open `guiy_FINALipynb.ipynb` in Jupyter Notebook.
2. Select a model from the dropdown menu (`ocr-full.h5` or `rf_model.pkl`).
3. Draw characters on the canvas using the mouse.
4. Press the **Enter** key to predict the drawn characters; the results will appear in the text box below the canvas.
5. Click the **Clear Canvas** button to start a new drawing.

## Model Training
### Neural Network Model (`ocr-full.h5`)
- **Script**: `ocr-full-using-anns.ipynb`
- **Framework**: TensorFlow/Keras
- **Architecture**: A sequential model with three dense layers (512, 256, and 47 units) using ReLU and softmax activations.
- **Training**: Trained on the EMNIST dataset with 100 epochs and a batch size of 32, normalized to [0, 1].
- **Process**: Run the notebook to train and save the model as `ocr-full.h5`. The script includes visualization of accuracy and loss, model evaluation metrics (accuracy, precision, recall, F1 score), and a confusion matrix.

### Random Forest Model (`rf_model.pkl`)
- **Script**: `rf.ipynb`
- **Framework**: scikit-learn
- **Algorithm**: Random Forest Classifier with `log_loss` criterion.
- **Training**: Trained on the EMNIST dataset with default parameters.
- **Process**: Run the notebook to train and save the model as `rf_model.pkl`. The script provides accuracy scores for training and test sets.

## Image Processing
The application uses OpenCV to process drawn images:
- Converts the canvas image to grayscale.
- Applies thresholding to create a binary image.
- Detects contours to isolate individual characters.
- Resizes and normalizes images to 28x28 pixels for model input, with additional transformations (bitwise not, flipping, rotation) to match EMNIST preprocessing.
