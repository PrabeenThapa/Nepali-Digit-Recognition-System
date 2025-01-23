# Nepali-Digit-Recognition-System

This project implements a Nepali Digit Recognition System using a convolutional neural network (CNN). The system is designed to allow users to draw Nepali digits on a canvas, which are then recognized and classified by the trained model.

## Features

- Interactive Canvas: Draw Nepali digits directly on a graphical canvas.
- Digit Prediction: The system predicts the digit based on the drawn image.
- Pre-trained Model: A pre-trained CNN model is included for accurate predictions.
- Simple UI: Built with Tkinter, ensuring a clean and user-friendly interface.
- Own Dataset: Custom dataset using collect_digit.py is made.

## Requirements

To run this project, make sure you have the following installed:
- Python 3.7 or higher
- Required Python packages: 
- torch
- torchvision
- Pillow
- numpy
- pickle

Install the dependencies using:
- pip install torch torchvision Pillow numpy 

## How It Works


- Launch the application, which opens a canvas for drawing.
- Use your mouse to draw a digit on the canvas.
- Click the "Submit" button to predict the digit.
- The predicted digit is displayed on the screen.
- You can clear the canvas to redraw or exit the application at any time.

## Project Structure

- app.py: Main application file. It contains the code for the GUI, digit drawing, and model integration.
- final_model.sav: Pre-trained model used for digit recognition.

## Usage Instructions

- Run the application: python app.py 

- Draw a Nepali digit on the canvas provided.

- Press the "Submit" button to predict the digit.

- Clear the canvas with the "Clear" button or exit with the "Quit" button.

## Model Architecture
The CNN model has the following architecture:
- Convolutional Layers: Extract features from the input image.
- Fully Connected Layers: Classify the digit into one of the ten possible categories (0–9).
- Activation Functions: ReLU and softmax functions are used to ensure non-linearity and probabilistic outputs.

## Preprocessing Steps
- The drawn digit is converted into grayscale.
- The image is resized to 32x32 pixels for compatibility with the CNN model.
- Pixel values are normalized before being passed into the model.

## Limitations
The recognition accuracy depends on the clarity and quality of the drawn digit.


## Future Improvements
- Enhance the UI for better usability.
- Add functionality to train the model on new data.

## Contributing

Feel free to contribute to this project by submitting pull requests or suggesting features and improvements.

