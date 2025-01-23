import tkinter as tk
from PIL import Image, ImageDraw, ImageGrab
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adam
from PIL import Image
from torchvision.transforms import v2


root = tk.Tk()
root.title("Nepali Digit Recognition")

canvas_width, canvas_height = 320, 320
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

draw_image = Image.new("L", (canvas_width, canvas_height), color="white")
draw = ImageDraw.Draw(draw_image)

# to draw on the canvas with the mouse
def paint(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="black")

# to clear the canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill="white")
    result_label.config(text="Predicted Digit: None")


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1)

        self.fc1 = nn.Linear(576, 300)
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 6 * 6)

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
    
# to load the pre-trained model
with open('final_model.sav', 'rb') as f:
    model = pickle.load(f)

# Set model to evaluation mode
model.eval()

# to predict the digit based on the drawn image
def predict_digit(image):
    # Convert the image to a tensor and normalize it
    transform = v2.Compose([
        # v2.RandomResizedCrop(224),
        # v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        v2.Grayscale(num_output_channels=1),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0)

    # Pass the image through the model
    with torch.no_grad():
        output = model(image)
        
    _, predicted = torch.max(output, 1)
    
    return predicted.item()

# to capture the drawn image, preprocess it and make a prediction
def submit_image():
    canvas.update()  # Ensure the canvas is up to date before capturing the image

    x, y, w, h = canvas.winfo_rootx(), canvas.winfo_rooty(), canvas.winfo_width(), canvas.winfo_height()
    canvas_image = ImageGrab.grab((x, y, x+w, y+h))

    digit_image = canvas_image.convert("L")
    digit_image.show()

    digit_image_np = np.array(digit_image)
    non_empty_columns = np.where(digit_image_np.min(axis=0) < 255)[0]
    non_empty_rows = np.where(digit_image_np.min(axis=1) < 255)[0]
 
    if non_empty_columns.size > 0 and non_empty_rows.size > 0:
        crop_box = (min(non_empty_columns), min(non_empty_rows),
                    max(non_empty_columns), max(non_empty_rows))
        digit_image = digit_image.crop(crop_box)

    digit_image = digit_image.resize((32, 32), Image.LANCZOS)


    predicted_digit = predict_digit(digit_image)
    return predicted_digit

# to handle the "Submit" button click and show the predicted result
def display_prediction():
    predicted_digit = submit_image()  # Get the predicted digit from the drawing
    result_label.config(text=f"Predicted Digit: {predicted_digit}")  # Display the prediction

# to quit the application
def quit_app():
    root.quit()

canvas.bind("<B1-Motion>", paint)


btn_frame = tk.Frame(root)
btn_frame.pack()

# to clear the canvas
clear_btn = tk.Button(btn_frame, text="Clear", command=clear_canvas)
clear_btn.grid(row=0, column=0, padx=10, pady=10)

# to submit the drawn image for prediction
submit_btn = tk.Button(btn_frame, text="Submit", command=display_prediction)
submit_btn.grid(row=0, column=1, padx=10, pady=10)

# to display the predicted digit
result_label = tk.Label(root, text="Predicted Digit: None", font=("Helvetica", 16))
result_label.pack(pady=10)

# to exit the application
quit_btn = tk.Button(root, text="Quit", command=quit_app)
quit_btn.pack(pady=10)

root.mainloop()
