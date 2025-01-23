
import tkinter as tk
# import tkcap
from PIL import Image, ImageDraw, ImageGrab
import numpy as np
import os

# Initialize the Tkinter window
root = tk.Tk()
root.title("Nepali Digit Recognition")

# Create a canvas for drawing
canvas_width, canvas_height = 320,320
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Initialize variables
draw_image = Image.new("L", (canvas_width, canvas_height), color="white")
draw = ImageDraw.Draw(draw_image)
save_counter = 0  # Counter for file naming
save_label = tk.StringVar()  # Variable to hold the current digit label

# Drawing function
def paint(event):
    x, y = event.x, event.y
    r = 8  # Radius of the pen stroke
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="black")

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill="white")

# Function to save the drawn image to a file
def save_image():
    global save_counter
    digit_label = save_label.get().strip()
    if not digit_label.isdigit():
        print("Please enter a valid digit label (0-9).")
        return
    
    # Create the dataset folder if it doesn't exist
    dataset_dir = "digit_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Capture canvas content
    canvas.update()
    x, y, w, h = canvas.winfo_rootx(), canvas.winfo_rooty(), canvas.winfo_width(), canvas.winfo_height()
    canvas_image = ImageGrab.grab((x+50, y+50, x+50+w, y+50+h))

    # canvas_image = tkcap.CAP(root)
    # Convert to grayscale
    digit_image = canvas_image.convert("L")
    digit_image.show()
    # # Remove extra whitespace
    digit_image_np = np.array(digit_image)
    non_empty_columns = np.where(digit_image_np.min(axis=0) < 255)[0]
    non_empty_rows = np.where(digit_image_np.min(axis=1) < 255)[0]
    if non_empty_columns.size > 0 and non_empty_rows.size > 0:
        crop_box = (min(non_empty_columns), min(non_empty_rows),
                    max(non_empty_columns), max(non_empty_rows))
        digit_image = digit_image.crop(crop_box)
    
    # Resize to 32x32 and save the image
    digit_image = digit_image.resize((32, 32), Image.LANCZOS)
    filename = os.path.join(dataset_dir, f"{digit_label}_{save_counter}.png")
    digit_image.save(filename)
    digit_image.show()
    save_counter += 1
    print(f"Image saved as {filename}")
    clear_canvas()

# Function to quit the application
def submit():
    print("Exiting the application.")
    root.quit()

# Bind the canvas with the paint function
canvas.bind("<B1-Motion>", paint)

# Add buttons and label entry
btn_frame = tk.Frame(root)
btn_frame.pack()

clear_btn = tk.Button(btn_frame, text="Clear", command=clear_canvas)
clear_btn.grid(row=0, column=0, padx=10, pady=10)

save_btn = tk.Button(btn_frame, text="Save", command=save_image)
save_btn.grid(row=0, column=1, padx=10, pady=10)

submit_btn = tk.Button(btn_frame, text="Submit", command=submit)
submit_btn.grid(row=0, column=2, padx=10, pady=10)

label_entry = tk.Entry(btn_frame, textvariable=save_label, width=5)
label_entry.grid(row=0, column=3, padx=10, pady=10)
label_entry.insert(0, "0")  # Default digit label

label_entry_label = tk.Label(btn_frame, text="Digit Label:")
label_entry_label.grid(row=0, column=4, padx=5, pady=5)

# Run the Tkinter loop
root.mainloop()