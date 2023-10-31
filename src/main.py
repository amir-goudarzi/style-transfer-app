import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import Style_Transfer  # Import the Python script converted from the Jupyter Notebook

# Function to apply style transfer
def apply_style_transfer():
    # Get the file paths of the style and content images
    style_image_path = style_image_var.get()
    content_image_path = content_image_var.get()

    # Call the style transfer function from Style_Transfer.py
    result_image = Style_Transfer.style_transfer_function(style_image_path, content_image_path)

    # Display the result image in the GUI
    result_image = Image.open(result_image)
    result_image.thumbnail((400, 400))  # Resize for display
    result_image = ImageTk.PhotoImage(result_image)
    result_label.config(image=result_image)
    result_label.image = result_image

# Function to open a file dialog and get the image file path
def open_image(file_var):
    file_path = filedialog.askopenfilename()
    file_var.set(file_path)

# Create the main application window
app = tk.Tk()
app.title("Style Transfer App")

# Variables to store image file paths
style_image_var = tk.StringVar()
content_image_var = tk.StringVar()

# Create labels and buttons
style_label = tk.Label(app, text="Style Image:")
content_label = tk.Label(app, text="Content Image")
style_image_button = tk.Button(app, text="Select", command=lambda: open_image(style_image_var))
content_image_button = tk.Button(app, text="Select", command=lambda: open_image(content_image_var))
apply_button = tk.Button(app, text="Apply Style Transfer", command=apply_style_transfer)
result_label = tk.Label(app)

# Arrange widgets on the GUI
style_label.grid(row=0, column=0)
style_image_button.grid(row=0, column=1)
content_label.grid(row=1, column=0)
content_image_button.grid(row=1, column=1)
apply_button.grid(row=2, columnspan=2)
result_label.grid(row=3, columnspan=2)

# Start the GUI application
app.mainloop()
