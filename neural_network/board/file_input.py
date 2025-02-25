import tkinter as tk
from tkinter import filedialog, messagebox
from neural_network.board.drawable import Drawable
from PIL import Image, ImageTk
from typing import Tuple
from neural_network.gcpu import driver
import os

class FileInput(Drawable):
    def __init__(self, title: str, img_resize: Tuple[int, int] = (50,50)):
        self.title = title
        self.image_size = img_resize        
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose an Image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")]
        )
        if file_path:
            self.selected_file_path = file_path
            self.file_path_label.config(text=os.path.basename(file_path), fg="black")
            self.display_image(file_path)
            self.send_button.config(state=tk.NORMAL)

    def display_image(self, file_path):
        try:
            image = Image.open(file_path)
            image.thumbnail((500, 500))
            img_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
        except Exception as e:
            messagebox.showerror("Erro", f"unable to load image: {e}")

    def send_file(self):
        if self.selected_file_path:
            try:
                handler = self.get_handler()
                result = handler(driver.gcpu.array([self.get_image_as_matrix()]) / 255.00)
                label = self.get_label(result)
                messagebox.showinfo("Analise completa", f"{label}!")
            except Exception as e:
                messagebox.showerror("Erro", f"Execution error: {e}")

    def get_image_as_matrix(self):
        image = Image.open(self.selected_file_path).convert('RGB')
        image = image.resize(self.image_size)
        return image
    
    def draw(self):
        self.root = tk.Tk()
        self.root.title(self.title)
        self.frame = tk.Frame(self.root)
        self.frame.pack(pady=20)

        self.upload_button = tk.Button(self.frame, text="Choose image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.file_path_label = tk.Label(self.frame, text="No file selected", fg="gray")
        self.file_path_label.pack(pady=5)

        self.image_label = tk.Label(self.frame)
        self.image_label.pack(pady=10)

        self.send_button = tk.Button(self.frame, text="Send", command=self.send_file, state=tk.DISABLED)
        self.send_button.pack(pady=10)

        self.selected_file_path = None

    def loop(self):
        self.root.mainloop()

