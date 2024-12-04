import tkinter as tk
from PIL import Image
import numpy as np
from typing import Callable, Union

class DrawBoard:
    def __init__(self, root: tk.Tk, size: int = 280, line_weight: int = 7):
        self.root = root
        self.__canvas_size = size
        self.__line_weight = line_weight

    def set_handler(self, handler: Callable) -> None:
        self.handler = handler

    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="white", width=14, capstyle=tk.ROUND, smooth=True)
        else:
            self.canvas.create_oval(
                event.x - self.__line_weight, 
                event.y - self.__line_weight, 
                event.x + self.__line_weight, 
                event.y + self.__line_weight, 
                fill="white", 
                outline=""
            )

        self.update_image(event.x, event.y)
        self.last_x, self.last_y = event.x, event.y

    def update_image(self, x, y):
        draw = self.image.load()
        for i in range(max(0, x - self.__line_weight), min(self.__canvas_size, x + self.__line_weight)):
            for j in range(max(0, y - self.__line_weight), min(self.__canvas_size, y + self.__line_weight)):
                draw[i, j] = 255

    def reset_cursor(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.__canvas_size, self.__canvas_size), 0)
        self.feedback_label.config(text="Desenhe um número e pressione Enter para prever")

    def predict_digit(self, event=None):
        self.feedback_label.config(text="Previsão em andamento...")
        prediction = self.handler(self.get_image_as_vector())
        self.feedback_label.config(text=f"Previsão: {prediction}")

    def get_image_as_vector(self):
        resized_image = self.image.resize((28, 28), Image.LANCZOS)
        image_array = np.array(resized_image) / 255
        print(image_array)
        return image_array.flatten().reshape(1, -1)
    
    def draw(self):
        self.canvas = tk.Canvas(self.root, width=self.__canvas_size, height=self.__canvas_size, bg="black")
        self.canvas.pack(padx=5, pady=5)

        self.image = Image.new("L", (self.__canvas_size, self.__canvas_size), 0)
        self.last_x, self.last_y = None, None

        self.feedback_label = tk.Label(self.root, text="Desenhe um número e pressione Enter para prever")
        self.feedback_label.pack(pady=5)

        tk.Button(self.root, text="Enviar", command=self.predict_digit).pack(pady=5)
        tk.Button(self.root, text="Limpar", command=self.clear_canvas).pack(pady=5)
        self.root.bind("<Return>", self.predict_digit)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_cursor)
