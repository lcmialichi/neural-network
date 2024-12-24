import tkinter as tk
from tkinter import filedialog, messagebox
from neural_network.board.drawable import Drawable
from PIL import Image, ImageTk
from typing import Tuple
import numpy as np
import os

class FileInput(Drawable):
    def __init__(self, title: str, img_resize: Tuple[int, int] = (50,50)):
        root = tk.Tk()
        root.title(title)

        self.image_size = img_resize
        self.root: tk.Tk = root
        self.frame = tk.Frame(root)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Selecione uma Imagem",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg"), ("Todos os Arquivos", "*.*")]
        )
        if file_path:
            self.selected_file_path = file_path
            self.file_path_label.config(text=os.path.basename(file_path), fg="black")
            self.display_image(file_path)
            self.send_button.config(state=tk.NORMAL)

    def display_image(self, file_path):
        try:
            image = Image.open(file_path)
            image.thumbnail((200, 200))
            img_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar a imagem: {e}")

    def send_file(self):
        if self.selected_file_path:
            try:
                label = self.get_label(self.get_handler()(np.array([self.get_image_as_matrix()])))
                messagebox.showinfo("Analise completa", f"{label}!")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao executar a ação: {e}")

    def get_image_as_matrix(self):
        image = Image.open(self.selected_file_path).convert('RGB')
        image = image.resize(self.image_size)
        return (np.transpose(image, (2, 0, 1)) / 255 - 0.5) / 0.5
    
    def draw(self):
        self.frame.pack(pady=20)

        self.upload_button = tk.Button(self.frame, text="Escolher Imagem", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.file_path_label = tk.Label(self.frame, text="Nenhum arquivo selecionado", fg="gray")
        self.file_path_label.pack(pady=5)

        self.image_label = tk.Label(self.frame)
        self.image_label.pack(pady=10)

        self.send_button = tk.Button(self.frame, text="Enviar", command=self.send_file, state=tk.DISABLED)
        self.send_button.pack(pady=10)

        self.selected_file_path = None

    def loop(self):
        self.root.mainloop()

