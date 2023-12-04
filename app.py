import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        self.save_button = tk.Button(
            root, text="Save", command=self.save_data, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(root, text="Reset", command=self.init_state)
        self.reset_button.pack(side=tk.LEFT)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion)

        self.rect = None
        self.start_x = None
        self.start_y = None
        self.vanishing_point = None
        self.image = None
        self.load_image()

    def init_state(self):
        if self.rect:
            self.canvas.delete(self.rect)
        self.canvas.delete("line")
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.vanishing_point = None
        self.save_button.config(state=tk.DISABLED)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        self.image = Image.open(file_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # fit window to image
        self.root.geometry(f"{self.image.width}x{self.image.height}")

    def save_data(self):
        if self.rect and self.vanishing_point:
            filename = filedialog.asksaveasfilename(defaultextension=".txt")
            if filename:
                with open(filename, "w") as file:
                    coords = self.canvas.coords(self.rect)
                    file.write(
                        f"{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}, {self.vanishing_point[0]}, {self.vanishing_point[1]}\n"
                    )

    def on_press(self, event):
        if self.rect and not self.vanishing_point:
            self.vanishing_point = (event.x, event.y)
            self.save_button.config(state=tk.NORMAL)
            print(f"Vanishing Point: {self.vanishing_point}")
            return
        if self.rect and self.vanishing_point:
            self.init_state()
            return
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red"
        )

    def on_drag(self, event):
        self.end_x, self.end_y = event.x, event.y
        self.canvas.coords(
            self.rect, self.start_x, self.start_y, self.end_x, self.end_y
        )

    def on_release(self, event):
        print("Released")

    def on_motion(self, event):
        if self.rect and not self.vanishing_point:
            self.canvas.delete("line")
            coords = self.canvas.coords(self.rect)
            # If vanishing point is not inside the rectangle, don't draw lines
            if (
                event.x < coords[0]
                or event.x > coords[2]
                or event.y < coords[1]
                or event.y > coords[3]
            ):
                return
            if coords:
                for point in [
                    (coords[0], coords[1]),
                    (coords[2], coords[1]),
                    (coords[2], coords[3]),
                    (coords[0], coords[3]),
                ]:
                    ray_origin = np.array([event.x, event.y])
                    ray_direction = np.array([point[0], point[1]]) - ray_origin
                    end_point = ray_origin + ray_direction * 10000
                    self.canvas.create_line(
                        event.x,
                        event.y,
                        end_point[0],
                        end_point[1],
                        fill="blue",
                        tags="line",
                    )
            print(f"Rectangle: {coords}, Vanishing Point: ({event.x}, {event.y})")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()
