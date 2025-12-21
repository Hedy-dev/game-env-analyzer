import tkinter as tk
import threading
import time

import numpy as np
from mss import mss
from PIL import Image

def model(x: np.ndarray):
    """
    x: numpy array (H, W, 3)
    """
    h, w, _ = x.shape
    return f"hello ({w}x{h})"

class Overlay(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Overlay")
        self.geometry("320x120+50+50")

        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.85)
        self.overrideredirect(True)

        self.label = tk.Label(
            self,
            text="Waiting...",
            fg="white",
            bg="black",
            font=("Arial", 14)
        )
        self.label.pack(padx=20, pady=20)

        self.bind("<ButtonPress-1>", self.start_move)
        self.bind("<B1-Motion>", self.do_move)

        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()

    def start_move(self, e):
        self._x = e.x
        self._y = e.y

    def do_move(self, e):
        x = self.winfo_x() + e.x - self._x
        y = self.winfo_y() + e.y - self._y
        self.geometry(f"+{x}+{y}")

    def capture_loop(self):
        with mss() as sct:
            monitor = sct.monitors[1] 

            while self.running:
                sct_img = sct.grab(monitor)

                frame = np.array(sct_img)[:, :, :3]
                frame = frame[..., ::-1]

                result = model(frame)

                self.after(0, self.label.config,
                           {"text": f"Model: {result}"})

                time.sleep(0.5)

    def on_close(self):
        self.running = False
        self.destroy()


if __name__ == "__main__":
    app = Overlay()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

