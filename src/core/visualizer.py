import time
import numpy as np
import simpleaudio as sa
from scipy.signal import spectrogram

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import sys

class SpectrogramWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Matplotlib setup
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        self.ax.set_xlabel('Time [s]', color='white')
        self.ax.set_ylabel('Frequency [Hz]', color='white')
        self.ax.set_title('Dotted Spectrogram Animation', color='white')

        self.scatter = self.ax.scatter([], [], c=[], cmap='cool', edgecolors='none', s=8)

        # Canvas for Qt
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()

        # Layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        self.ani = None  # Store the animation object

    def update_spectrogram(self, audio_data, sample_rate):
        """Updates the spectrogram plot with new audio data."""
        # Compute spectrogram
        f, t, Sxx = spectrogram(audio_data, fs=sample_rate, nperseg=1024)
        Sxx = 10 * np.log10(Sxx + 1e-10)  # Convert to dB scale

        def update(frame):
            self.ax.clear()
            self.ax.set_facecolor('black')
            self.ax.set_xlabel('Time [s]', color='white')
            self.ax.set_ylabel('Frequency [Hz]', color='white')
            self.ax.set_title('Dotted Spectrogram Animation', color='white')

            x = np.tile(t[:frame], len(f))
            y = np.repeat(f, frame)
            c = Sxx[:, :frame].flatten()
            scatter = self.ax.scatter(x, y, c=c, cmap='cool',
                                     edgecolors='none', s=8, alpha=0.7)
            return scatter,  # update needs to return a sequence (or collection) of artists

        if self.ani:
            self.ani._stop()

        self.ani = animation.FuncAnimation(
            self.fig, update, frames=len(t), interval=50, blit=False, repeat=False)

        self.canvas.draw()  # Redraw the canvas
