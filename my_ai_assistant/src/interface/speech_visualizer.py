from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt6.QtCore import Qt
import numpy as np

class AudioVisualizer(QLabel):
    def __init__(self, width=600, height=200, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: black;")
        self.bars = np.zeros(32)

    def update_bars(self, levels):
        self.bars = np.array(levels)
        self.repaint()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        w, h = self.width(), self.height()
        num_bars = len(self.bars)
        bar_width = w // (num_bars + 2)
        spacing = 4
        for i, level in enumerate(self.bars):
            x = 10 + i * (bar_width + spacing)
            bar_h = int(level * (h - 20))
            y = h - bar_h - 10
            color = QColor(255, 255, 0)
            painter.setBrush(color)
            # painter.setPen(Qt.NoPen)
            painter.drawRect(x, y, bar_width, bar_h)
        painter.end()