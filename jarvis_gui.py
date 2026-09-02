"""
Jarvis Desktop Application GUI (PyQt6)
----------------------------------------
Futuristic, native Windows Desktop Interface for Jarvis Voice Assistant.
Features:
- Animated Arc-Reactor HUD status visualizer (QPainter)
- Real-time chat history feed with glowing user/assistant speech bubbles
- Direct text query input bar
- System health status metrics (Whisper, Ollama, Audio Gain)
- System Tray integration with background minimize capability
"""

import sys
import os
import math
import time
import logging
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPoint, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QIcon, QLinearGradient, QRadialGradient, QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QFrame,
    QSystemTrayIcon,
    QMenu,
    QGraphicsDropShadowEffect,
    QSizePolicy,
)

import jarvis
from jarvis import JarvisAssistant, WAKE_WORD_MODEL, OLLAMA_MODEL, WHISPER_MODEL_SIZE, WHISPER_DEVICE, AUDIO_GAIN


# ==========================================
# THREAD-SAFE SIGNAL BRIDGE
# ==========================================
class AssistantSignals(QObject):
    state_changed = pyqtSignal(str, str)
    user_transcript = pyqtSignal(str)
    llm_response = pyqtSignal(str)


# ==========================================
# ANIMATED ARC REACTOR HUD WIDGET
# ==========================================
class AnimatedArcReactorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.state = "listening"
        self.angle = 0
        self.pulse = 0
        self.pulse_dir = 1

        # Timer for 30 FPS HUD animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(33)

    def set_state(self, state):
        self.state = state
        self.update()

    def update_animation(self):
        self.angle = (self.angle + 2) % 360
        self.pulse += 0.04 * self.pulse_dir
        if self.pulse >= 1.0:
            self.pulse = 1.0
            self.pulse_dir = -1
        elif self.pulse <= 0.0:
            self.pulse = 0.0
            self.pulse_dir = 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        cx, cy = width / 2.0, height / 2.0
        radius = min(width, height) / 2.0 - 15

        # Colors based on state
        if self.state == "listening":
            core_color = QColor(6, 182, 212)      # Cyan
            glow_color = QColor(6, 182, 212, 80)
        elif self.state in ("recording", "thinking"):
            core_color = QColor(245, 158, 11)     # Amber
            glow_color = QColor(245, 158, 11, 80)
        elif self.state == "speaking":
            core_color = QColor(168, 85, 247)     # Purple
            glow_color = QColor(168, 85, 247, 80)
        elif self.state == "paused":
            core_color = QColor(100, 116, 139)    # Muted Grey
            glow_color = QColor(100, 116, 139, 50)
        else:  # error
            core_color = QColor(239, 68, 68)      # Crimson Red
            glow_color = QColor(239, 68, 68, 80)

        # 1. Background Outer Ring
        pen = QPen(QColor(30, 41, 59), 4)
        painter.setPen(pen)
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(radius), int(radius))

        # 2. Pulsing Glow Gradient Background
        radial_grad = QRadialGradient(cx, cy, radius)
        radial_grad.setColorAt(0.0, QColor(core_color.red(), core_color.green(), core_color.blue(), int(120 + 60 * self.pulse)))
        radial_grad.setColorAt(0.7, QColor(core_color.red(), core_color.green(), core_color.blue(), int(30 + 20 * self.pulse)))
        radial_grad.setColorAt(1.0, QColor(15, 23, 42, 0))
        painter.setBrush(QBrush(radial_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(radius), int(radius))

        # 3. Outer Segmented Rotating Ring
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.angle)

        pen = QPen(core_color, 3)
        painter.setPen(pen)
        num_segments = 12
        for i in range(num_segments):
            angle_deg = i * (360 / num_segments)
            painter.drawArc(
                int(-radius + 8),
                int(-radius + 8),
                int((radius - 8) * 2),
                int((radius - 8) * 2),
                int(angle_deg * 16),
                18 * 16,
            )
        painter.restore()

        # 4. Counter-rotating Inner Segment Ring
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(-self.angle * 1.5)
        inner_r = radius * 0.65
        pen = QPen(QColor(255, 255, 255, 180), 2)
        painter.setPen(pen)
        for i in range(6):
            angle_deg = i * (360 / 6)
            painter.drawArc(
                int(-inner_r),
                int(-inner_r),
                int(inner_r * 2),
                int(inner_r * 2),
                int(angle_deg * 16),
                40 * 16,
            )
        painter.restore()

        # 5. Core Bright Glowing Center
        core_r = radius * 0.35 + (self.pulse * 4)
        radial_core = QRadialGradient(cx, cy, core_r)
        radial_core.setColorAt(0.0, QColor(255, 255, 255, 255))
        radial_core.setColorAt(0.5, core_color)
        radial_core.setColorAt(1.0, QColor(core_color.red(), core_color.green(), core_color.blue(), 100))
        painter.setBrush(QBrush(radial_core))
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(core_r), int(core_r))


# ==========================================
# CHAT BUBBLE WIDGET
# ==========================================
class ChatBubble(QFrame):
    def __init__(self, sender, message, timestamp=None, parent=None):
        super().__init__(parent)
        self.setObjectName("ChatBubble")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)

        time_str = timestamp or datetime.now().strftime("%I:%M %p")
        
        # Header (Sender Name + Time)
        header_layout = QHBoxLayout()
        sender_label = QLabel(sender)
        sender_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        
        time_label = QLabel(time_str)
        time_label.setFont(QFont("Segoe UI", 8))
        time_label.setStyleSheet("color: #64748b;")

        if sender.lower() == "you":
            sender_label.setStyleSheet("color: #38bdf8;") # Sky Blue
            header_layout.addWidget(sender_label)
            header_layout.addStretch()
            header_layout.addWidget(time_label)
            self.setStyleSheet("""
                #ChatBubble {
                    background-color: #1e293b;
                    border-left: 3px solid #38bdf8;
                    border-radius: 8px;
                }
            """)
        else:
            sender_label.setStyleSheet("color: #f59e0b;") # Amber Glow
            header_layout.addWidget(sender_label)
            header_layout.addStretch()
            header_layout.addWidget(time_label)
            self.setStyleSheet("""
                #ChatBubble {
                    background-color: #0f172a;
                    border-left: 3px solid #f59e0b;
                    border-radius: 8px;
                }
            """)

        # Message Text
        text_label = QLabel(message)
        text_label.setWordWrap(True)
        text_label.setFont(QFont("Segoe UI", 10))
        text_label.setStyleSheet("color: #f8fafc; line-height: 1.4;")
        text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout.addLayout(header_layout)
        layout.addWidget(text_label)


# ==========================================
# MAIN NATIVE DESKTOP WINDOW
# ==========================================
class JarvisMainWindow(QMainWindow):
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.signals = AssistantSignals()

        # Connect thread-safe signals to GUI updates
        self.signals.state_changed.connect(self.on_state_changed)
        self.signals.user_transcript.connect(self.on_user_transcript)
        self.signals.llm_response.connect(self.on_llm_response)

        # Attach callbacks to assistant
        self.assistant.on_state_change = lambda s, d: self.signals.state_changed.emit(s, d)
        self.assistant.on_user_transcript = lambda t: self.signals.user_transcript.emit(t)
        self.assistant.on_llm_response = lambda r: self.signals.llm_response.emit(r)

        self.init_ui()
        self.init_tray()

    def init_ui(self):
        self.setWindowTitle("Jarvis Desktop Assistant")
        self.resize(960, 680)
        self.setMinimumSize(780, 540)

        # Apply Global Dark Glassmorphism Stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0f19;
            }
            QWidget {
                color: #f8fafc;
                font-family: 'Segoe UI', sans-serif;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 12px 16px;
                font-size: 14px;
                color: #f8fafc;
            }
            QLineEdit:focus {
                border: 1px solid #06b6d4;
                background-color: #0f172a;
            }
            QPushButton {
                background-color: #06b6d4;
                border: none;
                border-radius: 10px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 13px;
                color: #0b0f19;
            }
            QPushButton:hover {
                background-color: #22d3ee;
            }
            QPushButton:pressed {
                background-color: #0891b2;
            }
            QPushButton#BtnPause {
                background-color: #334155;
                color: #f8fafc;
            }
            QPushButton#BtnPause:hover {
                background-color: #475569;
            }
        """)

        # Main Central Container
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # -------------------------------------------------------------
        # LEFT PANEL: Arc-Reactor HUD & System Status Metrics
        # -------------------------------------------------------------
        left_panel = QFrame()
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 16px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 24, 20, 24)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Title Label
        app_title = QLabel("JARVIS HUD")
        app_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        app_title.setStyleSheet("color: #06b6d4; letter-spacing: 2px;")
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(app_title)

        # Animated Arc Reactor Widget
        self.arc_reactor = AnimatedArcReactorWidget(self)
        left_layout.addWidget(self.arc_reactor)

        # State Status Text
        self.status_label = QLabel("LISTENING FOR 'HEY JARVIS'")
        self.status_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #38bdf8; text-transform: uppercase;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        left_layout.addSpacing(15)

        # System Metrics Badges
        metrics_box = QVBoxLayout()
        metrics_box.setSpacing(8)

        def make_badge(key, val, color="#94a3b8"):
            lbl = QLabel(f"<span style='color:#64748b;'>{key}:</span> <b style='color:{color};'>{val}</b>")
            lbl.setFont(QFont("Segoe UI", 9))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return lbl

        metrics_box.addWidget(make_badge("STT Model", f"Whisper ({WHISPER_MODEL_SIZE})", "#38bdf8"))
        metrics_box.addWidget(make_badge("Device", WHISPER_DEVICE.upper(), "#10b981"))
        metrics_box.addWidget(make_badge("LLM Model", OLLAMA_MODEL, "#f59e0b"))
        metrics_box.addWidget(make_badge("Digital Gain", f"{AUDIO_GAIN:.0f}x Boost", "#a855f7"))

        left_layout.addLayout(metrics_box)
        left_layout.addStretch()

        # Pause / Resume Voice Button
        self.btn_pause = QPushButton("Pause Listening")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_pause.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_pause.clicked.connect(self.toggle_pause)
        left_layout.addWidget(self.btn_pause)

        main_layout.addWidget(left_panel, stretch=1)

        # -------------------------------------------------------------
        # RIGHT PANEL: Live Chat Stream & Text Input Bar
        # -------------------------------------------------------------
        right_panel = QFrame()
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 16px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)

        # Right Panel Header
        header_lbl = QLabel("Conversation Feed")
        header_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header_lbl.setStyleSheet("color: #94a3b8;")
        right_layout.addWidget(header_lbl)

        # Scrollable Chat Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.setSpacing(12)
        self.chat_layout.addStretch()
        self.scroll_area.setWidget(self.chat_container)

        right_layout.addWidget(self.scroll_area, stretch=1)

        # Bottom Input Area
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a prompt or speak 'Hey Jarvis'...")
        self.input_field.returnPressed.connect(self.send_text_prompt)
        input_layout.addWidget(self.input_field)

        self.btn_send = QPushButton("Send")
        self.btn_send.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_send.clicked.connect(self.send_text_prompt)
        input_layout.addWidget(self.btn_send)

        right_layout.addLayout(input_layout)

        main_layout.addWidget(right_panel, stretch=2)

        # Add Initial Welcome Message
        self.add_message("Jarvis", f"Greetings! I am active and listening for your wake word '{WAKE_WORD_MODEL}'. How can I assist you?")

    def init_tray(self):
        """Initialize System Tray Icon for desktop background execution."""
        self.tray_icon = QSystemTrayIcon(self)
        tray_img = jarvis.create_tray_icon_image("listening")
        
        # Convert PIL Image to QIcon via bytes
        import io
        byte_arr = io.BytesIO()
        tray_img.save(byte_arr, format='PNG')
        pm = QIcon()
        from PyQt6.QtGui import QPixmap
        pix = QPixmap()
        pix.loadFromData(byte_arr.getvalue())
        pm.addPixmap(pix)
        
        self.tray_icon.setIcon(pm)
        self.tray_icon.setToolTip("Jarvis Voice Assistant")

        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show Jarvis Window")
        show_action.triggered.connect(self.show_normal_window)

        pause_action = tray_menu.addAction("Toggle Pause/Listen")
        pause_action.triggered.connect(self.toggle_pause)

        tray_menu.addSeparator()
        quit_action = tray_menu.addAction("Quit Jarvis")
        quit_action.triggered.connect(self.quit_app)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        self.tray_icon.show()

    def show_normal_window(self):
        self.showNormal()
        self.activateWindow()

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show_normal_window()

    def closeEvent(self, event):
        """Minimize to tray when close button is clicked."""
        if self.tray_icon.isVisible():
            self.hide()
            self.tray_icon.showMessage(
                "Jarvis Voice Assistant",
                "Jarvis is running in the background. Double-click tray icon to restore.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
            event.ignore()
        else:
            event.accept()

    def add_message(self, sender, message):
        bubble = ChatBubble(sender, message)
        # Insert before stretch
        count = self.chat_layout.count()
        self.chat_layout.insertWidget(count - 1, bubble)
        
        # Scroll to bottom after layout update
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def on_state_changed(self, new_state, description):
        self.arc_reactor.set_state(new_state)
        self.status_label.setText(description.upper())

    def on_user_transcript(self, text):
        if text:
            self.add_message("You", text)

    def on_llm_response(self, text):
        if text:
            self.add_message("Jarvis", text)

    def send_text_prompt(self):
        text = self.input_field.text().strip()
        if text:
            self.input_field.clear()
            self.assistant.process_text_prompt(text)

    def toggle_pause(self):
        self.assistant.is_paused = not self.assistant.is_paused
        if self.assistant.is_paused:
            self.assistant.set_state("paused", "Listening Paused")
            self.btn_pause.setText("Resume Listening")
        else:
            self.assistant.set_state("listening", f"Listening for '{WAKE_WORD_MODEL}'")
            self.btn_pause.setText("Pause Listening")

    def quit_app(self):
        self.assistant.stop_event.set()
        QApplication.quit()


# ==========================================
# WORKER THREAD FOR ASSISTANT LOOP
# ==========================================
class AssistantWorkerThread(QThread):
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant

    def run(self):
        self.assistant.run_assistant_loop()


# ==========================================
# MAIN ENTRYPOINT
# ==========================================
def main():
    jarvis.setup_logging()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    assistant = JarvisAssistant()
    window = JarvisMainWindow(assistant)
    window.show()

    # Start assistant core worker thread
    worker = AssistantWorkerThread(assistant)
    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
