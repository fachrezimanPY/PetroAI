from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QScrollArea,
    QFrame, QSizePolicy, QDialog, QFormLayout,
    QComboBox, QDialogButtonBox, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor
from ai.ollama_client import AIClient, PROVIDERS, check_ollama_connection
from ai.context_builder import ContextBuilder
import config


# ------------------------------------------------------------------ AI Settings Dialog
class AISettingsDialog(QDialog):
    def __init__(self, client: AIClient, parent=None):
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Pengaturan Model AI")
        self.setMinimumWidth(480)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #cdd6f4; }
            QComboBox, QLineEdit {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 4px;
                padding: 6px; font-size: 13px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #313244; color: #cdd6f4;
                selection-background-color: #45475a;
            }
            QPushButton {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 4px;
                padding: 6px 12px; font-size: 13px;
            }
            QPushButton:hover { background-color: #45475a; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)

        # Provider selector
        self.provider_combo = QComboBox()
        for key, val in PROVIDERS.items():
            self.provider_combo.addItem(val["label"], key)
        idx = list(PROVIDERS.keys()).index(client.provider) if client.provider in PROVIDERS else 0
        self.provider_combo.setCurrentIndex(idx)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow("Provider:", self.provider_combo)

        # Model name
        self.model_edit = QLineEdit(client.model)
        self.model_edit.setPlaceholderText("Nama model...")
        form.addRow("Model:", self.model_edit)

        # API Key
        self.key_label = QLabel("API Key:")
        self.key_edit = QLineEdit(client.api_key)
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("sk-... atau apikey-...")
        form.addRow(self.key_label, self.key_edit)

        layout.addLayout(form)

        # Model examples
        self.examples_label = QLabel()
        self.examples_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        self.examples_label.setWordWrap(True)
        layout.addWidget(self.examples_label)

        # Test connection button
        self.btn_test = QPushButton("🔌 Test Koneksi")
        self.btn_test.clicked.connect(self._test_connection)
        layout.addWidget(self.btn_test)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_provider_changed()

    def _on_provider_changed(self):
        key = self.provider_combo.currentData()
        pdef = PROVIDERS.get(key, {})
        needs_key = pdef.get("needs_key", False)
        self.key_label.setVisible(needs_key)
        self.key_edit.setVisible(needs_key)

        if needs_key:
            self.key_edit.setText(self.client.get_stored_key(key))

        examples = pdef.get("model_examples", [])
        self.examples_label.setText("Contoh model: " + ", ".join(examples))
        if not self.model_edit.text() or self.model_edit.text() in [
            p["default_model"] for p in PROVIDERS.values()
        ]:
            self.model_edit.setText(pdef.get("default_model", ""))

    def _test_connection(self):
        key = self.provider_combo.currentData()
        if key == "local_ollama":
            ok, models = check_ollama_connection()
            if ok:
                self.result_label.setText(
                    f"✅ Ollama terhubung! Model tersedia: {', '.join(models[:5])}"
                )
                self.result_label.setStyleSheet("color: #a6e3a1;")
            else:
                self.result_label.setText(
                    "❌ Ollama tidak berjalan. Jalankan: ollama serve"
                )
                self.result_label.setStyleSheet("color: #f38ba8;")
        else:
            api_key = self.key_edit.text().strip()
            if not api_key:
                self.result_label.setText("❌ API Key tidak boleh kosong untuk provider cloud.")
                self.result_label.setStyleSheet("color: #f38ba8;")
            else:
                self.result_label.setText("✅ API Key tersimpan. Akan diverifikasi saat pertama kali chat.")
                self.result_label.setStyleSheet("color: #a6e3a1;")

    def _save(self):
        key = self.provider_combo.currentData()
        model = self.model_edit.text().strip()
        api_key = self.key_edit.text().strip()
        if not model:
            QMessageBox.warning(self, "Error", "Nama model tidak boleh kosong.")
            return
        self.client.configure(key, model, api_key)
        self.accept()


# ------------------------------------------------------------------ Bubble widget
class MessageBubble(QFrame):
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        lbl = QLabel()
        lbl.setTextFormat(Qt.TextFormat.MarkdownText)
        lbl.setText(text)
        lbl.setWordWrap(True)
        lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lbl.setFont(QFont("Segoe UI", 12))
        lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )

        if is_user:
            lbl.setStyleSheet(
                "background-color: #313244; color: #cdd6f4; "
                "border-radius: 10px; padding: 8px 12px;"
            )
            layout.addStretch()
            layout.addWidget(lbl)
        else:
            lbl.setStyleSheet(
                "background-color: #181825; color: #cdd6f4; "
                "border-radius: 10px; padding: 8px 12px; "
                "border: 1px solid #313244;"
            )
            layout.addWidget(lbl)
            layout.addStretch()

        self.label = lbl


# ------------------------------------------------------------------ Chat Panel
class ChatPanel(QWidget):
    def __init__(self, ai_client: AIClient, context: ContextBuilder, parent=None):
        super().__init__(parent)
        self.ai_client = ai_client
        self.context = context
        self._history: list[dict] = []
        self._current_bubble: MessageBubble | None = None
        self._streaming = False

        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet("background-color: #181825;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet("background-color: #11111b; border-bottom: 1px solid #313244;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(12, 0, 8, 0)

        title = QLabel("🤖  AI Petrofisik")
        title.setStyleSheet("color: #cdd6f4; font-size: 13px; font-weight: bold;")
        h_layout.addWidget(title)
        h_layout.addStretch()

        self.provider_label = QLabel(self.ai_client.provider_label())
        self.provider_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        h_layout.addWidget(self.provider_label)

        btn_settings = QPushButton("⚙")
        btn_settings.setFixedSize(32, 32)
        btn_settings.setStyleSheet(
            "QPushButton { background: transparent; color: #6c7086; font-size: 16px; border: none; }"
            "QPushButton:hover { color: #cdd6f4; }"
        )
        btn_settings.clicked.connect(self._open_settings)
        h_layout.addWidget(btn_settings)

        layout.addWidget(header)

        # Scroll area untuk chat history
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: #181825; }")

        self._chat_container = QWidget()
        self._chat_container.setStyleSheet("background: #181825;")
        self._chat_layout = QVBoxLayout(self._chat_container)
        self._chat_layout.setSpacing(4)
        self._chat_layout.setContentsMargins(4, 8, 4, 8)
        self._chat_layout.addStretch()
        self._scroll.setWidget(self._chat_container)
        layout.addWidget(self._scroll)

        # Welcome message
        self._add_ai_message(
            "Halo! Saya adalah asisten petrofisik AI.\n\n"
            "Load file LAS terlebih dahulu, lalu saya bisa:\n"
            "- **Interpretasi otomatis** seluruh data sumur\n"
            "- Menjawab pertanyaan spesifik tentang data log\n"
            "- Menganalisis zona reservoir dan fluida\n\n"
            "Gunakan tombol **🤖 Auto Interpretasi** atau ketik pertanyaan di bawah."
        )

        # Input area
        input_area = QWidget()
        input_area.setStyleSheet("background-color: #11111b; border-top: 1px solid #313244;")
        input_layout = QVBoxLayout(input_area)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(6)

        # Quick action buttons
        quick_layout = QHBoxLayout()
        quick_layout.setSpacing(4)
        for label, prompt in [
            ("📊 Zona terbaik?", "Zona mana yang paling prospektif untuk diproduksikan?"),
            ("💧 Tipe fluida?", "Apa tipe fluida di masing-masing zona reservoir?"),
            ("🔢 Ringkasan statistik", "Berikan ringkasan statistik petrofisik untuk seluruh sumur ini."),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(
                "QPushButton { background: #313244; color: #a6adc8; border: 1px solid #45475a; "
                "border-radius: 4px; padding: 4px 8px; font-size: 11px; }"
                "QPushButton:hover { background: #45475a; color: #cdd6f4; }"
            )
            btn.clicked.connect(lambda _, p=prompt: self._send_message(p))
            quick_layout.addWidget(btn)
        quick_layout.addStretch()
        input_layout.addLayout(quick_layout)

        # Text input + send
        row = QHBoxLayout()
        row.setSpacing(6)
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Tanya tentang data sumur ini...")
        self.input_box.setStyleSheet(
            "QLineEdit { background: #313244; color: #cdd6f4; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 8px 12px; font-size: 13px; }"
            "QLineEdit:focus { border-color: #89b4fa; }"
        )
        self.input_box.returnPressed.connect(self._on_send)
        row.addWidget(self.input_box)

        self.send_btn = QPushButton("Kirim")
        self.send_btn.setFixedWidth(70)
        self.send_btn.setStyleSheet(
            "QPushButton { background: #89b4fa; color: #1e1e2e; border: none; "
            "border-radius: 6px; padding: 8px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #b4d0f7; }"
            "QPushButton:disabled { background: #313244; color: #585b70; }"
        )
        self.send_btn.clicked.connect(self._on_send)
        row.addWidget(self.send_btn)
        input_layout.addLayout(row)

        layout.addWidget(input_area)

    # ---------------------------------------------------------------- public
    def run_auto_interpretation(self):
        if not self.context.has_data():
            self._add_ai_message("⚠️ Load data LAS terlebih dahulu sebelum menjalankan interpretasi.")
            return
        if self._streaming:
            return

        self._add_user_message("🔍 Jalankan interpretasi otomatis untuk sumur ini.")
        messages = self.context.build_interpretation_prompt()
        self._stream_ai(messages)

    def update_provider_label(self):
        self.provider_label.setText(self.ai_client.provider_label())

    # ---------------------------------------------------------------- private
    def _on_send(self):
        text = self.input_box.text().strip()
        if text:
            self._send_message(text)

    def _send_message(self, text: str):
        if self._streaming or not text:
            return
        self.input_box.clear()
        self._add_user_message(text)
        messages = self.context.build_chat_messages(self._history, text)
        self._stream_ai(messages)

    def _stream_ai(self, messages: list[dict]):
        self._streaming = True
        self.send_btn.setEnabled(False)
        self.send_btn.setText("...")

        self._current_bubble = self._create_ai_bubble("")

        self.ai_client.chat_stream(
            messages,
            on_token=self._on_token,
            on_done=self._on_done,
            on_error=self._on_error,
        )

    def _on_token(self, token: str):
        if self._current_bubble:
            current_text = self._current_bubble.label.text()
            self._current_bubble.label.setText(current_text + token)
            QTimer.singleShot(10, self._scroll_to_bottom)

    def _on_done(self, full_text: str):
        self._history.append({"role": "assistant", "content": full_text})
        self._streaming = False
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Kirim")
        self._current_bubble = None

    def _on_error(self, msg: str):
        self._streaming = False
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Kirim")
        self._current_bubble = None
        self._add_ai_message(
            f"❌ **Error:** {msg}\n\n"
            "Pastikan:\n"
            "- Untuk Ollama: jalankan `ollama serve` di terminal\n"
            "- Untuk cloud: periksa API key di ⚙ pengaturan"
        )

    def _add_user_message(self, text: str):
        self._history.append({"role": "user", "content": text})
        bubble = MessageBubble(text, is_user=True)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _add_ai_message(self, text: str):
        bubble = MessageBubble(text, is_user=False)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _create_ai_bubble(self, text: str) -> MessageBubble:
        bubble = MessageBubble(text, is_user=False)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)
        return bubble

    def _scroll_to_bottom(self):
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _open_settings(self):
        dlg = AISettingsDialog(self.ai_client, self)
        if dlg.exec():
            self.update_provider_label()
            self._add_ai_message(
                f"✅ Konfigurasi AI diperbarui.\n"
                f"Provider: **{self.ai_client.provider_label()}**\n"
                f"Model: **{self.ai_client.model}**"
            )
