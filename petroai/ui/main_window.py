from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QStatusBar, QLabel,
    QFileDialog, QMessageBox, QToolBar
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QFont
import config
from core.las_reader import LASReader, WellData
from core.petrophysics import PetrophysicsEngine, PetroResult
from ai.ollama_client import AIClient, check_ollama_connection
from ai.context_builder import ContextBuilder
from ui.log_viewer import LogViewer
from ui.chat_panel import ChatPanel, AISettingsDialog


# ------------------------------------------------------------------ threads
class LoadLASThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.finished.emit(LASReader().read(self.path))
        except Exception as e:
            self.error.emit(str(e))


class CalcPetroThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, well: WellData):
        super().__init__()
        self.well = well

    def run(self):
        try:
            result = PetrophysicsEngine().analyze(self.well)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(None)


class CheckAIThread(QThread):
    result = pyqtSignal(bool, list)

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client

    def run(self):
        if self.client.is_local():
            ok, models = check_ollama_connection()
        else:
            ok, models = True, []
        self.result.emit(ok, models)


# ------------------------------------------------------------------ main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{config.APP_NAME} v{config.APP_VERSION}")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        self._well_data: WellData | None = None
        self._petro_result: PetroResult | None = None
        self._load_thread: LoadLASThread | None = None
        self._calc_thread: CalcPetroThread | None = None

        # AI components (created before UI)
        self._ai_client = AIClient()
        self._context = ContextBuilder()

        self._setup_style()
        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        # Check AI connection setelah window tampil
        QTimer.singleShot(500, self._check_ai_connection)

    # ------------------------------------------------------------------ style
    def _setup_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QMenuBar {
                background-color: #181825; color: #cdd6f4;
                border-bottom: 1px solid #313244; font-size: 13px;
            }
            QMenuBar::item:selected { background-color: #313244; }
            QMenu {
                background-color: #1e1e2e; color: #cdd6f4;
                border: 1px solid #313244;
            }
            QMenu::item:selected { background-color: #45475a; }
            QToolBar {
                background-color: #181825;
                border-bottom: 1px solid #313244;
                spacing: 4px; padding: 4px;
            }
            QToolButton {
                color: #cdd6f4; background-color: transparent;
                border: 1px solid transparent; border-radius: 4px;
                padding: 4px 8px; font-size: 12px;
            }
            QToolButton:hover { background-color: #313244; border-color: #45475a; }
            QStatusBar {
                background-color: #181825; color: #a6adc8;
                border-top: 1px solid #313244; font-size: 12px;
            }
            QSplitter::handle { background-color: #313244; }
            QSplitter::handle:horizontal { width: 2px; }
        """)

    # ------------------------------------------------------------------ menu
    def _build_menu(self):
        mb = self.menuBar()

        # File
        fm = mb.addMenu("File")
        self.act_open = QAction("Buka File LAS...", self)
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.triggered.connect(self._on_open_las)
        fm.addAction(self.act_open)

        self.act_close = QAction("Tutup Well", self)
        self.act_close.setEnabled(False)
        self.act_close.triggered.connect(self._on_close_well)
        fm.addAction(self.act_close)

        fm.addSeparator()
        act_exit = QAction("Keluar", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        fm.addAction(act_exit)

        # Tampilan
        vm = mb.addMenu("Tampilan")
        self.act_toggle_chat = QAction("Sembunyikan/Tampilkan Chat AI", self)
        self.act_toggle_chat.setShortcut("Ctrl+Shift+C")
        self.act_toggle_chat.triggered.connect(self._toggle_chat_panel)
        vm.addAction(self.act_toggle_chat)

        # AI
        am = mb.addMenu("AI")
        self.act_interpret = QAction("Interpretasi Otomatis", self)
        self.act_interpret.setShortcut("Ctrl+I")
        self.act_interpret.setEnabled(False)
        self.act_interpret.triggered.connect(self._on_auto_interpret)
        am.addAction(self.act_interpret)

        self.act_calc = QAction("Hitung Petrofisik", self)
        self.act_calc.setShortcut("Ctrl+P")
        self.act_calc.setEnabled(False)
        self.act_calc.triggered.connect(self._on_calc_petro)
        am.addAction(self.act_calc)

        am.addSeparator()
        act_ai_settings = QAction("Pengaturan Model AI...", self)
        act_ai_settings.triggered.connect(self._on_ai_settings)
        am.addAction(act_ai_settings)

        # Bantuan
        hm = mb.addMenu("Bantuan")
        act_about = QAction("Tentang PetroAI", self)
        act_about.triggered.connect(self._on_about)
        hm.addAction(act_about)

    # ------------------------------------------------------------------ toolbar
    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)

        act = QAction("📂  Buka LAS", self)
        act.triggered.connect(self._on_open_las)
        tb.addAction(act)

        tb.addSeparator()

        self.btn_calc = QAction("🧮  Hitung Petrofisik", self)
        self.btn_calc.setEnabled(False)
        self.btn_calc.triggered.connect(self._on_calc_petro)
        tb.addAction(self.btn_calc)

        self.btn_interpret = QAction("🤖  Auto Interpretasi", self)
        self.btn_interpret.setEnabled(False)
        self.btn_interpret.triggered.connect(self._on_auto_interpret)
        tb.addAction(self.btn_interpret)

        tb.addSeparator()

        act_settings = QAction("⚙  AI Settings", self)
        act_settings.triggered.connect(self._on_ai_settings)
        tb.addAction(act_settings)

    # ------------------------------------------------------------------ central
    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)

        # Left: well explorer
        self.left_panel = self._build_left_panel()
        self.splitter.addWidget(self.left_panel)

        # Center: log viewer
        self.log_viewer = LogViewer()
        self.splitter.addWidget(self.log_viewer)

        # Right: AI chat
        self.chat_panel = ChatPanel(self._ai_client, self._context)
        self.splitter.addWidget(self.chat_panel)

        self.splitter.setSizes([220, 780, 380])
        self.splitter.setStretchFactor(1, 1)

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(180)
        panel.setMaximumWidth(280)
        panel.setStyleSheet("background-color: #181825; border-right: 1px solid #313244;")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)

        title = QLabel("WELL EXPLORER")
        title.setStyleSheet(
            "color: #6c7086; font-size: 11px; font-weight: bold; letter-spacing: 1px;"
        )
        layout.addWidget(title)

        self.well_info_label = QLabel("Belum ada data\nBuka file LAS untuk memulai")
        self.well_info_label.setStyleSheet("color: #585b70; font-size: 12px; padding: 8px;")
        self.well_info_label.setWordWrap(True)
        self.well_info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.well_info_label)

        # Petro result summary (muncul setelah kalkulasi)
        self.petro_summary_label = QLabel("")
        self.petro_summary_label.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 4px;")
        self.petro_summary_label.setWordWrap(True)
        layout.addWidget(self.petro_summary_label)

        layout.addStretch()

        self.ai_status_dot = QLabel("● AI: Memeriksa...")
        self.ai_status_dot.setStyleSheet("color: #f9e2af; font-size: 11px; padding: 4px;")
        layout.addWidget(self.ai_status_dot)

        return panel

    # ------------------------------------------------------------------ statusbar
    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self.status_main = QLabel("Siap")
        sb.addWidget(self.status_main)

        sb.addPermanentWidget(QLabel("|"))
        self.status_well = QLabel("Well: -")
        sb.addPermanentWidget(self.status_well)

        sb.addPermanentWidget(QLabel("|"))
        self.status_depth = QLabel("Depth: -")
        sb.addPermanentWidget(self.status_depth)

        sb.addPermanentWidget(QLabel("|"))
        self.status_ai = QLabel("AI: Memeriksa...")
        self.status_ai.setStyleSheet("color: #f9e2af;")
        sb.addPermanentWidget(self.status_ai)

    # ------------------------------------------------------------------ slots
    def _on_open_las(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Buka File LAS", "", "LAS Files (*.las *.LAS);;All Files (*)"
        )
        if not path:
            return
        self.status_main.setText(f"Memuat: {path} ...")
        self._load_thread = LoadLASThread(path)
        self._load_thread.finished.connect(self._on_las_loaded)
        self._load_thread.error.connect(self._on_las_error)
        self._load_thread.start()

    def _on_las_loaded(self, well_data: WellData):
        self._well_data = well_data
        self._context.set_well(well_data)
        self.log_viewer.load_well(well_data)
        self._update_well_info(well_data)
        self.status_main.setText(
            f"Loaded: {well_data.name} — {len(well_data.curves)} kurva"
        )
        # Auto jalankan kalkulasi petrofisik
        self._on_calc_petro()

    def _on_las_error(self, msg: str):
        QMessageBox.critical(self, "Gagal Membaca LAS", f"Error:\n{msg}")
        self.status_main.setText("Gagal memuat file LAS.")

    def _on_close_well(self):
        self._well_data = None
        self._petro_result = None
        self._context.set_well(None)
        self._context.set_petro_result(None)
        self.log_viewer.clear()
        self.well_info_label.setText("Belum ada data\nBuka file LAS untuk memulai")
        self.petro_summary_label.setText("")
        self.status_well.setText("Well: -")
        self.status_depth.setText("Depth: -")
        self.status_main.setText("Well ditutup.")
        self.act_close.setEnabled(False)
        self.act_interpret.setEnabled(False)
        self.act_calc.setEnabled(False)
        self.btn_interpret.setEnabled(False)
        self.btn_calc.setEnabled(False)

    def _on_calc_petro(self):
        if not self._well_data:
            return
        self.status_main.setText("Menghitung petrofisik...")
        self._calc_thread = CalcPetroThread(self._well_data)
        self._calc_thread.finished.connect(self._on_petro_done)
        self._calc_thread.start()

    def _on_petro_done(self, result: PetroResult | None):
        if result is None:
            self.status_main.setText("Kalkulasi petrofisik gagal.")
            return
        self._petro_result = result
        self._context.set_petro_result(result)

        import numpy as np
        n_zones = len(result.zones)
        net_pay = result.net_pay_flag.sum() * self._well_data.depth_step
        avg_phie = float(np.nanmean(result.phie)) if not np.all(np.isnan(result.phie)) else 0.0
        avg_sw = float(np.nanmean(result.sw)) if not np.all(np.isnan(result.sw)) else 1.0

        summary = (
            f"<b style='color:#89b4fa'>Petrofisik OK</b><br>"
            f"Zona: {n_zones}<br>"
            f"Net Pay: {net_pay:.1f} m<br>"
            f"Avg PHIE: {avg_phie:.3f}<br>"
            f"Avg SW: {avg_sw:.3f}"
        )
        self.petro_summary_label.setText(summary)
        self.status_main.setText(
            f"Petrofisik selesai — {n_zones} zona, net pay {net_pay:.1f} m"
        )

        # Log kurva hasil kalkulasi ke viewer
        self._update_log_viewer_with_petro(result)

    def _update_log_viewer_with_petro(self, result: PetroResult):
        import pandas as pd
        petro_df = result.as_dataframe()
        if self._well_data:
            for col in petro_df.columns:
                self._well_data.df[col] = petro_df[col]
            from core.las_reader import CurveInfo
            import numpy as np
            for col in petro_df.columns:
                arr = petro_df[col].to_numpy()
                valid = arr[~np.isnan(arr)]
                if not any(c.mnemonic == col for c in self._well_data.curves):
                    self._well_data.curves.append(CurveInfo(
                        mnemonic=col, unit="", description="Calculated",
                        min_val=float(valid.min()) if len(valid) > 0 else 0,
                        max_val=float(valid.max()) if len(valid) > 0 else 1,
                        mean_val=float(valid.mean()) if len(valid) > 0 else 0,
                        nan_pct=100 * np.isnan(arr).sum() / len(arr),
                    ))
            self.log_viewer.load_well(self._well_data)

    def _on_auto_interpret(self):
        if not self._well_data:
            return
        self.status_main.setText("Mengirim data ke AI untuk interpretasi...")
        self.chat_panel.run_auto_interpretation()

    def _toggle_chat_panel(self):
        self.chat_panel.setVisible(not self.chat_panel.isVisible())

    def _on_ai_settings(self):
        dlg = AISettingsDialog(self._ai_client, self)
        if dlg.exec():
            self.chat_panel.update_provider_label()
            self._update_ai_status_label()

    def _on_about(self):
        QMessageBox.about(
            self, "Tentang PetroAI",
            f"<h3>{config.APP_NAME} v{config.APP_VERSION}</h3>"
            "<p>Software analisis petrofisik dengan AI lokal &amp; cloud.</p>"
            "<p><b>Fitur:</b></p>"
            "<ul>"
            "<li>Baca file LAS industri standar</li>"
            "<li>Hitung Vshale, Porosity, Water Saturation, Net Pay</li>"
            "<li>Visualisasi log multi-track interaktif</li>"
            "<li>Auto interpretasi via AI (Ollama lokal atau cloud)</li>"
            "<li>Chatbot petrofisik berbasis LLM</li>"
            "</ul>"
        )

    # ------------------------------------------------------------------ helpers
    def _update_well_info(self, w: WellData):
        curves_str = ", ".join(c.mnemonic for c in w.curves[:8])
        if len(w.curves) > 8:
            curves_str += f" +{len(w.curves)-8} lagi"
        self.well_info_label.setText(
            f"<b style='color:#cdd6f4'>{w.name}</b><br>"
            f"<span style='color:#a6adc8'>Field: {w.field}</span><br>"
            f"<span style='color:#a6adc8'>Depth: {w.depth_top:.0f}–{w.depth_bottom:.0f} {w.depth_unit}</span><br>"
            f"<span style='color:#6c7086; font-size:11px'>{curves_str}</span>"
        )
        self.status_well.setText(f"Well: {w.name}")
        self.status_depth.setText(f"Depth: {w.depth_top:.0f}–{w.depth_bottom:.0f} m")
        self.act_close.setEnabled(True)
        self.act_interpret.setEnabled(True)
        self.act_calc.setEnabled(True)
        self.btn_interpret.setEnabled(True)
        self.btn_calc.setEnabled(True)

    def _check_ai_connection(self):
        thread = CheckAIThread(self._ai_client, self)
        thread.result.connect(self._on_ai_check_done)
        thread.start()
        self._ai_check_thread = thread

    def _on_ai_check_done(self, ok: bool, models: list):
        self._update_ai_status(ok)

    def _update_ai_status(self, connected: bool = True):
        provider = self._ai_client.provider_label()
        if self._ai_client.is_local():
            if connected:
                self.ai_status_dot.setText(f"● AI: {self._ai_client.model}")
                self.ai_status_dot.setStyleSheet("color: #a6e3a1; font-size: 11px; padding: 4px;")
                self.status_ai.setText(f"AI: Online ({self._ai_client.model})")
                self.status_ai.setStyleSheet("color: #a6e3a1;")
            else:
                self.ai_status_dot.setText("● AI: Ollama Offline")
                self.ai_status_dot.setStyleSheet("color: #f38ba8; font-size: 11px; padding: 4px;")
                self.status_ai.setText("AI: Offline — jalankan ollama serve")
                self.status_ai.setStyleSheet("color: #f38ba8;")
        else:
            self.ai_status_dot.setText(f"● AI: {provider}")
            self.ai_status_dot.setStyleSheet("color: #a6e3a1; font-size: 11px; padding: 4px;")
            self.status_ai.setText(f"AI: {provider}")
            self.status_ai.setStyleSheet("color: #a6e3a1;")

    def _update_ai_status_label(self):
        self._update_ai_status(True)
