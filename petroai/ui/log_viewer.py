import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QFont
from core.las_reader import WellData

pg.setConfigOptions(antialias=True, background="#1e1e2e", foreground="#cdd6f4")

# Definisi track standar petrofisik
TRACK_DEFS = [
    {
        "title": "GR / SP",
        "curves": ["GR", "SP", "CGR"],
        "log_scale": False,
        "fill": True,
        "color": "#a6e3a1",
        "width": 160,
    },
    {
        "title": "Resistivity",
        "curves": ["RT", "RD", "RS", "ILD", "ILM", "LLD", "LLS", "MSFL", "RDEEP", "RMED", "RSHAL"],
        "log_scale": True,
        "fill": False,
        "color": "#f38ba8",
        "width": 160,
    },
    {
        "title": "NPHI / RHOB",
        "curves": ["NPHI", "TNPH", "CNPHI"],
        "log_scale": False,
        "fill": False,
        "color": "#89b4fa",
        "width": 160,
        "secondary": {
            "curves": ["RHOB", "ZDEN", "DEN"],
            "color": "#f9e2af",
        },
    },
    {
        "title": "DT / DTS",
        "curves": ["DT", "DTC", "AC"],
        "log_scale": False,
        "fill": False,
        "color": "#cba6f7",
        "width": 160,
    },
    {
        "title": "Vshale / SW",
        "curves": ["VSH", "VSHALE", "VCL"],
        "log_scale": False,
        "fill": True,
        "color": "#fab387",
        "width": 160,
        "secondary": {
            "curves": ["SW", "SWT", "SWE"],
            "color": "#89dceb",
        },
    },
]

CURVE_COLORS = [
    "#a6e3a1", "#f38ba8", "#89b4fa", "#f9e2af",
    "#cba6f7", "#fab387", "#89dceb", "#eba0ac",
]


class TrackWidget(QWidget):
    depth_changed = pyqtSignal(float)

    def __init__(self, title: str, width: int = 160, parent=None):
        super().__init__(parent)
        self.title = title
        self.setFixedWidth(width)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel(title)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFixedHeight(32)
        header.setStyleSheet(
            "background-color: #181825; color: #cdd6f4; font-size: 11px; "
            "font-weight: bold; border-bottom: 1px solid #313244; "
            "border-right: 1px solid #313244;"
        )
        layout.addWidget(header)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setStyleSheet("border-right: 1px solid #313244;")
        self.plot_widget.showGrid(x=True, y=False, alpha=0.15)
        self.plot_widget.getAxis("bottom").hide()
        self.plot_widget.getAxis("left").setStyle(tickFont=QFont("Courier", 8))

        # Y axis = depth (inverted)
        self.plot_widget.invertY(True)
        self.plot_widget.setLabel("left", "Depth", units="m")

        layout.addWidget(self.plot_widget)

        self._items: list[pg.PlotDataItem] = []

    def plot_curve(self, depth: np.ndarray, values: np.ndarray,
                   color: str = "#a6e3a1", log_scale: bool = False,
                   fill_below: bool = False, name: str = ""):
        pen = QPen(QColor(color))
        pen.setWidthF(1.5)

        x = values.copy().astype(float)
        y = depth.copy().astype(float)

        # mask NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) == 0:
            return

        if log_scale:
            x = np.where(x > 0, np.log10(x), np.nan)
            mask2 = ~np.isnan(x)
            x, y = x[mask2], y[mask2]

        item = self.plot_widget.plot(x, y, pen=pen, name=name)
        self._items.append(item)

        if fill_below:
            fill = pg.FillBetweenItem(
                item,
                pg.PlotDataItem(
                    np.zeros_like(x), y,
                    pen=pg.mkPen(None)
                ),
                brush=pg.mkBrush(QColor(color).darker(180))
            )
            self.plot_widget.addItem(fill)

        # batas min/max
        if len(x) > 0:
            x_range = (float(np.nanmin(x)), float(np.nanmax(x)))
            pad = (x_range[1] - x_range[0]) * 0.05 or 1.0
            self.plot_widget.setXRange(x_range[0] - pad, x_range[1] + pad, padding=0)

    def set_depth_range(self, top: float, bottom: float):
        self.plot_widget.setYRange(top, bottom, padding=0.01)

    def link_y_axis(self, other: "TrackWidget"):
        self.plot_widget.setYLink(other.plot_widget)
        self.plot_widget.getAxis("left").hide()

    def clear(self):
        self.plot_widget.clear()
        self._items.clear()


class DepthTrackWidget(QWidget):
    """Track kiri berisi skala kedalaman."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(70)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("DEPTH\n(m)")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFixedHeight(32)
        header.setStyleSheet(
            "background-color: #181825; color: #6c7086; font-size: 10px; "
            "border-bottom: 1px solid #313244; border-right: 1px solid #313244;"
        )
        layout.addWidget(header)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#181825")
        self.plot_widget.invertY(True)
        self.plot_widget.getAxis("bottom").hide()
        self.plot_widget.getAxis("left").setStyle(
            tickFont=QFont("Courier", 8), tickLength=-4
        )
        self.plot_widget.showGrid(x=False, y=True, alpha=0.2)
        layout.addWidget(self.plot_widget)

    def set_depth_range(self, top: float, bottom: float):
        self.plot_widget.setYRange(top, bottom, padding=0.01)


class LogViewer(QWidget):
    """Widget utama yang menampilkan semua track log."""
    cursor_depth_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._well_data: WellData | None = None
        self._tracks: list[TrackWidget] = []
        self._depth_track: DepthTrackWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area untuk track
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e2e; }")
        layout.addWidget(self._scroll)

        self._track_container = QWidget()
        self._track_container.setStyleSheet("background-color: #1e1e2e;")
        self._track_layout = QHBoxLayout(self._track_container)
        self._track_layout.setContentsMargins(0, 0, 0, 0)
        self._track_layout.setSpacing(0)
        self._scroll.setWidget(self._track_container)

    def load_well(self, well_data: WellData):
        self._well_data = well_data
        self._rebuild_tracks()

    def _rebuild_tracks(self):
        # hapus track lama
        for t in self._tracks:
            t.deleteLater()
        self._tracks.clear()
        if self._depth_track:
            self._depth_track.deleteLater()
            self._depth_track = None

        while self._track_layout.count():
            item = self._track_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._well_data is None:
            return

        depth = self._well_data.depth
        top, bottom = float(depth[0]), float(depth[-1])

        # Depth track (paling kiri)
        self._depth_track = DepthTrackWidget()
        self._depth_track.set_depth_range(top, bottom)
        self._track_layout.addWidget(self._depth_track)

        first_track = None

        # Buat track sesuai TRACK_DEFS
        for tdef in TRACK_DEFS:
            plotted_curves = self._find_curves(tdef["curves"])
            secondary_curves = []
            if "secondary" in tdef:
                secondary_curves = self._find_curves(tdef["secondary"]["curves"])

            if not plotted_curves and not secondary_curves:
                continue

            track = TrackWidget(tdef["title"], width=tdef.get("width", 160))
            track.set_depth_range(top, bottom)

            for i, (mnemonic, values) in enumerate(plotted_curves):
                color = tdef["color"] if i == 0 else CURVE_COLORS[(i + 2) % len(CURVE_COLORS)]
                track.plot_curve(
                    depth, values,
                    color=color,
                    log_scale=tdef.get("log_scale", False),
                    fill_below=tdef.get("fill", False) and i == 0,
                    name=mnemonic,
                )

            if "secondary" in tdef:
                sec_color = tdef["secondary"]["color"]
                for i, (mnemonic, values) in enumerate(secondary_curves):
                    track.plot_curve(depth, values, color=sec_color, name=mnemonic)

            # link Y axis ke depth track
            track.plot_widget.setYLink(self._depth_track.plot_widget)
            track.plot_widget.getAxis("left").hide()

            self._track_layout.addWidget(track)
            self._tracks.append(track)

            if first_track is None:
                first_track = track

        # Tambahkan kurva yang tidak masuk TRACK_DEFS (catch-all track)
        remaining = self._find_remaining_curves()
        if remaining:
            track = TrackWidget("Lainnya", width=160)
            track.set_depth_range(top, bottom)
            for i, (mnemonic, values) in enumerate(remaining):
                color = CURVE_COLORS[i % len(CURVE_COLORS)]
                track.plot_curve(depth, values, color=color, name=mnemonic)
            track.plot_widget.setYLink(self._depth_track.plot_widget)
            track.plot_widget.getAxis("left").hide()
            self._track_layout.addWidget(track)
            self._tracks.append(track)

        self._track_layout.addStretch()

    def _find_curves(self, mnemonics: list[str]) -> list[tuple[str, np.ndarray]]:
        found = []
        for m in mnemonics:
            arr = self._well_data.get_curve(m)
            if arr is not None:
                found.append((m, arr))
                break  # ambil satu saja dari daftar alias
        return found

    def _find_remaining_curves(self) -> list[tuple[str, np.ndarray]]:
        all_defined = set()
        for tdef in TRACK_DEFS:
            all_defined.update(c.upper() for c in tdef["curves"])
            if "secondary" in tdef:
                all_defined.update(c.upper() for c in tdef["secondary"]["curves"])

        remaining = []
        for curve in self._well_data.curves:
            if curve.mnemonic.upper() not in all_defined:
                arr = self._well_data.get_curve(curve.mnemonic)
                if arr is not None:
                    remaining.append((curve.mnemonic, arr))
        return remaining

    def clear(self):
        self._well_data = None
        self._rebuild_tracks()
