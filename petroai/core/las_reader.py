import lasio
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CurveInfo:
    mnemonic: str
    unit: str
    description: str
    min_val: float
    max_val: float
    mean_val: float
    nan_pct: float


@dataclass
class WellData:
    name: str
    field: str
    company: str
    location: str
    depth_top: float
    depth_bottom: float
    depth_step: float
    depth_unit: str
    df: pd.DataFrame                    # semua kurva sebagai DataFrame
    curves: list[CurveInfo] = field(default_factory=list)
    las_path: str = ""

    @property
    def depth(self) -> np.ndarray:
        return self.df.index.to_numpy()

    def get_curve(self, mnemonic: str) -> Optional[np.ndarray]:
        mnemonic = mnemonic.upper()
        for col in self.df.columns:
            if col.upper() == mnemonic:
                return self.df[col].to_numpy()
        return None

    def has_curve(self, mnemonic: str) -> bool:
        return self.get_curve(mnemonic) is not None

    def curve_names(self) -> list[str]:
        return list(self.df.columns)

    def summary(self) -> str:
        lines = [
            f"Well: {self.name}",
            f"Field: {self.field}",
            f"Company: {self.company}",
            f"Depth: {self.depth_top:.1f} – {self.depth_bottom:.1f} {self.depth_unit}",
            f"Step: {self.depth_step:.4f} {self.depth_unit}",
            f"Curves ({len(self.curves)}): {', '.join(c.mnemonic for c in self.curves)}",
        ]
        return "\n".join(lines)


class LASReader:
    NULL_VALUES = (-999.25, -9999.25, -999.0, 9999.0)

    def read(self, path: str) -> WellData:
        las = lasio.read(path, ignore_header_errors=True)
        df = las.df()

        # bersihkan null values
        for null in self.NULL_VALUES:
            df.replace(null, np.nan, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # pastikan index adalah depth
        df.index.name = "DEPT"
        df.columns = [c.upper() for c in df.columns]

        curves = self._extract_curve_info(las, df)

        well_name = las.well.get("WELL", lasio.HeaderItem(value="UNKNOWN")).value or "UNKNOWN"
        field_name = las.well.get("FLD", lasio.HeaderItem(value="-")).value or "-"
        company = las.well.get("COMP", lasio.HeaderItem(value="-")).value or "-"
        loc = las.well.get("LOC", lasio.HeaderItem(value="-")).value or "-"

        depth_arr = df.index.to_numpy()
        strt = float(depth_arr[0]) if len(depth_arr) > 0 else 0.0
        stop = float(depth_arr[-1]) if len(depth_arr) > 0 else 0.0
        step_val = float(las.well.get("STEP", lasio.HeaderItem(value=0.5)).value or 0.5)
        dept_unit = las.well.get("DEPT", lasio.HeaderItem(unit="M")).unit or "M"

        return WellData(
            name=str(well_name).strip(),
            field=str(field_name).strip(),
            company=str(company).strip(),
            location=str(loc).strip(),
            depth_top=strt,
            depth_bottom=stop,
            depth_step=step_val,
            depth_unit=dept_unit,
            df=df,
            curves=curves,
            las_path=path,
        )

    def _extract_curve_info(self, las, df: pd.DataFrame) -> list[CurveInfo]:
        infos = []
        curve_meta = {c.mnemonic.upper(): c for c in las.curves}

        for col in df.columns:
            arr = df[col].to_numpy(dtype=float)
            valid = arr[~np.isnan(arr)]
            nan_pct = 100.0 * np.isnan(arr).sum() / len(arr) if len(arr) > 0 else 100.0

            meta = curve_meta.get(col.upper())
            unit = meta.unit if meta else ""
            desc = meta.descr if meta else ""

            infos.append(CurveInfo(
                mnemonic=col,
                unit=unit,
                description=desc,
                min_val=float(valid.min()) if len(valid) > 0 else np.nan,
                max_val=float(valid.max()) if len(valid) > 0 else np.nan,
                mean_val=float(valid.mean()) if len(valid) > 0 else np.nan,
                nan_pct=nan_pct,
            ))
        return infos
