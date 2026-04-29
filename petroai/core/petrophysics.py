import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from core.las_reader import WellData
import config


@dataclass
class PetroParams:
    gr_sand: float = config.GR_SAND
    gr_shale: float = config.GR_SHALE
    rw: float = config.RW
    a: float = config.A
    m: float = config.M
    n: float = config.N
    rhob_ma: float = 2.65    # matrix density (sandstone)
    rhob_fl: float = 1.0     # fluid density
    nphi_ma: float = 0.0     # matrix neutron
    nphi_fl: float = 1.0     # fluid neutron
    sw_cutoff: float = 0.5   # SW max untuk net pay
    phie_cutoff: float = 0.08 # porosity min untuk net pay
    vsh_cutoff: float = 0.5  # Vshale max untuk net pay


@dataclass
class ZoneResult:
    name: str
    top: float
    bottom: float
    thickness: float
    avg_gr: float
    avg_rt: float
    avg_nphi: float
    avg_rhob: float
    avg_vsh: float
    avg_phie: float
    avg_sw: float
    net_pay: float
    is_reservoir: bool
    fluid_type: str
    quality: str

    def to_dict(self) -> dict:
        return {
            "Zone": self.name,
            "Top (m)": round(self.top, 1),
            "Bottom (m)": round(self.bottom, 1),
            "Thickness (m)": round(self.thickness, 1),
            "Avg GR": round(self.avg_gr, 1),
            "Avg RT": round(self.avg_rt, 2),
            "Avg Vsh": round(self.avg_vsh, 3),
            "Avg PHIE": round(self.avg_phie, 3),
            "Avg SW": round(self.avg_sw, 3),
            "Net Pay (m)": round(self.net_pay, 1),
            "Fluid": self.fluid_type,
            "Quality": self.quality,
        }


@dataclass
class PetroResult:
    depth: np.ndarray
    vsh: np.ndarray
    phie: np.ndarray
    sw: np.ndarray
    net_pay_flag: np.ndarray   # 1 = net pay, 0 = not
    zones: list[ZoneResult] = field(default_factory=list)
    params: PetroParams = field(default_factory=PetroParams)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "DEPTH": self.depth,
            "VSH": self.vsh,
            "PHIE": self.phie,
            "SW": self.sw,
            "NET_PAY": self.net_pay_flag,
        }).set_index("DEPTH")


class PetrophysicsEngine:
    def __init__(self, params: PetroParams | None = None):
        self.params = params or PetroParams()

    # --------------------------------------------------------- Vshale (GR)
    def calc_vsh_gr(self, gr: np.ndarray) -> np.ndarray:
        p = self.params
        vsh = (gr - p.gr_sand) / (p.gr_shale - p.gr_sand)
        return np.clip(vsh, 0.0, 1.0)

    # --------------------------------------------------------- Vshale (Larinov - tertiary)
    def calc_vsh_larinov(self, igr: np.ndarray) -> np.ndarray:
        return np.clip(0.083 * (2.0 ** (3.7 * igr) - 1.0), 0.0, 1.0)

    # --------------------------------------------------------- Porosity (density)
    def calc_phid(self, rhob: np.ndarray) -> np.ndarray:
        p = self.params
        phid = (p.rhob_ma - rhob) / (p.rhob_ma - p.rhob_fl)
        return np.clip(phid, 0.0, 0.45)

    # --------------------------------------------------------- Porosity (neutron)
    def calc_phin(self, nphi: np.ndarray) -> np.ndarray:
        return np.clip(nphi, 0.0, 0.45)

    # --------------------------------------------------------- Porosity (NPHI-RHOB crossplot)
    def calc_phie(self, nphi: np.ndarray, rhob: np.ndarray,
                  vsh: np.ndarray | None = None) -> np.ndarray:
        phin = self.calc_phin(nphi)
        phid = self.calc_phid(rhob)
        phit = (phin + phid) / 2.0

        if vsh is not None:
            # shale correction (Gaymard-Poupon)
            nphi_sh = np.nanmean(phin[vsh > 0.7]) if np.any(vsh > 0.7) else 0.3
            rhob_sh = np.nanmean(rhob[vsh > 0.7]) if np.any(vsh > 0.7) else 2.45
            phid_sh = self.calc_phid(np.full_like(rhob, rhob_sh))
            phit = phit - vsh * (nphi_sh + phid_sh[0]) / 2.0

        return np.clip(phit, 0.0, 0.45)

    # --------------------------------------------------------- Water Saturation (Archie)
    def calc_sw_archie(self, rt: np.ndarray, phie: np.ndarray) -> np.ndarray:
        p = self.params
        with np.errstate(divide="ignore", invalid="ignore"):
            sw = ((p.a * p.rw) / (rt * (phie ** p.m))) ** (1.0 / p.n)
        sw = np.where(np.isfinite(sw), sw, np.nan)
        return np.clip(sw, 0.0, 1.0)

    # --------------------------------------------------------- Water Saturation (Simandoux)
    def calc_sw_simandoux(self, rt: np.ndarray, phie: np.ndarray,
                          vsh: np.ndarray, rt_sh: float = 4.0) -> np.ndarray:
        p = self.params
        A = phie ** p.m / (p.a * p.rw)
        B = vsh / rt_sh
        with np.errstate(divide="ignore", invalid="ignore"):
            sw = (A / 2.0) * (np.sqrt(B ** 2 + 4.0 * A / rt) - B)
        sw = np.where(np.isfinite(sw), sw, np.nan)
        return np.clip(sw, 0.0, 1.0)

    # --------------------------------------------------------- Net Pay Flag
    def calc_net_pay(self, vsh: np.ndarray, phie: np.ndarray,
                     sw: np.ndarray) -> np.ndarray:
        p = self.params
        flag = (
            (vsh <= p.vsh_cutoff) &
            (phie >= p.phie_cutoff) &
            (sw <= p.sw_cutoff) &
            (~np.isnan(vsh)) &
            (~np.isnan(phie)) &
            (~np.isnan(sw))
        )
        return flag.astype(float)

    # --------------------------------------------------------- Auto Zone Detection
    def detect_zones(self, depth: np.ndarray, vsh: np.ndarray,
                     step: float = 0.5) -> list[tuple[float, float, str]]:
        """Deteksi zona reservoir vs shale secara otomatis."""
        zones = []
        in_reservoir = False
        zone_top = depth[0]
        SMOOTH = 5  # smoothing window

        vsh_smooth = pd.Series(vsh).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()

        for i, (d, v) in enumerate(zip(depth, vsh_smooth)):
            is_res = not np.isnan(v) and v < self.params.vsh_cutoff
            if is_res and not in_reservoir:
                zone_top = d
                in_reservoir = True
            elif not is_res and in_reservoir:
                thickness = d - zone_top
                if thickness >= 1.0:
                    zones.append((zone_top, d, "reservoir"))
                in_reservoir = False
            elif is_res is False and not in_reservoir:
                pass

        if in_reservoir:
            thickness = depth[-1] - zone_top
            if thickness >= 1.0:
                zones.append((zone_top, depth[-1], "reservoir"))

        return zones

    # --------------------------------------------------------- Full Analysis
    def analyze(self, well: WellData) -> PetroResult:
        depth = well.depth
        p = self.params

        def first(*mnemonics):
            for m in mnemonics:
                arr = well.get_curve(m)
                if arr is not None:
                    return arr
            return None

        # kurva dasar
        gr = first("GR")
        rt = first("RT", "ILD", "LLD", "RDEEP")
        nphi = first("NPHI", "TNPH", "CNPHI")
        rhob = first("RHOB", "ZDEN", "DEN")

        # Vshale
        if gr is not None:
            igr = np.clip((gr - p.gr_sand) / (p.gr_shale - p.gr_sand), 0, 1)
            vsh = self.calc_vsh_larinov(igr)
        else:
            vsh = np.full(len(depth), np.nan)

        # Porosity
        if nphi is not None and rhob is not None:
            phie = self.calc_phie(nphi, rhob, vsh if gr is not None else None)
        elif rhob is not None:
            phie = self.calc_phid(rhob)
        elif nphi is not None:
            phie = self.calc_phin(nphi)
        else:
            phie = np.full(len(depth), np.nan)

        # Water saturation
        if rt is not None and not np.all(np.isnan(phie)):
            sw = self.calc_sw_archie(rt, phie)
        else:
            sw = np.full(len(depth), np.nan)

        # Net pay
        net_pay = self.calc_net_pay(vsh, phie, sw)

        # Zone analysis
        zone_defs = self.detect_zones(depth, vsh, well.depth_step)
        zones = self._evaluate_zones(
            zone_defs, depth, gr, rt, nphi, rhob, vsh, phie, sw, net_pay, well.depth_step
        )

        return PetroResult(depth=depth, vsh=vsh, phie=phie, sw=sw,
                           net_pay_flag=net_pay, zones=zones, params=p)

    def _evaluate_zones(self, zone_defs, depth, gr, rt, nphi, rhob,
                        vsh, phie, sw, net_pay, step) -> list[ZoneResult]:
        results = []
        for i, (top, bottom, kind) in enumerate(zone_defs):
            mask = (depth >= top) & (depth <= bottom)
            if mask.sum() < 2:
                continue

            def avg(arr):
                if arr is None:
                    return np.nan
                sub = arr[mask]
                valid = sub[~np.isnan(sub)]
                return float(valid.mean()) if len(valid) > 0 else np.nan

            a_gr = avg(gr)
            a_rt = avg(rt)
            a_nphi = avg(nphi)
            a_rhob = avg(rhob)
            a_vsh = avg(vsh)
            a_phie = avg(phie)
            a_sw = avg(sw)
            net = float(net_pay[mask].sum()) * step

            # fluid classification
            if np.isnan(a_sw):
                fluid = "Tidak Diketahui"
            elif a_sw < 0.3:
                fluid = "Hidrokarbon (Gas/Minyak)"
            elif a_sw < 0.5:
                fluid = "Minyak / Air Campuran"
            elif a_sw < 0.7:
                fluid = "Air Campuran"
            else:
                fluid = "Air"

            # reservoir quality
            if np.isnan(a_phie):
                quality = "Tidak Diketahui"
            elif a_phie > 0.2:
                quality = "Sangat Baik"
            elif a_phie > 0.15:
                quality = "Baik"
            elif a_phie > 0.1:
                quality = "Sedang"
            else:
                quality = "Buruk"

            results.append(ZoneResult(
                name=f"Zone-{i+1}",
                top=top, bottom=bottom,
                thickness=bottom - top,
                avg_gr=a_gr, avg_rt=a_rt,
                avg_nphi=a_nphi, avg_rhob=a_rhob,
                avg_vsh=a_vsh, avg_phie=a_phie, avg_sw=a_sw,
                net_pay=net,
                is_reservoir=(kind == "reservoir"),
                fluid_type=fluid,
                quality=quality,
            ))
        return results
