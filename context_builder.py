"""
Mengubah data sumur + hasil petrofisik → prompt yang bisa dimengerti LLM.
"""
import numpy as np
from core.las_reader import WellData
from core.petrophysics import PetroResult, ZoneResult

SYSTEM_PROMPT = """Kamu adalah ahli petrofisik senior dengan pengalaman lebih dari 20 tahun di industri migas.
Kamu memahami analisis well log, interpretasi formasi, evaluasi reservoir, dan geologi bawah permukaan.

Saat menjawab:
- Gunakan bahasa Indonesia yang jelas dan profesional
- Berikan interpretasi berbasis data yang tersedia
- Sebutkan ketidakpastian jika data tidak lengkap
- Gunakan terminologi petrofisik yang tepat
- Jika diminta rekomendasi, berikan dengan justifikasi ilmiah
- Format jawaban dengan rapi (gunakan poin atau tabel jika perlu)
"""


class ContextBuilder:
    def __init__(self):
        self._well: WellData | None = None
        self._petro: PetroResult | None = None

    def set_well(self, well: WellData):
        self._well = well

    def set_petro_result(self, result: PetroResult):
        self._petro = result

    def has_data(self) -> bool:
        return self._well is not None

    def build_system_message(self) -> dict:
        ctx = SYSTEM_PROMPT
        if self._well:
            ctx += f"\n\n{self._build_well_context()}"
        return {"role": "system", "content": ctx}

    def build_interpretation_prompt(self) -> list[dict]:
        """Prompt untuk auto-interpretasi penuh dari AI."""
        if not self._well:
            return []

        well_ctx = self._build_well_context()
        petro_ctx = self._build_petro_context() if self._petro else ""

        user_msg = f"""Berdasarkan data sumur berikut, lakukan interpretasi petrofisik lengkap:

{well_ctx}
{petro_ctx}

Berikan interpretasi meliputi:
1. **Ringkasan Data Sumur** — kondisi umum data log
2. **Evaluasi Litologi** — tipe batuan dominan berdasarkan log GR dan densitas
3. **Identifikasi Zona Reservoir** — zona dengan potensi hidrokarbon
4. **Analisis Fluida** — tipe fluida (gas/minyak/air) per zona
5. **Kualitas Reservoir** — porositas dan permeabilitas relatif
6. **Zona Pay Terbaik** — rekomendasi zona yang paling prospektif
7. **Rekomendasi** — langkah selanjutnya (DST, perforasi, dll)
"""
        return [
            self.build_system_message(),
            {"role": "user", "content": user_msg},
        ]

    def build_chat_messages(self, history: list[dict], user_message: str) -> list[dict]:
        """Tambahkan konteks data ke conversation history."""
        messages = [self.build_system_message()]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    # ---------------------------------------------------------------- private
    def _build_well_context(self) -> str:
        w = self._well
        curves = ", ".join(c.mnemonic for c in w.curves)
        lines = [
            f"=== DATA SUMUR ===",
            f"Nama Sumur: {w.name}",
            f"Field: {w.field}",
            f"Company: {w.company}",
            f"Kedalaman: {w.depth_top:.1f} – {w.depth_bottom:.1f} {w.depth_unit}",
            f"Interval: {w.depth_step:.3f} {w.depth_unit}",
            f"Kurva tersedia: {curves}",
            "",
            "=== STATISTIK KURVA LOG ===",
        ]

        for c in w.curves:
            if c.nan_pct < 95:
                lines.append(
                    f"{c.mnemonic} ({c.unit}): "
                    f"min={c.min_val:.3f}, max={c.max_val:.3f}, "
                    f"mean={c.mean_val:.3f}, "
                    f"data coverage={100-c.nan_pct:.0f}%"
                )

        return "\n".join(lines)

    def _build_petro_context(self) -> str:
        r = self._petro
        if r is None:
            return ""

        lines = [
            "",
            "=== HASIL KALKULASI PETROFISIK ===",
            f"Parameter Archie: a={r.params.a}, m={r.params.m}, n={r.params.n}",
            f"Rw={r.params.rw} ohm.m, GR sand={r.params.gr_sand}, GR shale={r.params.gr_shale}",
            "",
            f"Statistik Vshale: mean={np.nanmean(r.vsh):.3f}, "
            f"min={np.nanmin(r.vsh):.3f}, max={np.nanmax(r.vsh):.3f}",
            f"Statistik PHIE: mean={np.nanmean(r.phie):.3f}, "
            f"min={np.nanmin(r.phie):.3f}, max={np.nanmax(r.phie):.3f}",
            f"Statistik SW: mean={np.nanmean(r.sw):.3f}, "
            f"min={np.nanmin(r.sw):.3f}, max={np.nanmax(r.sw):.3f}",
            f"Total Net Pay: {r.net_pay_flag.sum() * 0.5:.1f} m (estimasi)",
            "",
        ]

        if r.zones:
            lines.append("=== ZONA YANG TERDETEKSI ===")
            for z in r.zones:
                lines.append(
                    f"{z.name}: {z.top:.1f}–{z.bottom:.1f} m "
                    f"(tebal={z.thickness:.1f}m) | "
                    f"Vsh={z.avg_vsh:.2f} PHIE={z.avg_phie:.3f} SW={z.avg_sw:.2f} | "
                    f"Fluida: {z.fluid_type} | Kualitas: {z.quality} | "
                    f"Net Pay: {z.net_pay:.1f} m"
                )

        return "\n".join(lines)
