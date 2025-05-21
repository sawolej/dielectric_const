# -*- coding: utf-8 -*-
"""
Time-gating + de-embedding + pełna permittivity ε – zapis wszystkich wykresów
wersja 2025-05-20 — AUTO-fit fazy lub RT_DELAY = 4.2 ns
"""

from __future__ import annotations
import io, re, datetime, tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import skrf as rf      # pip install scikit-rf
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PLIK I PARAMETRY
# ---------------------------------------------------------------------
FILES = [
    # kalibracyjne
    r"C:/Users/kamil/Documents/1. KPO/notatki/stałe/glow-powietrze-S11-VF.s1p",
    r"C:/Users/kamil/Documents/1. KPO/notatki/stałe/glow-woda-S11-VF.s1p",
    # próbki
    r"C:/Users/kamil/Documents/1. KPO/notatki/stałe/glow100-S11-VF.s1p",
    r"C:/Users/kamil/Documents/1. KPO/notatki/stałe/glow25-S11-VF.s1p",
    r"C:/Users/kamil/Documents/1. KPO/notatki/stałe/glow50-S11-VF.s1p",
    r"C:/Users/kamil/Documents/1. KPO/notatki/stałe/glow50naswietlone-S11-VF.s1p",
    r"C:/Users/kamil/Documents/1. KPO/notatki/stałe/glow75-S11-VF.s1p",
    r"C:/Users/kamil/Documents/1. KPO/notatki/stałe/pla25cienkie-S11-VF.s1p",
    r"C:/Users/kamil/Documents/1. KPO/notatki/stałe/pla25gruby-S11-VF.s1p",
    # tu możesz dodać kolejne pliki .s1p
]

TEMPERATURE_C        = 25.0       # °C dla modelu Debye
OUTPUT_DIR_ROOT      = Path.cwd()
OUTPUT_CSV           = OUTPUT_DIR_ROOT / "epsilon_full.csv"

# ---------------------------------------------------------------------
# CZAS BRAMKOWANIA I DE-EMBEDDING
# ---------------------------------------------------------------------
GATE_CENTER_NS        = 5.0
GATE_SPAN_NS          = 2
Z0                    = 50.0
AUTO_FIT_DELAY        = False     # True → fit fazy; False → stały RT_DELAY
CABLE_LENGTH_M        = 0.515     # [m]
SIGNAL_DELAY_NS_PER_M = 4.86      # [ns/m]
TARGET_GHZ            = [ 0.99, 1.000, 1.223]

# ---------------------------------------------------------------------
# MODEL DEBYE i admitancja
# ---------------------------------------------------------------------
def debye_water_eps(f_Hz: np.ndarray, T_C: float = 25.0) -> np.ndarray:
    eps_inf   = 4.9
    eps_stat  = 78.54 * (1 - 0.004579*(T_C-25) + 0.000019*(T_C-25)**2)
    tau       = (1.1109e-10 - 3.824e-12*T_C + 6.938e-14*T_C**2 - 5.096e-16*T_C**3)
    w         = 2*np.pi * f_Hz
    return eps_inf + (eps_stat - eps_inf) / (1 + 1j*w*tau)

def s11_to_Y(s11: np.ndarray, Z0: float = 50) -> np.ndarray:
    return (1 - s11) / (1 + s11) / Z0

# ---------------------------------------------------------------------
# ŁADOWANIE .s1p Z NAPRAWĄ PRZECINKÓW
# ---------------------------------------------------------------------
def load_s1p_decimal_fix(path: Union[str, Path]) -> rf.Network:
    path = Path(path)
    with open(path, encoding="utf-8") as f_in, \
         tempfile.NamedTemporaryFile("w+", delete=False, suffix=".s1p") as f_out:
        for line in f_in:
            if line.lstrip().startswith(('!','#')):
                f_out.write(line)
            else:
                f_out.write(line.replace(',', '.'))
        tmp_name = f_out.name
    return rf.Network(tmp_name)

# ---------------------------------------------------------------------
# PRZYGOTOWANIE WYJŚCIOWEGO KATALOGU
# ---------------------------------------------------------------------
timestamp = datetime.datetime.now().strftime("TIMEGATING%Y%m%d_%H%M%S")
output_dir = OUTPUT_DIR_ROOT / timestamp
output_dir.mkdir(exist_ok=True)
print(f"Zapisuję wszystkie wykresy w: {output_dir}")

# ---------------------------------------------------------------------
# KALIBRACJA: powietrze vs woda
# ---------------------------------------------------------------------
net_air   = load_s1p_decimal_fix([p for p in FILES if "powietrze" in p.lower()][0])
net_water = load_s1p_decimal_fix([p for p in FILES if "woda"    in p.lower()][0])

if not np.allclose(net_air.f, net_water.f):
    raise RuntimeError("Kalibracyjne pliki .s1p mają różne osie częstotliwości!")

f       = net_air.f
w       = 2*np.pi * f
eps_w   = debye_water_eps(f, TEMPERATURE_C)
Y_air   = s11_to_Y(net_air.s.squeeze(),   Z0)
Y_w     = s11_to_Y(net_water.s.squeeze(), Z0)

C0 = (Y_w.imag - Y_air.imag) / (w * (eps_w.real - 1))
Cp = Y_air.imag / w - C0

# ---------------------------------------------------------------------
# PRZETWARZANIE PRÓBEK
# ---------------------------------------------------------------------
spectra     = {}   # do wspólnego wykresu |Zin|
eps_results = []   # pełna ε (complex) w punktach TARGET_GHZ

for path in FILES:
    low = path.lower()
    if "powietrze" in low or "woda" in low:
        continue    # pomiń pliki kalibracyjne

    net   = load_s1p_decimal_fix(path)
    s11   = net.s11
    s11_g = s11.time_gate(center=GATE_CENTER_NS, span=GATE_SPAN_NS)

     # --- zapisujemy gated S11 do pliku .s1p ---
    # (skrf.Network, więc ma metodę .write)
    sample = Path(path).stem
    gated_s1p_path = output_dir / f"{sample}_S11_gated.s1p"
    s11_g.write(str(gated_s1p_path))

    # oblicz RT_DELAY
    f_g    = s11_g.f
    gamma  = s11_g.s.flatten()
    if AUTO_FIT_DELAY:
        phase, = np.unwrap(np.angle(gamma)), 
        slope, _ = np.polyfit(f_g, phase, 1)
        RT_DELAY = -slope / (2*np.pi)
    else:
        tof_one_way = SIGNAL_DELAY_NS_PER_M * CABLE_LENGTH_M * 1e-9
        RT_DELAY    = 2 * tof_one_way

    # de-embedding |Zin|
    gamma_tip = gamma * np.exp(1j * 2*np.pi * f_g * RT_DELAY)
    zin_abs   = np.abs(Z0 * (1 + gamma_tip) / (1 - gamma_tip))
    freqs_ghz = f_g / 1e9
    sample    = Path(path).stem
    spectra[sample] = (freqs_ghz, zin_abs)

    # pełne ε (complex)
    Y_samp    = s11_to_Y(s11_g.s.squeeze(), Z0)
    eps_samp  = (Y_samp/(1j*w) - Cp) / C0

    for tg in TARGET_GHZ:
        idx = np.argmin(np.abs(freqs_ghz - tg))
        eps_real = eps_samp[idx].real      # <-- wyciągamy tylko część rzeczywistą
        eps_results.append({
            "sample":   sample,
            "freq_GHz": freqs_ghz[idx],
            "Zin_Ω":    zin_abs[idx],
            "epsilon":  eps_real             # teraz tylko ε′
        })

    # wykres raw vs gated
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
    s11.plot_s_db(ax=ax1,   label="raw")
    s11_g.plot_s_db(ax=ax1, label="gated")
    ax1.set_title(f"{sample} – S11 dB(f)");   ax1.legend()
    s11.plot_s_db_time(ax=ax2,   label="raw")
    s11_g.plot_s_db_time(ax=ax2, label="gated")
    ax2.set_xlim(0,40);  ax2.set_title(f"{sample} – S11 dB(t)");  ax2.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{sample}_S11_raw_gated.png")
    plt.close(fig)

# ---------------------------------------------------------------------
# WYKRES WSZYSTKICH SPEKTR |Zin|
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10,6))
for sample,(fghz, zin) in spectra.items():
    ax.plot(fghz, zin, label=sample)
ax.set_title("|Zin| – wszystkie próbki")
ax.set_xlabel("Częstotliwość [GHz]")
ax.set_ylabel("|Zin| [Ω]")
ax.legend(); ax.grid(True); fig.tight_layout()
fig.savefig(output_dir / "all_samples_Zin_vs_freq.png")
plt.close(fig)

# ---------------------------------------------------------------------
# ZAPIS PEŁNEJ PERMITTYWNOŚCI
# ---------------------------------------------------------------------
df_eps = pd.DataFrame(eps_results)
df_eps.to_csv(OUTPUT_CSV, index=False)
print(f"✔ Wyniki pełnej ε zapisane do: {OUTPUT_CSV}")

# ---------------------------------------------------------------------
# WYDRUK TABELI ε I |Zin| DLA TARGET_GHZ (ε jako complex)
# ---------------------------------------------------------------------
print(df_eps.pivot_table(
    index="sample",
    columns="freq_GHz",
    values=["Zin_Ω","epsilon"]
))
