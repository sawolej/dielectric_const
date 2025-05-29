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
    # r"C:/Users/kamil/Documents/1. KPO/notatki/glowInTheDark/glow-powietrze-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/glowInTheDark/glow-woda-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/glowInTheDark/glow100-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/glowInTheDark/glow25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/glowInTheDark/glow50-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/glowInTheDark/glow50naswietlone-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/glowInTheDark/glow75-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/glowInTheDark/pla25cienkie-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/glowInTheDark/pla25gruby-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/PLA100-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/PLA25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/PLA50-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/PLA75-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/PLA_aceton-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/PLA_hips-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/PLA_poli-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/PLA_powietrze-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/PLA_woda-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/PLA/pow/PLA_powietrze+stojak-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA100-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA50-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA75-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA_aceton-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA_hips-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA_poli-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA_pow-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA_pow2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/1. KPO/notatki/HS PLA/HS_PLA_wod-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/aloes/sma/PLA25_aloes-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/aloes/sma/PLA50_aloes-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/aloes/sma/PLA75_aloes-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/aloes/sma/PLA_puste_aloes-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/BioCreate/BioCREATE25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/BioCreate/BioCREATE50-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/Copper/Copper25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/Copper/Copper75-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/Iron/Iron25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/Iron/Iron75-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/n/PLA25_aloes_n-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/n/PLA25_myskinbooster_n-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/n/PLA50_aloes_n-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/n/PLA50_myskinbooster_n-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/n/PLA75_aloes_n-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/n/PLA75_myskinbooster_n-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/myskinbooster/sma/BioCREATE25_myskinbooster-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/myskinbooster/sma/PLA25_myskinbooster-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/myskinbooster/sma/PLA50_myskinbooster-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/myskinbooster/sma/PLA75_myskiinbooster-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/23.05/myskinbooster/sma/PLA_puste_myskinbooster-S11-VF.s1p",
    #pico106
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/106Test/2001 pkt/test106_probe_cooper.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/106Test/2001 pkt/test106_probe_hips.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/106Test/2001 pkt/test106_probe_iron.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/106Test/2001 pkt/test106_probe_powietrze.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/106Test/2001 pkt/test106_probe_woda.s1p",
    #N
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/n_hips-S11-VF.s1p",
    # # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/n_poli-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/n_powietrze-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/n_woda-S11-VF.s1p",
    # # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/n_wod.s1p",

    # # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/BioCreate/n_biocreate100-S11-VF.s1p",
    # # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/BioCreate/n_biocreate25-S11-VF.s1p",
    # # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/BioCreate/n_biocreate50-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/Copper/n_copper100-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/Copper/n_copper25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/Copper/n_copper75-S11-VF.s1p",
    # # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/HS PLA/n_hspla100-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/HS PLA/n_hspla25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/HS PLA/n_hspla50-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/HS PLA/n_hspla75-S11-VF.s1p",
    # ##### r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/Iron/n_iron100-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/Iron/n_iron25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/Iron/n_iron75-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/PLA/n_pla100-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/PLA/n_pla25-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/PLA/n_pla50-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/PLA/n_pla75-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/PLA+aloes/n_pla25+aloes-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/PLA+aloes/n_pla50+aloes-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/PLA+aloes/n_pla75+aloes-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/26.05 (pomiary na n)/PLA+myskinbooster/n_pla25+myskinbooster-S11-VF.s1p",

    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25_hips-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25_poli-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25_powietrze-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25_woda-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla15+woda/pla15_woda-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla15+woda/pla15_woda2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla15+woda/pla15_woda3-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla15+woda/pla15_woda4-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla15+woda/pla15_woda5-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes1-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes10-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes3-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes4-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes5-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes6-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes7-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes8-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/pla25+aloes/pla25_aloes9-S11-VF.s1p",

    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/oplatek/pla25_aloes_oplatek-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/oplatek/pla25_aloes_oplatek2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/27.05 (n)/oplatek/pla25_aloes_oplatek3-S11-VF.s1p",

    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/powietrze-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/pp1gnp10-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/pp1gnp10_2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/pp2gnp10-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/pp2gnp10_2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/pur12-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/pur12_2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/refpp1-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/refpp2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/uhwmpe_gnp2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/28.05/stala/woda-S11-VF.s1p",

    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/copper100-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/copper100_napewno-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/copper75_1-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/copper75_napewno-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/ECG-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/iron100-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/powietrze-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/sigma_gel1-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/sigma_gel2-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/spirol-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/USG-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/woda-destylowana-S11-VF.s1p",
    # r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/29.05/woda-kranowa-S11-VF.s1p",

    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/ASA_ESD_202510-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/ASA_ESD_202520-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/ASA_ESD_202530-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/ASA_ESD_203040-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/ASA_ESD_203050-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/ASA_ESD_203060-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_100000-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_100001-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_100002-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_102510-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_102520-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_102530-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_103040-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_103050-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_103060-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_104070-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_104080-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PETg_ESD_104090-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PLA_ESD_300510-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PLA_ESD_300530-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/PLA_ESD_500520-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pomiaryxd69xd/próbki rosa/powietrze-S11-VF.s1p",
    
    ]

TEMPERATURE_C         = 25.0
# **dokładnie taki gating jak w Twoim przykładzie:**
GATE_CENTER_NS        = 5.2    # środek okna w [ns]
GATE_SPAN_NS          = 1    # bardzo wąskie okno!
Z0                    = 50.0
CABLE_LENGTH_M        = 0.5363
SIGNAL_DELAY_NS_PER_M = 4.86
TARGET_GHZ            = [0.8, 1.000, 1.3, 1.6, 2.0, 2.39]

OUTPUT_DIR_ROOT       = Path.cwd()
OUTPUT_CSV            = OUTPUT_DIR_ROOT / "epsilon_full.csv"

# ---------------------------------------------------------------------
# FUNKCJE pomocnicze
# ---------------------------------------------------------------------
def load_s1p_decimal_fix(path: Union[str, Path]) -> rf.Network:
    """
    Wczytuje .s1p nawet z przecinkami. Tworzy plik tymczasowy w trybie binarnym,
    do którego zapisuje linie przekonwertowane na UTF-8.
    """
    path = Path(path)
    # odczyt całego oryginału jako tekst
    text = path.read_text(encoding="utf-8")
    # zamiana przecinków na kropki tylko w liniach danych
    lines = []
    for ln in text.splitlines(keepends=True):
        if ln.lstrip().startswith(('#','!')):
            lines.append(ln)
        else:
            lines.append(ln.replace(',', '.'))
    # zapis do pliku binarnego
    with tempfile.NamedTemporaryFile(mode="w+b", delete=False, suffix=".s1p") as fo:
        for ln in lines:
            fo.write(ln.encode("utf-8"))
        tmp_name = fo.name
    # teraz skrf.Network odczyta go poprawnie
    return rf.Network(tmp_name)


def debye_water_eps(f_Hz, T_C=25.0):
    eps_inf  = 4.9
    eps_stat = 78.54*(1 - 0.004579*(T_C-25) + 0.000019*(T_C-25)**2)
    tau      = (1.1109e-10 - 3.824e-12*T_C + 6.938e-14*T_C**2 - 5.096e-16*T_C**3)
    w        = 2*np.pi*f_Hz
    return eps_inf + (eps_stat-eps_inf)/(1+1j*w*tau)

def s11_to_Y(s11, Z0=50.0):
    return (1 - s11)/(1 + s11)/Z0

# ---------------------------------------------------------------------
# KATALOG wyników
# ---------------------------------------------------------------------
stamp = datetime.datetime.now().strftime("TG%Y%m%d_%H%M%S")
outdir = OUTPUT_DIR_ROOT / stamp
outdir.mkdir()

# delay [s] dla de-embedingu
tof_one_way = SIGNAL_DELAY_NS_PER_M * CABLE_LENGTH_M * 1e-9
RT_DELAY    = 2 * tof_one_way

# ---------------------------------------------------------------------
# 1) KALIBRACJA (powietrze / woda)
# ---------------------------------------------------------------------
net_air   = load_s1p_decimal_fix([p for p in FILES if "powietrze" in p.lower()][0])
net_water = load_s1p_decimal_fix([p for p in FILES if "woda"    in p.lower()][0])

# **1a) najpierw time-gate – bez de-embeddingu**  
s11g_air   = net_air.s11.time_gate(center=GATE_CENTER_NS, span=GATE_SPAN_NS)
s11g_water = net_water.s11.time_gate(center=GATE_CENTER_NS, span=GATE_SPAN_NS)
# zapis .s1p
# s11g_air.write(outdir/"air_S11_gated.s1p")
# s11g_water.write(outdir/"water_S11_gated.s1p")

# **1b) potem de-embedingu fazy**  
f = s11g_air.f
phase = np.exp(1j*2*np.pi*f*RT_DELAY)[:,None,None]

net_air_corr   = net_air.copy();   net_air_corr.s = s11g_air.s * phase
net_water_corr = net_water.copy(); net_water_corr.s = s11g_water.s * phase
# zapis poprawionych
# net_air_corr.write_touchstone(str(outdir/"air_S11_gated_corr"))
# net_water_corr.write_touchstone(str(outdir/"water_S11_gated_corr"))

# 1c) oblicz C0 i Cp po gatingu + de-embed  
w     = 2*np.pi * f
Y_air = s11_to_Y(net_air_corr.s.squeeze(), Z0)
Y_w   = s11_to_Y(net_water_corr.s.squeeze(), Z0)
eps_w = debye_water_eps(f, TEMPERATURE_C)
C0    = (Y_w.imag - Y_air.imag)/(w*(eps_w.real-1))
Cp    = Y_air.imag/w - C0

print("\nStałe kalibracyjne:")
for tg in TARGET_GHZ:
    idx = np.argmin(np.abs(f/1e9 - tg))
    print(f"{f[idx]/1e9:.3f} GHz → C0={C0[idx]:.3e}  Cp={Cp[idx]:.3e}")

# ---------------------------------------------------------------------
# 2) PRÓBKI
# ---------------------------------------------------------------------
spectra_zin, spectra_eps, eps_results = {}, {}, []

for p in FILES:
    name = Path(p).stem
    if "powietrze" in name.lower() or "woda" in name.lower():
        continue

    net = load_s1p_decimal_fix(p)

    # **2a) gating na oryginalnym S11**  
    s11g = net.s11.time_gate(center=GATE_CENTER_NS, span=GATE_SPAN_NS)
    save_path = outdir / f"{name}_S11_gated.s1p"
    # s11g.write(str(save_path))

    # **2b) de-embed fazy**  
    phase = np.exp(1j*2*np.pi*s11g.f*RT_DELAY)[:,None,None]
    net_corr = net.copy()
    net_corr.s = s11g.s * phase
    # net_corr.write_touchstone(str(outdir/f"{name}_S11_gated_corr"))

    # |Zin|
    gamma_tip = net_corr.s.flatten()
    zin_abs   = np.abs(Z0*(1+gamma_tip)/(1-gamma_tip))
    spectra_zin[name] = (s11g.f/1e9, zin_abs)

    # ε′
    Y_samp   = s11_to_Y(gamma_tip, Z0)
    eps_samp = (Y_samp/(1j*w) - Cp)/C0
    spectra_eps[name] = (s11g.f/1e9, eps_samp.real)

    eps_results_imag = []

    # wyniki pod TARGET_GHZ
    for tg in TARGET_GHZ:
        ix = np.argmin(np.abs(s11g.f/1e9 - tg))
        eps_results.append({
            "sample":   name,
            "freq_GHz": tg,
            "Zin_Ω":    zin_abs[ix],
            "epsilon":  eps_samp.real[ix]
        })
        eps_results_imag.append({
            "sample":  name,
            "freq_GHz": tg,
            "epsilon_imag": eps_samp.imag[ix]
        })

    # **opcjonalnie**: rysuj raw vs gated jak w Twoim przykładzie
    fig,(axf, axt) = plt.subplots(1,2,figsize=(10,4))
    net.s11.plot_s_db(ax=axf,label="raw")
    s11g.plot_s_db(ax=axf,label="gated")
    axf.set_title(f"{name} – S11 dB(f)"); axf.legend()
    net.s11.plot_s_db_time(ax=axt,label="raw")
    s11g.plot_s_db_time(ax=axt,label="gated")
    axt.set_xlim(0,50); axt.set_title(f"{name} – S11 dB(t)"); axt.legend()
    fig.tight_layout(); fig.savefig(outdir/f"{name}_raw_vs_gated.png"); plt.close(fig)

# ---------------------------------------------------------------------
# 3) Podsumowanie: wszystkie wykresy i tabele
# ---------------------------------------------------------------------
# |Zin|
fig, ax = plt.subplots(figsize=(8,6))
for k,(fg,zin) in spectra_zin.items():
    ax.plot(fg, zin, label=k)
ax.set(title="|Zin| – wszystkie próbki", xlabel="GHz", ylabel="Ω")
ax.legend(); ax.grid(True); fig.savefig(outdir/"all_Zin.png"); 
plt.show()
plt.close(fig)

# Re(ε)
fig, ax = plt.subplots(figsize=(8,6))
for k,(fg,epsr) in spectra_eps.items():
    ax.plot(fg, epsr, label=k)
ax.set(title="Re(ε) – wszystkie próbki", xlabel="GHz", ylabel="Re(ε)")
ax.legend(); ax.grid(True); fig.savefig(outdir/"all_eps.png");
plt.show()
plt.close(fig)

# ---------------------------------------------------------------------
# 3) CSV i pivot + odchylenia
# ---------------------------------------------------------------------
df = pd.DataFrame(eps_results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nZapis ε' → {OUTPUT_CSV}")

# pivot tylko dla TARGET_GHZ
pt = df.pivot_table(index="sample",
                    columns="freq_GHz",
                    values="epsilon")

# # std dla wybranych freq (tak jak wcześniej)
# pt["std_sel"] = pt[TARGET_GHZ].std(axis=1)

# # std po całym widmie:
# # spectra_eps[name] = (f_vec, eps_vec) z epoki 2)
# std_all = {}
# for name, (f_vec, eps_vec) in spectra_eps.items():
#     # wybieramy tylko te punkty, które leżą w interesującym nas paśmie 0.8–3 GHz
#     mask = (f_vec >= 0.8) & (f_vec <= 3.0)
#     std_all[name] = np.std(eps_vec[mask])

# # dodajemy std_all do pivotu
# pt["std_all"] = pt.index.map(std_all)

# wyświetlamy wynik
print("\nTabela ε' + odchylenie (całe widmo i wybrane):")
print(pt)

# ---------------------------------------------------------------------
# 2c) STATYSTYKA dla każdej próbki
#    • mean ε′   (uśrednione w paśmie 0.8-3 GHz)
#    • mean |Zin|
#    • std ε′
# ---------------------------------------------------------------------
stats = []
LOW, HIGH = 0.8, 3.0      # pasmo do uśredniania [GHz]

for name, (f_vec, eps_real) in spectra_eps.items():
    zin_vec = spectra_zin[name][1]

    mask = (f_vec >= LOW) & (f_vec <= HIGH)
    stats.append({
        "sample"      : name,
        "mean_eps"    : eps_real[mask].mean(),
        "std_eps"     : eps_real[mask].std(ddof=1),   # odchylenie standardowe
        "mean_Zin_Ω"  : zin_vec[mask].mean()
    })

stats_df = (
    pd.DataFrame(stats)
      .set_index("sample")
      .round(3)              # zaokrąglij dla czytelności
      .sort_index()          # kolejność alfabetyczna
)

print("\nStatystyka 0.8–3 GHz:")
print(stats_df)
stats_df.to_csv(outdir / "stats_summary.csv")
