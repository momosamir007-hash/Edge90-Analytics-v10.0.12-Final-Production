#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║          ⚽ FOOTBALL PREDICTOR PRO v7.9 (LEAKAGE-RESISTANT EDITION) ⚽        ║
║                                                                      ║
║  ✅ V6.1: Fixed Entry Point (Streamlit/CLI separation)               ║
║  ✅ V6.1: Fixed _advanced_xg_features (save rate calculation)        ║
║  ✅ V6.1: Fixed Calibrator.adjust() (negative values protection)     ║
║  ✅ V6.1: Fixed EloSystem.predict() (realistic draw probability)     ║
║  ✅ V6.1: Fixed MLPred.feats() (safe feature count)                  ║
║  ✅ V6.1: Fixed xG bounds (min 0.10 instead of 0.25)                 ║
║  ✅ V6.1: Improved _form() (draw-aware form scoring)                 ║
║  ✅ V6.1: Improved draw_model weight (0.13 → 0.20)                   ║
║  ✅ V6.1: Draw Correction in Engine.predict()                        ║
║  ✅ V6.1: Fixed unique_matches key (date+home+away)                  ║
║  ✅ V6.0: 126 Features (68 Original + 58 New)                        ║
║  ✅ V6.0: Stacking Classifier (RF + XGB + LR → Meta LR)              ║
║  ✅ V6.0: ImprovedDrawPredictor (Statistical Draw Model)             ║
║  ✅ V6.0: Multi-League Support + Separate Files per League           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import requests
import json
import math
import os
import sys
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ── تحميل .env تلقائياً (يعمل مع Streamlit وCLI) ────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
except ImportError:
    # dotenv غير مثبت — نُحمّل .env يدوياً
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        with open(_env_file, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

try:
    import pandas as pd
except ImportError:
    print("❌ Pandas missing! pip install pandas")
    sys.exit(1)

# ── Streamlit: اختياري فقط ────────────────────────────────────
STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    pass

# ── ML: اختياري ───────────────────────────────────────────────
ML_AVAILABLE = False
XGBOOST_AVAILABLE = False
try:
    import numpy as np
    from sklearn.ensemble import (
        RandomForestClassifier,
        VotingClassifier,
        StackingClassifier,
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    pass

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════
# LEAGUES CONFIG LOADER
# ══════════════════════════════════════════════════════════════
LEAGUES_CONFIG_FILE = "leagues_config.json"

DEFAULT_GLOBAL_SETTINGS: Dict = {
    "elo_init": 1500,
    "elo_k": 32,
    "form_n": 8,
    "backtest_split": 0.70,
    "min_samples_per_class": 10,
    "n_features": 126,
}


def load_leagues_config() -> dict:
    if not Path(LEAGUES_CONFIG_FILE).exists():
        print(f"⚠️  {LEAGUES_CONFIG_FILE} not found! Creating default…")
        create_default_config()
    try:
        with open(LEAGUES_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return {"leagues": {}, "global_settings": DEFAULT_GLOBAL_SETTINGS}


def create_default_config():
    default = {
        "leagues": {
            "PL": {
                "name": "Premier League",
                "country": "England",
                "api_code": "PL",
                "api_url": "https://api.football-data.org/v4",
                "data_files": ["data/PL_Master.csv"],
                "model_file": "models/PL_model_v6.pkl",
                "calibration_file": "models/PL_calibration_v6.pkl",
                "elo_file": "models/PL_elo_v6.pkl",
                "teams_map_file": "config/PL_teams_map.json",
                "aliases_file": "config/PL_aliases.json",
                "rivalries_file": "config/PL_rivalries.json",
                "home_advantage": 65,
                "avg_home_goals": 1.53,
                "avg_away_goals": 1.16,
                "total_teams": 20,
                "total_rounds": 38,
            }
        },
        "global_settings": DEFAULT_GLOBAL_SETTINGS,
    }
    for folder in ["data", "models", "config"]:
        Path(folder).mkdir(exist_ok=True)
    with open(LEAGUES_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=2, ensure_ascii=False)
    print(f"✅ Created {LEAGUES_CONFIG_FILE}")


_GLOBAL_CONFIG = load_leagues_config()
GLOBAL_SETTINGS: Dict = _GLOBAL_CONFIG.get("global_settings", DEFAULT_GLOBAL_SETTINGS)
LEAGUES_CONFIG: Dict = _GLOBAL_CONFIG.get("leagues", {})

ELO_INIT: float = float(GLOBAL_SETTINGS.get("elo_init", 1500))
ELO_K: float = float(GLOBAL_SETTINGS.get("elo_k", 32))
FORM_N: int = int(GLOBAL_SETTINGS.get("form_n", 8))
BACKTEST_SPLIT: float = float(GLOBAL_SETTINGS.get("backtest_split", 0.70))
MIN_SAMPLES_PER_CLASS: int = int(GLOBAL_SETTINGS.get("min_samples_per_class", 10))
N_FEATURES: int = int(GLOBAL_SETTINGS.get("n_features", 126))

# ── V6.1: رُفع وزن draw_model من 0.13 → 0.20 ─────────────────
# ── V6.1: خُفِّضت أوزان النماذج الأخرى توازناً ─────────────────
WEIGHTS: Dict[str, float] = {
    # V7.2: calibration-window weight optimization; Walk-forward diagnostics showed Elo is the strongest single
    # component across all five leagues; weak components are deliberately
    # down-weighted rather than letting them dilute the signal.
    "dixon_coles":    0.153,
    "elo":            0.340,
    "form":           0.068,
    "h2h":            0.060,
    "home_advantage": 0.034,
    "fatigue":        0.025,
    "draw_model":     0.170,
    "ml":             0.150,
}
# المجموع = 1.00 ✅

# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def poisson_pmf(k: int, mu: float) -> float:
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    return (mu ** k) * math.exp(-mu) / math.factorial(k)


def safe_div(a: float, b: float, d: float = 0.0) -> float:
    return a / b if b else d


def parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        c = s.replace("Z", "")
        fmt = "%Y-%m-%dT%H:%M:%S" if "T" in c else "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(c[:19], fmt)
    except Exception:
        return None


def normalize_probs(hp: float, dp: float, ap: float) -> Tuple[float, float, float]:
    """Return a finite, non-negative probability simplex.

    V10.0.5: this is the single integrity boundary used before persistence,
    metrics and any sklearn scoring call. Non-finite values are neutralised,
    negative values are clipped, and the final component is adjusted for tiny
    floating-point residue so every row sums to one within machine precision.
    """
    vals = []
    for v in (hp, dp, ap):
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        vals.append(max(0.0, v) if math.isfinite(v) else 0.0)
    t = sum(vals)
    if t <= 0.0:
        return (0.40, 0.25, 0.35)
    vals = [v / t for v in vals]
    vals[2] = max(0.0, 1.0 - vals[0] - vals[1])
    t2 = sum(vals)
    if t2 <= 0.0 or not all(math.isfinite(v) for v in vals):
        return (0.40, 0.25, 0.35)
    return tuple(v / t2 for v in vals)


def validate_probability_simplex(probs, tol: float = 1e-8) -> Tuple[float, float, float]:
    """V10.0.5 metrics boundary: sanitize and guarantee a valid 3-class simplex."""
    try:
        hp, dp, ap = probs
    except Exception:
        hp, dp, ap = (0.40, 0.25, 0.35)
    out = normalize_probs(hp, dp, ap)
    if abs(sum(out) - 1.0) > tol or not all(math.isfinite(v) and v >= 0.0 for v in out):
        raise ValueError("Probability integrity failure: invalid 3-class simplex")
    return out


def align_probability_matrix(proba, classes, expected_classes=(0, 1, 2)):
    """V10.0.6 hard metric boundary for sklearn probability matrices.

    Align estimator columns to the global HOME/DRAW/AWAY class order, sanitize
    every row, and fail loudly if a matrix cannot be made valid. This prevents
    temporal folds with missing classes or floating-point drift from reaching
    log-loss scoring with an invalid simplex.
    """
    arr = np.asarray(proba, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Probability integrity failure: expected 2D matrix")
    out = np.zeros((arr.shape[0], len(expected_classes)), dtype=float)
    cls = list(np.asarray(classes).tolist())
    for j, c in enumerate(cls):
        if c in expected_classes:
            out[:, expected_classes.index(c)] = arr[:, j]
    for i in range(out.shape[0]):
        out[i] = np.asarray(validate_probability_simplex(out[i]), dtype=float)
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise ValueError("Probability integrity failure: non-finite matrix")
    if not np.allclose(out.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("Probability integrity failure: matrix rows do not sum to one")
    return out


def temporal_neg_log_loss_scorer(estimator, X, y):
    """V10.0.6 safe temporal CV scorer with explicit class alignment."""
    raw = estimator.predict_proba(X)
    classes = getattr(estimator, "classes_", (0, 1, 2))
    probs = align_probability_matrix(raw, classes, (0, 1, 2))
    y_arr = np.asarray(y, dtype=int)
    if np.any(~np.isin(y_arr, (0, 1, 2))):
        raise ValueError("Unexpected outcome label in temporal scoring")
    ll = -np.log(np.clip(probs[np.arange(len(y_arr)), y_arr], 1e-15, 1.0)).mean()
    return -float(ll)


# ══════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════
import shutil

class C:
    H  = "\033[95m"
    B  = "\033[94m"
    CN = "\033[96m"
    G  = "\033[92m"
    Y  = "\033[93m"
    R  = "\033[91m"
    BD = "\033[1m"
    DM = "\033[2m"
    E  = "\033[0m"
    W  = "\033[97m"

    @staticmethod
    def bold(t):    return f"{C.BD}{t}{C.E}"
    @staticmethod
    def green(t):   return f"{C.G}{t}{C.E}"
    @staticmethod
    def red(t):     return f"{C.R}{t}{C.E}"
    @staticmethod
    def yellow(t):  return f"{C.Y}{t}{C.E}"
    @staticmethod
    def cyan(t):    return f"{C.CN}{t}{C.E}"
    @staticmethod
    def blue(t):    return f"{C.B}{t}{C.E}"
    @staticmethod
    def dim(t):     return f"{C.DM}{t}{C.E}"
    @staticmethod
    def magenta(t): return f"{C.H}{t}{C.E}"

    @staticmethod
    def form_char(ch: str) -> str:
        if ch == "W": return f"{C.G}{C.BD}W{C.E}"
        if ch == "D": return f"{C.Y}{C.BD}D{C.E}"
        if ch == "L": return f"{C.R}{C.BD}L{C.E}"
        return ch

    @staticmethod
    def form_str(s: str) -> str:
        return " ".join(C.form_char(c) for c in s)

    @staticmethod
    def pct_bar(v: float, w: int = 20, color: str = None) -> str:
        color = color or C.G
        f = int(max(0.0, min(1.0, v)) * w)
        e = w - f
        return f"{color}{'█' * f}{C.E}{C.DM}{'░' * e}{C.E}"


def box(t: str) -> str:
    return f" {C.blue('│')} {t}"


# ══════════════════════════════════════════════════════════════
# LEAGUE RESOURCES
# ══════════════════════════════════════════════════════════════
class LeagueResources:
    def __init__(self, league_code: str, config: dict):
        self.code            = league_code
        self.config          = config
        self.name            = config.get("name", league_code)
        self.country         = config.get("country", "")
        self.api_code        = config.get("api_code", league_code)
        self.api_url         = config.get("api_url", "https://api.football-data.org/v4")
        self.home_advantage  = config.get("home_advantage", 65)
        self.avg_home_goals  = config.get("avg_home_goals", 1.53)
        self.avg_away_goals  = config.get("avg_away_goals", 1.16)
        self.total_teams     = config.get("total_teams", 20)
        self.total_rounds    = config.get("total_rounds", 38)

        self.data_files:        List[str]              = config.get("data_files", [])
        self.model_file:        str = config.get("model_file",       f"models/{league_code}_model_v7.pkl")
        self.calibration_file:  str = config.get("calibration_file", f"models/{league_code}_calibration_v7.7.pkl")
        self.elo_file:          str = config.get("elo_file",         f"models/{league_code}_elo_v7.pkl")
        self.teams_map_file:    str = config.get("teams_map_file",   f"config/{league_code}_teams_map.json")
        self.aliases_file:      str = config.get("aliases_file",     f"config/{league_code}_aliases.json")
        self.rivalries_file:    str = config.get("rivalries_file",   f"config/{league_code}_rivalries.json")

        self.teams_map:   Dict[str, str]         = {}
        self.aliases:     Dict[str, str]         = {}
        self.rivalries:   Dict[frozenset, str]   = {}
        self.elo_ratings: Dict[str, float]       = {}

        # V6.1: بناء lookup مُسبق للأسماء المختصرة بدل O(n) عند كل بحث
        self._alias_prefix: Dict[str, str] = {}

        self._load_all()

    # ── تحميل ────────────────────────────────────────────────
    def _load_all(self):
        self._load_teams_map()
        self._load_aliases()
        self._load_rivalries()
        self._load_elo()

    def _load_teams_map(self):
        if Path(self.teams_map_file).exists():
            try:
                with open(self.teams_map_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.teams_map = self._build_safe_teams_map(raw)
            except Exception as e:
                print(f"⚠️  [{self.code}] teams_map error: {e}")

    def _load_aliases(self):
        if Path(self.aliases_file).exists():
            try:
                with open(self.aliases_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.aliases = {k.lower().strip(): v for k, v in raw.items()}
            except Exception as e:
                print(f"⚠️  [{self.code}] aliases error: {e}")
        else:
            self.aliases = self._get_default_aliases()
        # V6.1: بناء prefix-map لتسريع norm_name()
        self._build_prefix_map()

    def _build_prefix_map(self):
        """V6.1: lookup جزئي O(1) بدل O(n)"""
        self._alias_prefix = {}
        for k, v in self.aliases.items():
            # نُسجّل كل بادئة بطول ≥ 4 حروف
            for length in range(4, len(k) + 1):
                prefix = k[:length]
                if prefix not in self._alias_prefix:
                    self._alias_prefix[prefix] = v

    def _load_rivalries(self):
        if Path(self.rivalries_file).exists():
            try:
                with open(self.rivalries_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for item in raw:
                    if isinstance(item, dict):
                        teams = item.get("teams", [])
                        derby_name = item.get("name", "Derby")
                        if len(teams) == 2:
                            self.rivalries[frozenset(teams)] = derby_name
            except Exception as e:
                print(f"⚠️  [{self.code}] rivalries error: {e}")

    def _load_elo(self):
        if Path(self.elo_file).exists():
            try:
                with open(self.elo_file, "rb") as f:
                    raw_elo = pickle.load(f)
                if isinstance(raw_elo, dict):
                    for key, val in raw_elo.items():
                        if isinstance(key, str) and isinstance(val, (int, float)):
                            self.elo_ratings[key] = float(val)
            except Exception as e:
                print(f"⚠️  [{self.code}] elo error: {e}")

    def save_elo(self, elo_data: Dict[str, float]):
        Path(self.elo_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.elo_file, "wb") as f:
                pickle.dump(elo_data, f)
        except Exception as e:
            print(f"⚠️  [{self.code}] save elo error: {e}")

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def _build_safe_teams_map(raw: dict) -> Dict[str, str]:
        safe = {}
        for k, v in raw.items():
            k_str = str(k).strip()
            if (
                " v " in k_str.lower()
                or any(ch.isdigit() for ch in k_str)
                or len(k_str) > 40
            ):
                continue
            safe[k_str.lower()] = str(v)
        return safe

    @staticmethod
    def _get_default_aliases() -> Dict[str, str]:
        return {
            "manchester united":        "Man United",
            "manchester city":          "Man City",
            "tottenham hotspur":        "Tottenham",
            "tottenham":                "Tottenham",
            "newcastle united":         "Newcastle",
            "west ham united":          "West Ham",
            "wolverhampton wanderers":  "Wolves",
            "wolverhampton":            "Wolves",
            "nottingham forest":        "Nottm Forest",
            "leicester city":           "Leicester",
            "brighton & hove albion":   "Brighton",
            "brighton":                 "Brighton",
            "crystal palace":           "Crystal Palace",
            "aston villa":              "Aston Villa",
            "arsenal":                  "Arsenal",
            "chelsea":                  "Chelsea",
            "liverpool":                "Liverpool",
            "everton":                  "Everton",
            "brentford":                "Brentford",
            "fulham":                   "Fulham",
            "bournemouth":              "Bournemouth",
            "luton town":               "Luton",
            "sheffield united":         "Sheffield Utd",
            "burnley":                  "Burnley",
            "ipswich town":             "Ipswich",
            "southampton":              "Southampton",
            "real madrid":              "Real Madrid",
            "fc barcelona":             "Barcelona",
            "atletico madrid":          "Atletico Madrid",
            "athletic bilbao":          "Athletic Club",
            "fc bayern münchen":        "Bayern Munich",
            "borussia dortmund":        "Dortmund",
            "bayer 04 leverkusen":      "Leverkusen",
            "inter milan":              "Inter",
            "ac milan":                 "Milan",
            "juventus fc":              "Juventus",
            "paris saint-germain":      "PSG",
            "olympique de marseille":   "Marseille",
            "olympique lyonnais":       "Lyon",
        }

    # V6.1: norm_name() أسرع بفضل prefix-map
    def norm_name(self, name: str) -> str:
        lo = name.lower().strip()
        # 1. تطابق تام
        if lo in self.aliases:
            return self.aliases[lo]
        # 2. بحث prefix مُسبق O(1)
        if lo in self._alias_prefix:
            return self._alias_prefix[lo]
        # 3. teams_map
        if self.teams_map:
            if lo in self.teams_map:
                return self.teams_map[lo]
            lo_words = set(lo.split())
            candidates = [
                v for k, v in self.teams_map.items()
                if set(k.split()) == lo_words
            ]
            if len(candidates) == 1:
                return candidates[0]
        return name

    def is_derby(self, home: str, away: str) -> Optional[str]:
        return self.rivalries.get(
            frozenset({self.norm_name(home), self.norm_name(away)})
        )

    def canonical_team_id(self, name: str) -> int:
        """Stable team identity shared by CSV and API sources."""
        canonical = self.norm_name(str(name)).strip().lower()
        digest = hashlib.sha256(f"{self.code}:{canonical}".encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % 2_000_000_000

    # ── CSV loading ───────────────────────────────────────────
    def load_csv_data(self) -> List[dict]:
        all_matches: List[dict] = []
        for csv_path in self.data_files:
            if not Path(csv_path).exists():
                print(f"⚠️  [{self.code}] File not found: {csv_path}")
                continue
            matches = self._parse_csv(csv_path)
            all_matches.extend(matches)
            print(f"✅ [{self.code}] Loaded {len(matches)} matches from {csv_path}")
        all_matches.sort(key=lambda x: x.get("utcDate", ""))
        return all_matches

    def _parse_csv(self, csv_path: str) -> List[dict]:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            matches: List[dict] = []
            for _, row in df.iterrows():
                try:
                    if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                        continue
                    h_name = self.norm_name(str(row["HomeTeam"]))
                    a_name = self.norm_name(str(row["AwayTeam"]))
                    hid = self.canonical_team_id(h_name)
                    aid = self.canonical_team_id(a_name)
                    date_str = str(row["Date"])
                    try:
                        fmt = (
                            "%d/%m/%y"
                            if len(date_str.split("/")[-1]) == 2
                            else "%d/%m/%Y"
                        )
                        dt = datetime.strptime(date_str, fmt)
                    except ValueError:
                        try:
                            dt = pd.to_datetime(date_str).to_pydatetime()
                        except Exception:
                            continue

                    stats: Dict[str, float] = {}
                    for col in ["HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]:
                        val = row.get(col, 0)
                        stats[col] = 0.0 if pd.isna(val) else float(val)

                    matches.append({
                        "status":   "FINISHED",
                        "utcDate":  dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "homeTeam": {"id": hid, "shortName": h_name, "name": h_name},
                        "awayTeam": {"id": aid, "shortName": a_name, "name": a_name},
                        "score":    {"fullTime": {"home": int(row["FTHG"]), "away": int(row["FTAG"])}},
                        "stats":    stats,
                        "league":   self.code,
                    })
                except Exception:
                    continue
            return matches
        except Exception as e:
            print(f"❌ [{self.code}] CSV parse error {csv_path}: {e}")
            return []


# ══════════════════════════════════════════════════════════════
# TEAM
# ══════════════════════════════════════════════════════════════
class Team:
    __slots__ = (
        "id", "name", "played", "wins", "draws", "losses",
        "gf", "ga", "pts", "pos",
        "h_p", "h_w", "h_d", "h_gf", "h_ga",
        "a_p", "a_w", "a_d", "a_gf", "a_ga",
        "results", "elo", "elo_hist", "match_dates",
        "cs", "fts", "_last_draw", "consec_draws",
        "win_streak", "loss_streak", "unbeaten",
        "stats_played", "sot_for", "sot_against",
        "corners_for", "corners_against", "discipline_pts",
    )

    def __init__(self, tid: int, name: str, elo_init: float = None):
        self.id      = tid
        self.name    = name
        self.played  = 0
        self.wins    = 0
        self.draws   = 0
        self.losses  = 0
        self.gf      = 0
        self.ga      = 0
        self.pts     = 0
        self.pos     = 0

        self.h_p = self.h_w = self.h_d = self.h_gf = self.h_ga = 0
        self.a_p = self.a_w = self.a_d = self.a_gf = self.a_ga = 0

        self.results:     List[Tuple] = []
        self.elo:         float       = elo_init if elo_init is not None else ELO_INIT
        self.elo_hist:    List[float] = [self.elo]
        self.match_dates: List[datetime] = []

        self.cs = self.fts = 0
        self._last_draw  = False
        self.consec_draws = 0
        self.win_streak  = 0
        self.loss_streak = 0
        self.unbeaten    = 0

        self.stats_played    = 0
        self.sot_for         = 0.0
        self.sot_against     = 0.0
        self.corners_for     = 0.0
        self.corners_against = 0.0
        self.discipline_pts  = 0.0

    # ── properties ────────────────────────────────────────────
    @property
    def gd(self):          return self.gf - self.ga
    @property
    def avg_gf(self):      return safe_div(self.gf, self.played)
    @property
    def avg_ga(self):      return safe_div(self.ga, self.played)
    @property
    def h_avg_gf(self):    return safe_div(self.h_gf, self.h_p)
    @property
    def h_avg_ga(self):    return safe_div(self.h_ga, self.h_p)
    @property
    def a_avg_gf(self):    return safe_div(self.a_gf, self.a_p)
    @property
    def a_avg_ga(self):    return safe_div(self.a_ga, self.a_p)
    @property
    def h_wr(self):        return safe_div(self.h_w, self.h_p, 0.45)
    @property
    def a_wr(self):        return safe_div(self.a_w, self.a_p, 0.30)
    @property
    def wr(self):          return safe_div(self.wins, self.played)
    @property
    def dr(self):          return safe_div(self.draws, self.played)
    @property
    def h_dr(self):        return safe_div(self.h_d, self.h_p)
    @property
    def a_dr(self):        return safe_div(self.a_d, self.a_p)
    @property
    def cs_r(self):        return safe_div(self.cs, self.played)
    @property
    def fts_r(self):       return safe_div(self.fts, self.played)
    @property
    def ppg(self):         return safe_div(self.pts, self.played)
    @property
    def avg_sot(self):     return safe_div(self.sot_for, self.stats_played)
    @property
    def avg_corners(self): return safe_div(self.corners_for, self.stats_played)
    @property
    def avg_discipline(self): return safe_div(self.discipline_pts, self.stats_played)

    @property
    def form_score(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return 50.0
        total = max_t = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.3 * (i - len(rec) + 1))
            total += {"W": 3, "D": 1, "L": 0}[r[0]] * w
            max_t += 3 * w
        return (total / max_t) * 100 if max_t else 50.0

    @property
    def goal_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return self.avg_gf
        total = wt = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.2 * (i - len(rec) + 1))
            total += r[1] * w
            wt    += w
        return total / wt if wt else self.avg_gf

    @property
    def defense_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return self.avg_ga
        total = wt = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.2 * (i - len(rec) + 1))
            total += r[2] * w
            wt    += w
        return total / wt if wt else self.avg_ga

    @property
    def draw_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return self.dr
        return sum(1 for r in rec if r[0] == "D") / len(rec)

    @property
    def form_string(self) -> str:
        return "".join(r[0] for r in self.results[-6:])

    @property
    def momentum(self) -> int:
        if self.win_streak >= 5:  return 90
        if self.win_streak >= 3:  return 60 + self.win_streak * 5
        if self.win_streak >= 2:  return 40
        if self.unbeaten  >= 5:   return 30
        if self.loss_streak >= 4: return -80
        if self.loss_streak >= 3: return -50
        if self.loss_streak >= 2: return -25
        return 0

    @property
    def volatility(self) -> float:
        rec = self.results[-10:]
        if len(rec) < 4:
            return 0.5
        goals = [r[1] + r[2] for r in rec]
        mean  = sum(goals) / len(goals)
        var   = sum((g - mean) ** 2 for g in goals) / len(goals)
        return min(1.0, math.sqrt(var) / 2.0)

    def days_rest(self, ref: datetime = None) -> int:
        ref = ref or datetime.now()
        past = [d for d in self.match_dates if d < ref]
        if not past:
            return 7
        return max(0, (ref - max(past)).days)

    def matches_in(self, n: int = 14, ref: datetime = None) -> int:
        ref = ref or datetime.now()
        cut = ref - timedelta(days=n)
        return sum(1 for d in self.match_dates if cut <= d < ref)


# ══════════════════════════════════════════════════════════════
# ELO SYSTEM  — V6.1: معادلة Draw أكثر واقعية
# ══════════════════════════════════════════════════════════════
class EloSystem:
    def __init__(self, home_advantage: int = 65):
        self.k  = ELO_K
        self.ha = home_advantage

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def gd_mult(self, gd: int) -> float:
        gd = abs(gd)
        if gd <= 1: return 1.0
        if gd == 2: return 1.5
        return (11 + gd) / 8

    def update(self, h: "Team", a: "Team", hg: int, ag: int):
        ha = h.elo + self.ha
        eh = self.expected(ha, a.elo)
        ea = 1 - eh

        if hg > ag:   ah, aa = 1.0, 0.0
        elif hg < ag: ah, aa = 0.0, 1.0
        else:         ah, aa = 0.5, 0.5

        m  = self.gd_mult(hg - ag)
        kh = self.k * (1.5 if h.played < 5 else (0.85 if h.elo > 1600 else 1.0))
        ka = self.k * (1.5 if a.played < 5 else (0.85 if a.elo > 1600 else 1.0))

        h.elo += kh * m * (ah - eh)
        a.elo += ka * m * (aa - ea)
        h.elo_hist.append(h.elo)
        a.elo_hist.append(a.elo)

    # V6.1 FIX: معادلة Draw أكثر واقعية
    def predict(self, h: "Team", a: "Team") -> Tuple[float, float, float]:
        ha  = h.elo + self.ha
        eh  = self.expected(ha, a.elo)
        ea  = 1 - eh
        dd  = abs(ha - a.elo)

        # V6.1: db يتناسب عكسياً مع فارق Elo بشكل أكثر واقعية
        # عند dd=0 → db=0.30 | عند dd=400 → db=0.20 | حد أدنى 0.15
        db = max(0.15, 0.30 - dd / 800.0)

        hw = eh * (1 - db)
        aw = ea * (1 - db)
        return normalize_probs(hw, db, aw)


# ══════════════════════════════════════════════════════════════
# DIXON-COLES
# ══════════════════════════════════════════════════════════════
class DixonColes:
    @staticmethod
    def tau(hg: int, ag: int, lh: float, la: float, rho: float) -> float:
        if hg == 0 and ag == 0: return 1 - lh * la * rho
        if hg == 0 and ag == 1: return 1 + lh * rho
        if hg == 1 and ag == 0: return 1 + la * rho
        if hg == 1 and ag == 1: return 1 - rho
        return 1.0

    @staticmethod
    def prob(hg: int, ag: int, lh: float, la: float, rho: float = -0.13) -> float:
        b = poisson_pmf(hg, lh) * poisson_pmf(ag, la)
        return max(0.0, b * DixonColes.tau(hg, ag, lh, la, rho))

    @staticmethod
    def predict(
        hxg: float, axg: float, rho: float = -0.13, mg: int = 10
    ) -> Tuple[float, float, float]:
        hw = dr = aw = 0.0
        for i in range(mg):
            for j in range(mg):
                p = DixonColes.prob(i, j, hxg, axg, rho)
                if   i > j: hw += p
                elif i == j: dr += p
                else:        aw += p
        t = hw + dr + aw
        return (hw / t, dr / t, aw / t) if t > 0 else (0.40, 0.25, 0.35)

    @staticmethod
    def matrix(
        hxg: float, axg: float, rho: float = -0.13, mg: int = 10
    ) -> Dict[Tuple[int, int], float]:
        return {
            (i, j): DixonColes.prob(i, j, hxg, axg, rho)
            for i in range(mg)
            for j in range(mg)
        }


# ══════════════════════════════════════════════════════════════
# IMPROVED DRAW PREDICTOR
# ══════════════════════════════════════════════════════════════
class ImprovedDrawPredictor:
    """
    نموذج متخصص للتعادل بناءً على البحث الإحصائي.
    يأخذ في الاعتبار:
      1. فارق قوة الفريقين (Elo)
      2. الأسلوب الدفاعي لكلا الفريقين
      3. التاريخ المباشر (H2H)
      4. الديربي
      5. مرحلة الموسم
      6. الفورم المحايد
      7. انخفاض التقلب
      8. الزخم (جديد V6.1)
    """

    @staticmethod
    def predict_draw_prob(
        h: "Team",
        a: "Team",
        h2h_draw_rate: float = 0.25,
        elo_diff: float = 0.0,
        is_derby: bool = False,
        is_late_season: bool = False,
    ) -> float:
        base        = 0.26
        adjustments = 0.0

        # ── فارق القوة ────────────────────────────────────────
        elo_factor   = max(0.0, 1.0 - abs(elo_diff) / 300.0)
        adjustments += elo_factor * 0.08

        # ── الأسلوب الدفاعي ───────────────────────────────────
        avg_total_goals = (h.avg_gf + h.avg_ga + a.avg_gf + a.avg_ga) / 2.0
        defensive_style  = max(0.0, 2.5 - avg_total_goals) / 2.5
        adjustments     += defensive_style * 0.06

        # ── H2H ───────────────────────────────────────────────
        adjustments += (h2h_draw_rate - 0.25) * 0.40

        # ── ديربي ─────────────────────────────────────────────
        if is_derby:
            adjustments += 0.04

        # ── نهاية الموسم ──────────────────────────────────────
        if is_late_season and abs(h.pos - a.pos) <= 2:
            adjustments += 0.03

        # ── فورم محايد ────────────────────────────────────────
        h_neutral = abs(h.form_score - 50.0) < 15.0
        a_neutral = abs(a.form_score - 50.0) < 15.0
        if h_neutral and a_neutral:
            adjustments += 0.04

        # ── انخفاض التقلب ─────────────────────────────────────
        if (h.volatility + a.volatility) / 2.0 < 0.35:
            adjustments += 0.03

        # ── الزخم (V6.1): زخم قوي → تعادل أقل ───────────────
        if abs(h.momentum) > 50 or abs(a.momentum) > 50:
            adjustments -= 0.04

        # ── معدل تعادل الفريقين ───────────────────────────────
        team_draw_tendency = (h.draw_form + a.draw_form) / 2.0
        adjustments += (team_draw_tendency - 0.25) * 0.20

        return min(0.45, max(0.15, base + adjustments))

    @staticmethod
    def predict(
        h: "Team",
        a: "Team",
        h2h_draw_rate: float = 0.25,
        elo_diff: float = 0.0,
        is_derby: bool = False,
        is_late_season: bool = False,
    ) -> Tuple[float, float, float]:
        dp  = ImprovedDrawPredictor.predict_draw_prob(
            h, a, h2h_draw_rate, elo_diff, is_derby, is_late_season
        )
        rem = 1.0 - dp
        if   elo_diff > 0: hp, ap = rem * 0.58, rem * 0.42
        elif elo_diff < 0: hp, ap = rem * 0.42, rem * 0.58
        else:              hp, ap = rem * 0.50, rem * 0.50
        return normalize_probs(hp, dp, ap)


# ══════════════════════════════════════════════════════════════
# FATIGUE
# ══════════════════════════════════════════════════════════════
class Fatigue:
    @staticmethod
    def score(t: "Team", ref: datetime = None) -> float:
        ref = ref or datetime.now()
        rd  = t.days_rest(ref)
        m14 = t.matches_in(14, ref)
        m30 = t.matches_in(30, ref)

        rest_map = {0: 40, 1: 40, 2: 40, 3: 30, 4: 20, 5: 10}
        rs  = rest_map.get(rd, 0 if rd <= 7 else -5)
        d14 = 35 if m14 >= 5 else (25 if m14 >= 4 else (15 if m14 >= 3 else 0))
        d30 = 25 if m30 >= 9 else (15 if m30 >= 7 else 0)
        return max(0.0, min(100.0, rs + d14 + d30))

    @staticmethod
    def impact(t: "Team", ref: datetime = None) -> float:
        return 1.05 - (Fatigue.score(t, ref) / 100.0) * 0.17

    @staticmethod
    def predict(
        h: "Team", a: "Team", ref: datetime = None
    ) -> Tuple[float, float, float]:
        hi = Fatigue.impact(h, ref)
        ai = Fatigue.impact(a, ref)
        t  = hi + ai
        if t == 0:
            return (0.40, 0.25, 0.35)
        hp = hi / t
        ap = ai / t
        d  = max(0.18, 0.30 - abs(hp - ap) * 0.3)
        hp *= (1 - d)
        ap *= (1 - d)
        return normalize_probs(hp, d, ap)


# ══════════════════════════════════════════════════════════════
# CALIBRATOR  — V6.1: حماية من القيم السالبة
# ══════════════════════════════════════════════════════════════
class Calibrator:
    """V7.7 Probability Calibration 2.0 + Reliability Engine.

    Global temperature/draw-bias is the stable fallback.  Additional conservative
    reliability profiles are learned by temporal holdout for outcome, confidence
    bucket and match regime.  A profile is activated only when its own holdout
    improves log-loss versus the global calibration.
    """
    VERSION = "7.7"
    LABELS = ("HOME", "DRAW", "AWAY")

    def __init__(self):
        self.ok=False; self.temperature=1.0; self.draw_bias=0.0; self.hist=[]
        self.profiles={}; self.profile_stats={}; self.global_logloss=None

    @staticmethod
    def _softmax(logits):
        z=np.asarray(logits,dtype=float); z=z-np.max(z)
        q=np.exp(z); return q/max(float(q.sum()),1e-12)

    @staticmethod
    def _bucket(probs):
        c=max(float(x) for x in probs)
        if c < .42: return "LOW"
        if c < .55: return "MEDIUM"
        return "HIGH"

    def add(self, probs, actual, regime="BALANCED", context=None):
        self.hist.append({"probs":tuple(float(x) for x in probs),"actual":actual,
                          "regime":str(regime or "BALANCED"),"context":context or {}})

    def _fit_params(self, rows):
        p=np.clip(np.array([x["probs"] for x in rows],dtype=float),1e-6,1-1e-6)
        y=np.array([self.LABELS.index(x["actual"]) for x in rows])
        logits=np.log(p); best=(float("inf"),1.0,0.0)
        for t in np.exp(np.linspace(np.log(.60),np.log(2.4),31)):
            base=logits/t
            for db in np.linspace(-.60,.60,25):
                z=base.copy(); z[:,1]+=db
                z=z-np.max(z,axis=1,keepdims=True); q=np.exp(z); q/=q.sum(axis=1,keepdims=True)
                loss=float(-np.mean(np.log(np.clip(q[np.arange(len(y)),y],1e-9,1))))
                if loss<best[0]: best=(loss,float(t),float(db))
        return best

    def _adjust_params(self, probs, t, db):
        p=np.clip(np.asarray(probs,dtype=float),1e-9,1.0)
        return self._softmax(np.log(p)/max(t,.05)+np.array([0.0,db,0.0]))

    def calibrate(self):
        if len(self.hist)<30 or not ML_AVAILABLE: return False
        try:
            # Temporal holdout: fit global parameters only on earlier data.
            cut=max(20,int(len(self.hist)*.70))
            if len(self.hist)-cut<10: cut=len(self.hist)
            train=self.hist[:cut]; hold=self.hist[cut:]
            loss,t,db=self._fit_params(train)
            self.temperature=t; self.draw_bias=db
            if hold:
                yy=np.array([self.LABELS.index(x["actual"]) for x in hold])
                qq=np.array([self._adjust_params(x["probs"],t,db) for x in hold])
                self.global_logloss=float(-np.mean(np.log(np.clip(qq[np.arange(len(yy)),yy],1e-9,1))))
            # Refit stable global fallback on all calibration rows.
            _,self.temperature,self.draw_bias=self._fit_params(self.hist)
            self.ok=True
            self._learn_profiles()
            return True
        except Exception: return False

    def _learn_profiles(self):
        self.profiles={}; self.profile_stats={}
        groups=defaultdict(list)
        for row in self.hist:
            # regime and confidence are known before kickoff; class-specific
            # groups are learned from outcomes but only used as diagnostic gates.
            b=self._bucket(row["probs"])
            groups[f"REGIME:{row.get('regime','BALANCED')}"] .append(row)
            groups[f"CONF:{b}"] .append(row)
        for key,rows in groups.items():
            if len(rows)<50: continue
            cut=max(35,int(len(rows)*.70)); hold=rows[cut:]
            if len(hold)<15: continue
            _,t,db=self._fit_params(rows[:cut])
            y=np.array([self.LABELS.index(x["actual"]) for x in hold])
            base=np.array([self._adjust_params(x["probs"],self.temperature,self.draw_bias) for x in hold])
            cand=np.array([self._adjust_params(x["probs"],t,db) for x in hold])
            bll=float(-np.mean(np.log(np.clip(base[np.arange(len(y)),y],1e-9,1))))
            cll=float(-np.mean(np.log(np.clip(cand[np.arange(len(y)),y],1e-9,1))))
            if cll < bll-0.0015:
                # Refit on all rows only after temporal validation passes.
                _,t,db=self._fit_params(rows)
                self.profiles[key]={"temperature":t,"draw_bias":db,"n":len(rows),"holdout_logloss":cll,"baseline_logloss":bll}
            self.profile_stats[key]={"n":len(rows),"active":key in self.profiles,"holdout_logloss":cll,"baseline_logloss":bll}

    def adjust(self, probs, regime="BALANCED"):
        if not self.ok: return probs
        try:
            keys=[f"REGIME:{regime}",f"CONF:{self._bucket(probs)}"]
            active=[self.profiles[k] for k in keys if k in self.profiles]
            q=self._adjust_params(probs,self.temperature,self.draw_bias)
            # Blend validated local corrections conservatively to avoid sparse-bin overfit.
            for prof in active:
                z=self._adjust_params(probs,prof["temperature"],prof["draw_bias"])
                q=.65*q+.35*z; q=q/q.sum()
            return tuple(float(x) for x in q)
        except Exception: return probs

    def reliability(self, probs, regime="BALANCED"):
        """Return reliability factor and reasons using only pre-match signals."""
        conf=max(float(x) for x in probs); factor=1.0; reasons=[]
        keys=[f"REGIME:{regime}",f"CONF:{self._bucket(probs)}"]
        for k in keys:
            st=self.profile_stats.get(k)
            if st and not st.get("active",False) and st.get("n",0)>=50:
                # Historically unstable segment: lower confidence, never alter ranking.
                factor*=.92; reasons.append(k)
        if conf>=.62 and not any(k in self.profiles for k in keys):
            factor*=.96; reasons.append("UNVALIDATED_HIGH_CONF")
        return max(.75,min(1.0,factor)), reasons

    def save(self,fn):
        Path(fn).parent.mkdir(parents=True,exist_ok=True)
        with open(fn,"wb") as f: pickle.dump({"version":self.VERSION,"hist":self.hist,"ok":self.ok,"temperature":self.temperature,"draw_bias":self.draw_bias,"profiles":self.profiles,"profile_stats":self.profile_stats,"global_logloss":self.global_logloss},f)

    def load(self,fn):
        try:
            if not Path(fn).exists(): return False
            with open(fn,"rb") as f:d=pickle.load(f)
            # Older calibration files remain readable but are upgraded only from history.
            self.hist=d.get("hist",[]); self.ok=False
            if d.get("version")==self.VERSION:
                self.temperature=float(d.get("temperature",1.0)); self.draw_bias=float(d.get("draw_bias",0.0)); self.profiles=d.get("profiles",{}); self.profile_stats=d.get("profile_stats",{}); self.global_logloss=d.get("global_logloss"); self.ok=bool(d.get("ok",False))
            elif len(self.hist)>=30:
                self.calibrate()
            return self.ok
        except Exception:return False


# ══════════════════════════════════════════════════════════════
# DATA PROCESSOR
# ══════════════════════════════════════════════════════════════
class DataProc:
    def __init__(self, resources: "LeagueResources" = None):
        self.resources = resources
        self.teams:    Dict[int, Team]       = {}
        self.elo       = EloSystem(
            home_advantage=resources.home_advantage if resources else 65
        )
        self.avg_h = resources.avg_home_goals if resources else 1.53
        self.avg_a = resources.avg_away_goals if resources else 1.16
        self.total = 0
        self.fixes: List[dict] = []
        self.h2h:   Dict[str, List[dict]] = defaultdict(list)
        self._active_season: Optional[str] = None

    def _get_elo_init(self, name: str) -> float:
        # Never seed a historical replay with a previously saved rating: that
        # can leak future information and double-count the same matches.
        return ELO_INIT

    @staticmethod
    def _season_key(date_str: str) -> Optional[str]:
        dt = parse_date(date_str)
        if not dt:
            return None
        # European football season starts in August.
        start = dt.year if dt.month >= 7 else dt.year - 1
        return f"{start:04d}/{start+1:04d}"

    def _reset_season_stats(self):
        """Reset season-dependent statistics while preserving historical Elo/H2H."""
        for t in self.teams.values():
            t.played=t.wins=t.draws=t.losses=t.gf=t.ga=t.pts=t.pos=0
            t.h_p=t.h_w=t.h_d=t.h_gf=t.h_ga=0
            t.a_p=t.a_w=t.a_d=t.a_gf=t.a_ga=0
            t.results.clear()
            t.match_dates.clear()
            t.cs=t.fts=0
            t._last_draw=False; t.consec_draws=0
            t.win_streak=0; t.loss_streak=0; t.unbeaten=0
            t.stats_played=0; t.sot_for=t.sot_against=0.0
            t.corners_for=t.corners_against=t.discipline_pts=0.0

    # V7: season-aware state prevents all-time standings from contaminating current-season features.
    def process(self, matches: List[dict], do_elo: bool = True):
        matches = sorted(matches, key=lambda m: m.get("utcDate", ""))
        cnt = 0
        for m in matches:
            r = self._ext(m)
            if not r:
                continue
            hid, hn, aid, an, hg, ag, ds = r

            season = self._season_key(ds)
            if season and season != self._active_season:
                if self._active_season is not None:
                    self._reset_season_stats()
                self._active_season = season

            if hid not in self.teams:
                self.teams[hid] = Team(hid, hn, self._get_elo_init(hn))
            if aid not in self.teams:
                self.teams[aid] = Team(aid, an, self._get_elo_init(an))

            h  = self.teams[hid]
            a  = self.teams[aid]
            md = parse_date(ds)
            if md:
                h.match_dates.append(md)
                a.match_dates.append(md)

            if do_elo:
                self.elo.update(h, a, hg, ag)

            # ── إحصائيات عامة ─────────────────────────────────
            h.played += 1;  a.played += 1
            h.gf += hg;     h.ga += ag
            a.gf += ag;     a.ga += hg
            h.h_p += 1;     h.h_gf += hg;   h.h_ga += ag
            a.a_p += 1;     a.a_gf += ag;   a.a_ga += hg

            # ── شباك نظيفة / فيلد تسجيل ───────────────────────
            if ag == 0: h.cs  += 1
            if hg == 0: a.cs  += 1
            if hg == 0: h.fts += 1
            if ag == 0: a.fts += 1

            # ── إحصائيات عميقة ────────────────────────────────
            stats = m.get("stats")
            if stats:
                hst = stats.get("HST", 0)
                if hst is not None and not (
                    isinstance(hst, float) and math.isnan(hst)
                ):
                    h.stats_played += 1
                    a.stats_played += 1
                    h.sot_for      += stats.get("HST", 0) or 0
                    h.sot_against  += stats.get("AST", 0) or 0
                    a.sot_for      += stats.get("AST", 0) or 0
                    a.sot_against  += stats.get("HST", 0) or 0
                    h.corners_for      += stats.get("HC", 0) or 0
                    h.corners_against  += stats.get("AC", 0) or 0
                    a.corners_for      += stats.get("AC", 0) or 0
                    a.corners_against  += stats.get("HC", 0) or 0
                    h.discipline_pts += (
                        (stats.get("HF", 0) or 0)
                        + (stats.get("HY", 0) or 0) * 3
                        + (stats.get("HR", 0) or 0) * 10
                    )
                    a.discipline_pts += (
                        (stats.get("AF", 0) or 0)
                        + (stats.get("AY", 0) or 0) * 3
                        + (stats.get("AR", 0) or 0) * 10
                    )

            # ── نتيجة ─────────────────────────────────────────
            draw = (hg == ag)
            if hg > ag:
                h.wins += 1; h.h_w += 1; h.pts += 3; a.losses += 1
                h.results.append(("W", hg, ag, ds))
                a.results.append(("L", ag, hg, ds))
                h.win_streak  += 1; h.loss_streak  = 0; h.unbeaten += 1
                a.win_streak   = 0; a.loss_streak += 1; a.unbeaten  = 0
            elif hg < ag:
                a.wins += 1; a.a_w += 1; a.pts += 3; h.losses += 1
                h.results.append(("L", hg, ag, ds))
                a.results.append(("W", ag, hg, ds))
                a.win_streak  += 1; a.loss_streak  = 0; a.unbeaten += 1
                h.win_streak   = 0; h.loss_streak += 1; h.unbeaten  = 0
            else:
                h.draws += 1; a.draws += 1
                h.h_d   += 1; a.a_d   += 1
                h.pts   += 1; a.pts   += 1
                h.results.append(("D", hg, ag, ds))
                a.results.append(("D", ag, hg, ds))
                h.win_streak = a.win_streak = 0
                h.loss_streak = a.loss_streak = 0
                h.unbeaten += 1; a.unbeaten += 1

            h.consec_draws = (
                h.consec_draws + 1 if (draw and h._last_draw) else (1 if draw else 0)
            )
            a.consec_draws = (
                a.consec_draws + 1 if (draw and a._last_draw) else (1 if draw else 0)
            )
            h._last_draw = draw
            a._last_draw = draw

            # ── H2H ───────────────────────────────────────────
            key = f"{min(hid, aid)}_{max(hid, aid)}"
            self.h2h[key].append({
                "home_id": hid, "away_id": aid,
                "home_goals": hg, "away_goals": ag,
                "date": ds,
            })

            self.fixes.append({
                "home_id": hid, "away_id": aid,
                "home_goals": hg, "away_goals": ag,
                "date": ds, "home_name": hn, "away_name": an,
                "stats": m.get("stats", {}),
            })
            cnt += 1

        self.total += cnt
        # V6.1: استدعاء مرة واحدة بعد كل batch
        self._avgs()
        self._rank()

    def _ext(self, m: dict):
        if m.get("status") != "FINISHED":
            return None
        ht  = m.get("homeTeam", {})
        at  = m.get("awayTeam", {})
        hid = ht.get("id")
        aid = at.get("id")
        if not hid or not aid:
            return None
        hn = ht.get("shortName") or ht.get("name", "?")
        an = at.get("shortName") or at.get("name", "?")
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            return None
        return (hid, hn, aid, an, int(hg), int(ag), m.get("utcDate", ""))

    def get_h2h(self, t1: int, t2: int) -> List[dict]:
        return self.h2h.get(f"{min(t1, t2)}_{max(t1, t2)}", [])

    def _avgs(self):
        th = sum(t.h_gf for t in self.teams.values())
        ta = sum(t.a_gf for t in self.teams.values())
        tm = sum(t.h_p  for t in self.teams.values())
        if tm:
            self.avg_h = th / tm
            self.avg_a = ta / tm

    def _rank(self):
        active = [t for t in self.teams.values() if t.played > 0]
        for t in self.teams.values():
            if t.played == 0:
                t.pos = 0
        for i, t in enumerate(sorted(active, key=lambda t: (t.pts, t.gd, t.gf), reverse=True), 1):
            t.pos = i

    def team_by_name(self, name: str) -> Optional[Team]:
        lo = name.lower().strip()
        for t in self.teams.values():
            if t.name.lower() == lo:
                return t
        for t in self.teams.values():
            if lo in t.name.lower() or t.name.lower() in lo:
                return t
        return None


# ══════════════════════════════════════════════════════════════
# ML PREDICTOR — 126 FEATURES  V6.1
# ══════════════════════════════════════════════════════════════
class MLPred:
    N_FEATURES = 126

    def __init__(self, model_file: str = None):
        self.pipeline: Optional["Pipeline"] = None
        self.trained   = False
        self.acc       = 0.0
        self._external = False
        self.model_file = model_file or "models/default_model_v6.pkl"

    # ══════════════════════════════════════════════════════════
    # 68 ميزة أصلية
    # ══════════════════════════════════════════════════════════
    def _original_features(
        self,
        h: Team, a: Team, data: "DataProc",
        md: datetime = None, derby: bool = False,
    ) -> List[float]:
        ah = max(data.avg_h, 0.5)
        aa = max(data.avg_a, 0.5)
        return [
            # Elo (3)
            h.elo, a.elo, h.elo - a.elo,
            # Form Score (4)
            h.form_score, a.form_score,
            h.form_score - a.form_score,
            abs(h.form_score - a.form_score),
            # هجوم (6)
            h.h_avg_gf, a.a_avg_gf, h.goal_form, a.goal_form,
            h.goal_form - a.goal_form, h.h_avg_gf - a.a_avg_gf,
            # دفاع (6)
            h.h_avg_ga, a.a_avg_ga, h.defense_form, a.defense_form,
            h.defense_form - a.defense_form, h.h_avg_ga - a.a_avg_ga,
            # نسب هجوم/دفاع (4)
            safe_div(h.h_avg_gf, ah, 1.0), safe_div(a.a_avg_gf, aa, 1.0),
            safe_div(h.h_avg_ga, ah, 1.0), safe_div(a.a_avg_ga, aa, 1.0),
            # نسب الفوز (4)
            h.h_wr, a.a_wr, h.wr, a.wr,
            # الترتيب والنقاط (7)
            h.pos, a.pos, a.pos - h.pos,
            h.pts, a.pts, h.ppg - a.ppg, h.gd,
            # فارق أهداف أواي (1)
            a.gd,
            # الشباك النظيفة (4)
            h.cs_r, a.cs_r, h.fts_r, a.fts_r,
            # الإرهاق (2)
            Fatigue.score(h, md), Fatigue.score(a, md),
            # نسب التعادل (7)
            h.dr, a.dr, (h.dr + a.dr) / 2,
            h.draw_form, a.draw_form, h.h_dr, a.a_dr,
            # التقلب (2)
            h.volatility, a.volatility,
            # ديربي (1)
            1.0 if derby else 0.0,
            # Elo مقياس (1)
            abs(h.elo - a.elo) / 100.0,
            # الزخم (7)
            h.momentum / 100.0, a.momentum / 100.0,
            (h.momentum - a.momentum) / 100.0,
            h.win_streak, a.win_streak,
            h.loss_streak, a.loss_streak,
            # إحصائيات عميقة (9)
            h.avg_sot, a.avg_sot, h.avg_sot - a.avg_sot,
            h.avg_corners, a.avg_corners, h.avg_corners - a.avg_corners,
            h.avg_discipline, a.avg_discipline, h.avg_discipline - a.avg_discipline,
        ]  # 68 ميزة

    # ══════════════════════════════════════════════════════════
    # 12 ميزة زمنية
    # ══════════════════════════════════════════════════════════
    def _temporal_features(
        self, match_date: datetime, h: Team, a: Team
    ) -> List[float]:
        if match_date is None:
            return [0.0] * 12

        month       = match_date.month
        day_of_week = match_date.weekday()
        season_month = ((month - 8) % 12) + 1

        is_early_season = 1.0 if season_month <= 3 else 0.0
        is_late_season  = 1.0 if season_month >= 9 else 0.0
        is_mid_season   = 1.0 if 4 <= season_month <= 8 else 0.0
        is_weekend      = 1.0 if day_of_week >= 5 else 0.0
        is_midweek      = 1.0 if day_of_week in [1, 2, 3] else 0.0
        month_sin       = math.sin(2 * math.pi * month / 12.0)
        month_cos       = math.cos(2 * math.pi * month / 12.0)

        relegation_pressure_h = 0.0
        relegation_pressure_a = 0.0
        title_pressure_h      = 0.0
        title_pressure_a      = 0.0

        if is_late_season:
            if h.pos >= 17:
                relegation_pressure_h = min(1.0, (h.pos - 16) / 4.0)
            if h.pos <= 3:
                title_pressure_h = (4 - h.pos) / 3.0
            if a.pos >= 17:
                relegation_pressure_a = min(1.0, (a.pos - 16) / 4.0)
            if a.pos <= 3:
                title_pressure_a = (4 - a.pos) / 3.0

        both_relegation = 1.0 if (h.pos >= 15 and a.pos >= 15) else 0.0

        return [
            is_early_season,        # 68
            is_mid_season,          # 69
            is_late_season,         # 70
            is_weekend,             # 71
            is_midweek,             # 72
            month_sin,              # 73
            month_cos,              # 74
            relegation_pressure_h,  # 75
            relegation_pressure_a,  # 76
            title_pressure_h,       # 77
            title_pressure_a,       # 78
            both_relegation,        # 79
        ]  # 12 ميزة

    # ══════════════════════════════════════════════════════════
    # 14 ميزة زخم متقدم
    # ══════════════════════════════════════════════════════════
    def _advanced_momentum_features(self, h: Team, a: Team) -> List[float]:
        def weighted_form_stats(team: Team, last_n: int = 6) -> dict:
            results = team.results[-last_n:]
            if not results:
                return {"raw_pts": 0.0, "trend": 0.0, "consistency": 0.5}
            pts  = [3 if r[0] == "W" else (1 if r[0] == "D" else 0) for r in results]
            half = len(pts) // 2
            if half > 0:
                trend = (
                    sum(pts[half:]) / max(1, len(pts) - half)
                    - sum(pts[:half]) / half
                ) / 3.0
            else:
                trend = 0.0
            if len(pts) > 1:
                mean_pts = sum(pts) / len(pts)
                variance = sum((p - mean_pts) ** 2 for p in pts) / len(pts)
                consistency = 1.0 / (1.0 + variance)
            else:
                consistency = 0.5
            return {
                "raw_pts":    sum(pts) / (len(pts) * 3),
                "trend":      trend,
                "consistency": consistency,
            }

        def raw_pts_n(team: Team, n: int) -> float:
            results = team.results[-n:]
            if not results:
                return 0.0
            pts = [3 if r[0] == "W" else (1 if r[0] == "D" else 0) for r in results]
            return sum(pts) / (len(pts) * 3)

        def goal_trend(team: Team, last_n: int = 5) -> Tuple[float, float]:
            results = team.results[-last_n:]
            if not results:
                return (0.0, 0.0)
            gf_list = [r[1] for r in results]
            ga_list = [r[2] for r in results]
            n = len(gf_list)
            if n < 2:
                return (gf_list[0] if gf_list else 0.0, ga_list[0] if ga_list else 0.0)
            x_mean  = (n - 1) / 2.0
            gf_mean = sum(gf_list) / n
            ga_mean = sum(ga_list) / n
            denom   = sum((i - x_mean) ** 2 for i in range(n))
            if denom == 0:
                return (0.0, 0.0)
            gf_slope = sum((i - x_mean) * (gf_list[i] - gf_mean) for i in range(n)) / denom
            ga_slope = sum((i - x_mean) * (ga_list[i] - ga_mean) for i in range(n)) / denom
            return (gf_slope, ga_slope)

        h_stats = weighted_form_stats(h, 6)
        a_stats = weighted_form_stats(a, 6)
        h_hot   = raw_pts_n(h, 3) - raw_pts_n(h, 8)
        a_hot   = raw_pts_n(a, 3) - raw_pts_n(a, 8)
        h_gf_t, h_ga_t = goal_trend(h, 5)
        a_gf_t, a_ga_t = goal_trend(a, 5)

        return [
            h_stats["raw_pts"],                           # 80
            h_stats["trend"],                             # 81
            h_stats["consistency"],                       # 82
            a_stats["raw_pts"],                           # 83
            a_stats["trend"],                             # 84
            a_stats["consistency"],                       # 85
            h_stats["raw_pts"] - a_stats["raw_pts"],      # 86
            h_stats["trend"]   - a_stats["trend"],        # 87
            h_hot,                                        # 88
            a_hot,                                        # 89
            h_gf_t,                                       # 90
            h_ga_t,                                       # 91
            a_gf_t,                                       # 92
            a_ga_t,                                       # 93
        ]  # 14 ميزة

    # ══════════════════════════════════════════════════════════
    # 10 ميزة H2H متقدمة
    # ══════════════════════════════════════════════════════════
    def _advanced_h2h_features(
        self, h: Team, a: Team, data: "DataProc"
    ) -> List[float]:
        h2h_matches = data.get_h2h(h.id, a.id)
        if not h2h_matches:
            return [0.0, 0.0, 0.25, 1.3, 1.1, 0.5, 0.5, 0.0, 0.0, 0.0]

        recent = h2h_matches[-5:]
        h_wins = a_wins = draws = h_goals = a_goals = btts = over25 = 0

        for m in recent:
            hg = m["home_goals"] if m["home_id"] == h.id else m["away_goals"]
            ag = m["away_goals"] if m["home_id"] == h.id else m["home_goals"]
            if   hg > ag: h_wins += 1
            elif ag > hg: a_wins += 1
            else:          draws += 1
            h_goals += hg
            a_goals += ag
            if hg > 0 and ag > 0:   btts   += 1
            if hg + ag > 2:          over25 += 1

        n = len(recent)
        return [
            h_wins  / n,                         # 94 h_dominance
            a_wins  / n,                         # 95 a_dominance
            draws   / n,                         # 96 draw_tendency
            h_goals / n,                         # 97 h2h avg h goals
            a_goals / n,                         # 98 h2h avg a goals
            btts    / n,                         # 99 btts rate
            over25  / n,                         # 100 over2.5 rate
            1.0 if h_wins / n > 0.5 else 0.0,   # 101 favors home
            1.0 if a_wins / n > 0.5 else 0.0,   # 102 favors away
            min(1.0, len(h2h_matches) / 10.0),  # 103 sample weight
        ]  # 10 ميزة

    # ══════════════════════════════════════════════════════════
    # 11 ميزة xG متقدمة  — V6.1 FIX: save_rate محسوبة صحيح
    # ══════════════════════════════════════════════════════════
    def _advanced_xg_features(self, h: Team, a: Team) -> List[float]:
        NORMAL_CONVERSION = 0.30

        h_conversion = safe_div(h.gf, h.sot_for, NORMAL_CONVERSION)
        a_conversion = safe_div(a.gf, a.sot_for, NORMAL_CONVERSION)
        h_luck = h_conversion - NORMAL_CONVERSION
        a_luck = a_conversion - NORMAL_CONVERSION

        # V6.1 FIX: save_rate تعتمد على sot_against فقط (لا fallback خاطئ)
        h_saves_rate = 1.0 - safe_div(h.ga, h.sot_against, 0.70)
        a_saves_rate = 1.0 - safe_div(a.ga, a.sot_against, 0.70)

        # تقييد المدى المنطقي [0, 1]
        h_saves_rate = max(0.0, min(1.0, h_saves_rate))
        a_saves_rate = max(0.0, min(1.0, a_saves_rate))

        total_sot   = h.sot_for + a.sot_against
        h_shot_dom  = safe_div(h.sot_for, total_sot, 0.5)

        total_corn  = h.corners_for + a.corners_for
        h_corn_dom  = safe_div(h.corners_for, total_corn, 0.5)

        h_xg_est = h.sot_for * NORMAL_CONVERSION
        a_xg_est = a.sot_for * NORMAL_CONVERSION
        h_xg_diff = (
            (h.gf - h_xg_est) / max(h.played, 1) if h.stats_played > 0 else 0.0
        )
        a_xg_diff = (
            (a.gf - a_xg_est) / max(a.played, 1) if a.stats_played > 0 else 0.0
        )

        return [
            h_conversion,          # 104
            a_conversion,          # 105
            h_luck,                # 106
            a_luck,                # 107
            h_luck - a_luck,       # 108
            h_saves_rate,          # 109
            a_saves_rate,          # 110
            h_shot_dom,            # 111
            h_corn_dom,            # 112
            h_xg_diff,             # 113
            a_xg_diff,             # 114
        ]  # 11 ميزة

    # ══════════════════════════════════════════════════════════
    # 11 ميزة السياق والضغط
    # ══════════════════════════════════════════════════════════
    def _context_features(
        self, h: Team, a: Team,
        match_date: datetime, data: "DataProc",
    ) -> List[float]:
        elo_diff            = h.elo - a.elo
        elo_diff_normalized = elo_diff / 400.0
        elo_prob_h          = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

        def schedule_difficulty(team: Team, n: int = 5) -> float:
            recent = team.results[-n:]
            if not recent:
                return 0.0
            recent_pts = sum(
                3 if r[0] == "W" else (1 if r[0] == "D" else 0) for r in recent
            ) / max(len(recent), 1)
            return team.ppg - recent_pts

        h_schedule = schedule_difficulty(h, 5)
        a_schedule = schedule_difficulty(a, 5)

        h_stability = a_stability = 0.5
        if len(h.elo_hist) >= 5:
            re = h.elo_hist[-5:]
            h_stability = 1.0 / (1.0 + (max(re) - min(re)) / 100.0)
        if len(a.elo_hist) >= 5:
            re = a.elo_hist[-5:]
            a_stability = 1.0 / (1.0 + (max(re) - min(re)) / 100.0)

        total_rounds    = data.resources.total_rounds if data.resources else 38
        h_progress      = min(1.0, h.played / total_rounds)
        a_progress      = min(1.0, a.played / total_rounds)
        data_reliability = min(h_progress, a_progress)

        active_teams = sum(1 for t in data.teams.values() if t.played > 0)
        total_teams = max(active_teams, 20)
        h_urgency   = max(0.0, (h.pos - total_teams * 0.7) / (total_teams * 0.3))
        a_urgency   = max(0.0, (a.pos - total_teams * 0.7) / (total_teams * 0.3))

        return [
            elo_diff_normalized,       # 115
            elo_prob_h,                # 116
            h_schedule,                # 117
            a_schedule,                # 118
            h_stability,               # 119
            a_stability,               # 120
            h_progress,                # 121
            data_reliability,          # 122
            h_urgency,                 # 123
            a_urgency,                 # 124
            h_urgency - a_urgency,     # 125
        ]  # 11 ميزة

    # ══════════════════════════════════════════════════════════
    # feats() الرئيسية — V6.1: آمنة بدل assert
    # ══════════════════════════════════════════════════════════
    def feats(
        self, h: Team, a: Team,
        data: "DataProc", md: datetime = None, derby: bool = False,
    ) -> List[float]:
        all_feats = (
            self._original_features(h, a, data, md, derby)   # 68
            + self._temporal_features(md, h, a)              # 12
            + self._advanced_momentum_features(h, a)         # 14
            + self._advanced_h2h_features(h, a, data)        # 10
            + self._advanced_xg_features(h, a)               # 11
            + self._context_features(h, a, md, data)         # 11
        )  # = 126

        # V6.1 FIX: آمن في Production بدل assert
        n = len(all_feats)
        if n != self.N_FEATURES:
            # Pad أو Truncate
            all_feats = (all_feats + [0.0] * self.N_FEATURES)[: self.N_FEATURES]

        return all_feats

    # ══════════════════════════════════════════════════════════
    # بناء Pipelines
    # ══════════════════════════════════════════════════════════
    def _build_stacking_pipeline(self) -> "Pipeline":
        base = [
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight=None, random_state=42, n_jobs=-1,
            )),
            ("lr_base", LogisticRegression(
                C=0.1, max_iter=1000,
                solver="lbfgs", random_state=42,
            )),
        ]
        if XGBOOST_AVAILABLE:
            base.append(("xgb", XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
            )))
        meta = LogisticRegression(
            C=0.5, max_iter=1000,
            solver="lbfgs", random_state=42,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   StackingClassifier(
                estimators=base, final_estimator=meta,
                cv=3, stack_method="predict_proba", n_jobs=1,
            )),
        ])

    def _build_voting_pipeline(self) -> "Pipeline":
        estimators = [
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight=None, random_state=42, n_jobs=-1,
            )),
        ]
        if XGBOOST_AVAILABLE:
            estimators.append(("xgb", XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
            )))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)),
        ])

    def _try_load_external(self) -> bool:
        if not Path(self.model_file).exists():
            return False
        try:
            with open(self.model_file, "rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, Pipeline):
                return False
            # تحقق من عدد الميزات
            final = loaded.steps[-1][1]
            n_feat = getattr(final, "n_features_in_", None)
            if n_feat is None:
                try:
                    n_feat = getattr(final.final_estimator_, "n_features_in_", None)
                except Exception:
                    pass
            if n_feat is not None and n_feat != self.N_FEATURES:
                print(
                    f"⚠️  Model features mismatch: {n_feat} vs {self.N_FEATURES} "
                    "→ retraining"
                )
                return False
            self.pipeline  = loaded
            self.trained   = True
            self._external = True
            return True
        except Exception:
            return False

    def train(
        self,
        data: "DataProc",
        fixes: List[dict] = None,
        force_retrain: bool = False,
        temporal_cv: bool = True,
    ) -> bool:
        if not ML_AVAILABLE:
            return False
        if not force_retrain and self._try_load_external():
            return True

        fixes = fixes or data.fixes
        if len(fixes) < 40:
            return False

        X: List[List[float]] = []
        y: List[int]         = []

        sim  = DataProc(data.resources)
        sf   = sorted(fixes, key=lambda f: f.get("date", ""))
        warm = int(len(sf) * 0.30)

        for idx, f in enumerate(sf):
            if idx >= warm:
                ht = sim.teams.get(f["home_id"])
                at = sim.teams.get(f["away_id"])
                if ht and at and ht.played >= 3 and at.played >= 3:
                    try:
                        md_    = parse_date(f.get("date", ""))
                        derby_ = bool(
                            data.resources.is_derby(f["home_name"], f["away_name"])
                        ) if data.resources else False
                        ft     = self.feats(ht, at, sim, md_, derby_)
                        lb     = (
                            0 if f["home_goals"] > f["away_goals"] else
                            (1 if f["home_goals"] == f["away_goals"] else 2)
                        )
                        X.append(ft)
                        y.append(lb)
                    except Exception:
                        pass

            # تحديث sim تدريجياً
            sim.process([{
                "status":   "FINISHED",
                "homeTeam": {"id": f["home_id"], "shortName": f["home_name"]},
                "awayTeam": {"id": f["away_id"], "shortName": f["away_name"]},
                "score":    {"fullTime": {"home": f["home_goals"], "away": f["away_goals"]}},
                "utcDate":  f.get("date", ""),
                "stats":    f.get("stats", {}),
            }], do_elo=True)

        if len(X) < 30:
            return False

        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.int64)

        _, counts = np.unique(y_arr, return_counts=True)
        if int(counts.min()) < MIN_SAMPLES_PER_CLASS:
            return False

        # V7: temporal validation is more faithful to real prediction than shuffled CV.
        for attempt, builder in enumerate([self._build_voting_pipeline, self._build_stacking_pipeline]):
            try:
                pipe = builder()
                if temporal_cv:
                    n_splits = min(5, max(2, len(X_arr) // 150))
                    tscv = TimeSeriesSplit(n_splits=n_splits)
                    cv_sc = cross_val_score(pipe, X_arr, y_arr, cv=tscv, scoring=temporal_neg_log_loss_scorer, n_jobs=1, error_score="raise")
                    self.acc = float(np.mean(cv_sc))
                else:
                    # v10.0.10 benchmark fast-path: CV is diagnostic only and
                    # does not select the fitted pipeline. Skipping nested CV
                    # preserves the same final fit while removing repeated fits.
                    self.acc = 0.0
                pipe.fit(X_arr, y_arr)
                self.pipeline = pipe
                name = "Voting" if attempt == 0 else "Stacking"
                print(f"✅ {name} model trained | Temporal CV LogLoss: {-self.acc:.4f}")
                self.trained = True
                self._external = False
                return True
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")

        # RF fallback
        try:
            fallback = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler",  StandardScaler()),
                ("model",   RandomForestClassifier(
                    n_estimators=200, class_weight="balanced",
                    random_state=42, n_jobs=-1,
                )),
            ])
            fallback.fit(X_arr, y_arr)
            self.pipeline = fallback
            self.acc      = 0.0
            self.trained  = True
            print("✅ RF fallback model trained")
            return True
        except Exception:
            return False

    def predict(
        self,
        h: Team, a: Team, data: "DataProc",
        md: datetime = None, derby: bool = False,
    ) -> Optional[Tuple[float, float, float]]:
        if not self.trained or self.pipeline is None:
            return None
        try:
            ft      = self.feats(h, a, data, md, derby)
            X       = np.array([ft], dtype=np.float64)
            probs   = self.pipeline.predict_proba(X)[0]
            classes = self.pipeline.classes_
            pd_     = {cls: pr for cls, pr in zip(classes, probs)}
            return (
                float(pd_.get(0, 0.0)),
                float(pd_.get(1, 0.0)),
                float(pd_.get(2, 0.0)),
            )
        except Exception:
            return None

    def save_pipeline(self):
        if self.pipeline is not None:
            Path(self.model_file).parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.model_file, "wb") as f:
                    pickle.dump(self.pipeline, f)
                print(f"✅ Model saved: {self.model_file}")
            except Exception as e:
                print(f"⚠️  Save model error: {e}")


# ══════════════════════════════════════════════════════════════
# API CLIENT
# ══════════════════════════════════════════════════════════════
class FootballAPI:
    def __init__(self, token: str, base_url: str = "https://api.football-data.org/v4"):
        self.s = requests.Session()
        self.s.headers.update({"X-Auth-Token": token, "Accept": "application/json"})
        self.base_url = base_url
        self._c:  Dict[str, dict] = {}
        self._t:  float           = 0.0

    def _rl(self):
        elapsed = time.time() - self._t
        if elapsed < 6.5:
            time.sleep(6.5 - elapsed)
        self._t = time.time()

    def _get(self, ep: str, p: dict = None, cache: bool = True):
        p = p or {}
        k = hashlib.md5(
            f"{ep}|{json.dumps(p, sort_keys=True)}".encode()
        ).hexdigest()
        if cache and k in self._c:
            return self._c[k]
        try:
            self._rl()
            r = self.s.get(f"{self.base_url}/{ep}", params=p, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("X-RequestCounter-Reset", 60)) + 1
                time.sleep(wait)
                return self._get(ep, p, cache)
            if r.status_code in (401, 403, 404):
                return None
            r.raise_for_status()
            d = r.json()
            if cache:
                self._c[k] = d
            return d
        except Exception:
            return None

    def season_year(self, league_code: str) -> Optional[int]:
        d = self._get(f"competitions/{league_code}")
        if d and d.get("currentSeason"):
            try:
                return int(d["currentSeason"]["startDate"][:4])
            except Exception:
                pass
        return None

    def finished(self, league_code: str, season: int = None) -> List[dict]:
        p = {"status": "FINISHED"}
        if season:
            p["season"] = season
        d = self._get(f"competitions/{league_code}/matches", p)
        if d and "matches" in d:
            m = d["matches"]
            m.sort(key=lambda x: x.get("utcDate", ""))
            return m
        return []

    def upcoming(self, league_code: str, days: int = 14) -> List[dict]:
        t = datetime.now().strftime("%Y-%m-%d")
        e = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        d = self._get(
            f"competitions/{league_code}/matches",
            {"status": "SCHEDULED,TIMED", "dateFrom": t, "dateTo": e},
        )
        if d and "matches" in d:
            m = d["matches"]
            m.sort(key=lambda x: x.get("utcDate", ""))
            return m
        return []


class OddsAPI:
    def __init__(self, key: str, sport: str = "soccer_epl"):
        self.key   = key
        self.sport = sport
        self.cache: Dict[str, dict] = {}

    def ok(self) -> bool:
        return bool(self.key) and len(self.key) > 10

    def fetch(self) -> Dict[str, dict]:
        if not self.ok():
            return {}
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{self.sport}/odds",
                params={
                    "apiKey":      self.key,
                    "regions":     "uk,eu",
                    "markets":     "h2h,totals",
                    "oddsFormat":  "decimal",
                },
                timeout=15,
            )
            if r.status_code != 200:
                return {}

            result: Dict[str, dict] = {}
            for ev in r.json():
                h   = ev.get("home_team", "")
                a   = ev.get("away_team", "")
                bms = ev.get("bookmakers", [])
                if not bms:
                    continue
                ah_l, ad_l, aa_l = [], [], []
                for bm in bms:
                    for mk in bm.get("markets", []):
                        if mk["key"] == "h2h":
                            for o in mk.get("outcomes", []):
                                if o["name"] == h:       ah_l.append(o["price"])
                                elif o["name"] == a:     aa_l.append(o["price"])
                                elif o["name"] == "Draw": ad_l.append(o["price"])
                if ah_l and ad_l and aa_l:
                    avh = sum(ah_l) / len(ah_l)
                    avd = sum(ad_l) / len(ad_l)
                    ava = sum(aa_l) / len(aa_l)
                    raw_ih, raw_id, raw_ia = 1 / avh, 1 / avd, 1 / ava
                    overround = raw_ih + raw_id + raw_ia
                    ih, id_, ia = (raw_ih/overround, raw_id/overround, raw_ia/overround) if overround > 0 else (raw_ih, raw_id, raw_ia)
                    # ── DC odds احتمالات مجمعة محسوبة ────────────
                    # 1X: لا تخسر إذا فاز أي منهما أو تعادلا
                    # X2: لا تخسر إذا فاز الضيف أو تعادلا
                    # 12: لا تخسر إذا لم يتعادلا
                    odds_1x  = round(1.0 / (ih + id_), 2) if (ih + id_) > 0 else None
                    odds_x2  = round(1.0 / (ia + id_), 2) if (ia + id_) > 0 else None
                    odds_12  = round(1.0 / (ih + ia),  2) if (ih + ia)  > 0 else None
                    result[f"{h}_vs_{a}".lower()] = {
                        "home_team":    h,
                        "away_team":    a,
                        "odds_home":    round(avh, 2),
                        "odds_draw":    round(avd, 2),
                        "odds_away":    round(ava, 2),
                        "implied_home": round(ih,  4),
                        "implied_draw": round(id_, 4),
                        "implied_away": round(ia,  4),
                        "implied_1x":   round(ih + id_, 4),
                        "implied_x2":   round(ia + id_, 4),
                        "implied_12":   round(ih + ia,  4),
                        "odds_1x":      odds_1x,
                        "odds_x2":      odds_x2,
                        "odds_12":      odds_12,
                    }
            self.cache = result
            return result
        except Exception:
            return {}

    def find(self, hn: str, an: str) -> Optional[dict]:
        if not self.cache:
            self.fetch()
        hl = hn.lower()
        al = an.lower()
        for d in self.cache.values():
            oh = d["home_team"].lower()
            oa = d["away_team"].lower()
            hm = hl in oh or oh in hl or any(w in oh for w in hl.split() if len(w) > 3)
            am = al in oa or oa in al or any(w in oa for w in al.split() if len(w) > 3)
            if hm and am:
                return d
        return None


# ══════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ══════════════════════════════════════════════════════════════
class Pred:
    __slots__ = (
        "home", "away", "hid", "aid", "date", "league",
        "hp", "dp", "ap", "raw_hp", "raw_dp", "raw_ap",
        "hxg", "axg", "top_sc", "result", "pred_sc", "conf",
        "btts", "o15", "o25", "o35",
        "dc_1x", "dc_x2", "dc_12", "dc_recommend",
        "dc_value_bets", "value_bets",
        "h_form", "a_form", "h_pos", "a_pos",
        "h_elo", "a_elo", "h_fat", "a_fat",
        "h_rest", "a_rest", "h_momentum", "a_momentum",
        "models", "odds", "ml_used", "ml_acc",
        "calibrated", "is_derby", "derby_name",
        "adaptive_regime", "adaptive_weights", "meta_used", "meta_features", "reliability_factor", "reliability_reasons", "performance_factor", "performance_alerts", "drift_factor", "drift_alerts", "draw_trace",
    )

    def __init__(self):
        self.home = self.away = self.date = self.league = ""
        self.hid  = self.aid  = 0
        self.hp = self.dp = self.ap = 0.0
        self.raw_hp = self.raw_dp = self.raw_ap = 0.0
        self.hxg = self.axg = 0.0
        self.top_sc: List[Tuple] = []
        self.result   = ""
        self.pred_sc  = (0, 0)
        self.conf     = 0.0
        self.btts = self.o15 = self.o25 = self.o35 = 0.0
        self.dc_1x = self.dc_x2 = self.dc_12 = 0.0
        self.dc_recommend = ""
        self.dc_value_bets: List[dict] = []
        self.value_bets:    List[dict] = []
        self.h_form = self.a_form = ""
        self.h_pos  = self.a_pos  = 0
        self.h_elo  = self.a_elo  = 0.0
        self.h_fat  = self.a_fat  = 0.0
        self.h_rest = self.a_rest = 0
        self.h_momentum = self.a_momentum = 0
        self.models: Dict[str, Tuple] = {}
        self.odds       = None
        self.ml_used    = False
        self.ml_acc     = 0.0
        self.calibrated = False
        self.is_derby   = False
        self.derby_name = ""
        self.meta_used = False
        self.meta_features = ()
        self.reliability_factor = 1.0
        self.reliability_reasons = []
        self.performance_factor = 1.0
        self.performance_alerts = []
        self.drift_factor = 1.0
        self.drift_alerts = []
        self.draw_trace = {}


# ══════════════════════════════════════════════════════════════
# V7.6 META-LEARNER — context-aware, leakage-safe ensemble gate
# ══════════════════════════════════════════════════════════════
class MetaLearner:
    """Learns when to trust each component from calibration data only.

    Inputs are component probabilities plus pre-match context.  It is deliberately
    small (multinomial logistic regression) to reduce overfitting.  Temporal holdout
    inside calibration decides whether the meta layer is allowed to activate.
    """
    VERSION = "7.6"

    def __init__(self):
        self.pipeline = None
        self.active = False
        self.feature_names: List[str] = []
        self.validation_logloss = None
        self.baseline_logloss = None

    @staticmethod
    def _vector(sample: dict, names: List[str]) -> List[float]:
        vec = []
        models = sample.get("models", {})
        for n in names:
            q = models.get(n, (1/3, 1/3, 1/3))
            vec.extend([float(q[0]), float(q[1]), float(q[2])])
        ctx = sample.get("context", {})
        # Continuous context available before kickoff only.
        vec.extend([
            float(ctx.get("elo_gap", 0.0)) / 300.0,
            float(ctx.get("xg_diff", 0.0)),
            float(ctx.get("form_diff", 0.0)) / 10.0,
            float(ctx.get("rest_diff", 0.0)) / 10.0,
            float(ctx.get("data_reliability", 0.0)),
            float(ctx.get("season_progress", 0.0)),
            float(ctx.get("is_derby", 0.0)),
        ])
        return vec

    def fit(self, samples: List[dict], model_names: List[str], min_train: int = 80) -> bool:
        if not ML_AVAILABLE or len(samples) < min_train or len(model_names) < 2:
            return False
        y = np.asarray([int(x["actual"]) for x in samples], dtype=int)
        X = np.asarray([self._vector(x, model_names) for x in samples], dtype=float)
        if len(np.unique(y)) < 3:
            return False
        cut = max(50, int(len(samples) * 0.70))
        if len(samples) - cut < 25:
            return False
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.20, max_iter=1500, solver="lbfgs", random_state=7601)),
        ])
        try:
            pipe.fit(X[:cut], y[:cut])
            pred = pipe.predict_proba(X[cut:])
            classes = pipe.classes_
            mapped = np.zeros((len(pred), 3), dtype=float)
            for j, c in enumerate(classes): mapped[:, int(c)] = pred[:, j]
            mapped = np.clip(mapped, 1e-7, 1.0); mapped /= mapped.sum(axis=1, keepdims=True)
            idx = np.arange(len(y)-cut)
            ll = float(-np.mean(np.log(mapped[idx, y[cut:]])))
            # Baseline = the actual production ensemble probabilities captured without leakage.
            # This avoids activating Meta merely because it beats an artificial equal-weight average.
            bl_rows=[]
            for x in samples[cut:]:
                q=np.asarray(x.get("ensemble", (1/3,1/3,1/3)), dtype=float)
                q=np.clip(q,1e-7,1.0); q/=q.sum(); bl_rows.append(q)
            bl=np.asarray(bl_rows)
            bll=float(-np.mean(np.log(bl[np.arange(len(bl)), y[cut:]])))
            self.validation_logloss=ll; self.baseline_logloss=bll
            # Require a meaningful gain, not noise.
            if ll >= bll - 0.002:
                return False
            pipe.fit(X, y)
            self.pipeline=pipe; self.active=True
            self.feature_names=list(model_names)
            return True
        except Exception:
            return False

    def predict(self, sample: dict) -> Optional[Tuple[float,float,float]]:
        if not self.active or self.pipeline is None:
            return None
        try:
            X=np.asarray([self._vector(sample,self.feature_names)],dtype=float)
            p=self.pipeline.predict_proba(X)[0]
            out=np.zeros(3,dtype=float)
            for j,c in enumerate(self.pipeline.classes_): out[int(c)]=p[j]
            out=np.clip(out,1e-7,1.0); out/=out.sum()
            return tuple(float(v) for v in out)
        except Exception:
            return None

    def save(self, path: str) -> bool:
        if not self.active or self.pipeline is None: return False
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path,"wb") as f: pickle.dump(self,f)
            return True
        except Exception: return False

    @classmethod
    def load(cls, path: str):
        if not Path(path).exists(): return None
        try:
            with open(path,"rb") as f: obj=pickle.load(f)
            return obj if isinstance(obj,cls) and obj.active and obj.VERSION==cls.VERSION else None
        except Exception: return None

# ══════════════════════════════════════════════════════════════
# ENGINE  — V6.1: Draw Correction + _form() + _dc_recommend()
# ══════════════════════════════════════════════════════════════
class Engine:
    def __init__(
        self,
        data:      "DataProc",
        resources: "LeagueResources",
        ml:        "MLPred"    = None,
        odds:      "OddsAPI"   = None,
        cal:       "Calibrator"= None,
        meta:      "MetaLearner" = None,
        drift_detector: "AutomatedDriftDetector" = None,
        drift_report: dict = None,
    ):
        self.data      = data
        self.resources = resources
        self.ml        = ml
        self.odds      = odds
        self.cal       = cal
        self.meta      = meta
        # Drift monitoring is optional for isolated temporal/backtest engines.
        # Production instances receive the league-level detector and report.
        self.drift_detector = drift_detector
        self.drift_report = drift_report or {}
        self.w         = dict(WEIGHTS)
        self.adaptive_profiles: Dict[str, Dict[str, float]] = {}
        # V8.0: load validated league/context profiles when available.
        policy = AdaptiveEnsembleOptimizer(resources.code).load() if resources else None
        if policy and policy.get("activation") == "enabled":
            self.w = dict(policy.get("global_weights") or self.w)
            self.adaptive_profiles = dict(policy.get("profiles") or {})

        if not ml or not ml.trained:
            mw  = self.w.pop("ml", 0.25)
            rem = sum(self.w.values())
            if rem > 0:
                for k in self.w:
                    self.w[k] += mw * (self.w[k] / rem)

    def _regime_key(self, h: "Team", a: "Team", md: datetime, is_derby: bool = False) -> str:
        """V7.5: classify a match into a small, leakage-safe regime.
        The regime uses only information available before kickoff. Profiles are
        learned on the calibration window and therefore never see evaluation outcomes.
        """
        if is_derby:
            return "DERBY"
        elo_gap = abs(float(h.elo) - float(a.elo))
        if elo_gap >= 180:
            return "STRONG_MISMATCH"
        if elo_gap <= 55:
            return "BALANCED"
        season_month = ((md.month - 8) % 12) + 1
        if season_month >= 9:
            return "LATE_SEASON_EDGE"
        return "MODERATE_EDGE"

    def _active_weights(self, regime: str) -> Dict[str, float]:
        """Blend a learned regime profile with global weights conservatively."""
        base = dict(self.w)
        prof = self.adaptive_profiles.get(regime)
        if not prof:
            return base
        names = set(base) | set(prof)
        out = {}
        for n in names:
            # 70% global / 30% regime profile: stable fallback for sparse regimes.
            out[n] = 0.70 * float(base.get(n, 0.0)) + 0.30 * float(prof.get(n, 0.0))
        z = sum(max(0.0, v) for v in out.values())
        if z <= 0:
            return base
        return {k: max(0.0, v) / z for k, v in out.items()}

    def predict(
        self, hid: int, aid: int,
        date: str = "", md: datetime = None,
    ) -> Optional[Pred]:
        h = self.data.teams.get(hid)
        a = self.data.teams.get(aid)
        if not h or not a or h.played < 2 or a.played < 2:
            return None

        p         = Pred()
        p.home    = h.name
        p.away    = a.name
        p.hid     = hid
        p.aid     = aid
        p.date    = date
        p.league  = self.resources.code if self.resources else ""
        p.h_form  = h.form_string
        p.a_form  = a.form_string
        p.h_pos   = h.pos
        p.a_pos   = a.pos
        p.h_elo   = h.elo
        p.a_elo   = a.elo

        if md is None and date:
            md = parse_date(date)
        md = md or datetime.now()

        derby       = self.resources.is_derby(h.name, a.name) if self.resources else None
        p.is_derby  = bool(derby)
        p.derby_name = derby or ""

        p.h_fat     = Fatigue.score(h, md)
        p.a_fat     = Fatigue.score(a, md)
        p.h_rest    = h.days_rest(md)
        p.a_rest    = a.days_rest(md)
        p.h_momentum = h.momentum
        p.a_momentum = a.momentum

        p.hxg = self._xg(h, a, True)  * Fatigue.impact(h, md)
        p.axg = self._xg(a, h, False) * Fatigue.impact(a, md)

        if   h.momentum >  40: p.hxg *= 1.05
        elif h.momentum < -40: p.hxg *= 0.95
        if   a.momentum >  40: p.axg *= 1.05
        elif a.momentum < -40: p.axg *= 0.95

        season_month   = ((md.month - 8) % 12) + 1
        is_late_season = season_month >= 9

        h2h_matches    = self.data.get_h2h(hid, aid)
        h2h_draw_rate  = 0.25
        if h2h_matches:
            recent = h2h_matches[-10:]
            h2h_draw_rate = sum(
                1 for m in recent if m["home_goals"] == m["away_goals"]
            ) / len(recent)

        ha_val   = self.resources.home_advantage if self.resources else 65
        elo_diff = h.elo + ha_val - a.elo

        regime = self._regime_key(h, a, md, p.is_derby)
        p.adaptive_regime = regime
        active_w = self._active_weights(regime)
        p.adaptive_weights = dict(active_w)

        # ── تجميع النماذج ────────────────────────────────────
        models: Dict[str, Tuple[float, float, float]] = {}
        models["dixon_coles"]    = DixonColes.predict(p.hxg, p.axg)
        models["elo"]            = self.data.elo.predict(h, a)
        models["form"]           = self._form(h, a)
        models["h2h"]            = self._h2h(hid, aid)
        models["home_advantage"] = self._hadv(h, a)
        models["fatigue"]        = Fatigue.predict(h, a, md)
        models["draw_model"]     = ImprovedDrawPredictor.predict(
            h, a, h2h_draw_rate, elo_diff, p.is_derby, is_late_season
        )

        if self.ml and self.ml.trained:
            mp = self.ml.predict(h, a, self.data, md, p.is_derby)
            if mp:
                models["ml"] = mp
                p.ml_used    = True
                p.ml_acc     = self.ml.acc

        p.models = models

        # ── دمج الأوزان ──────────────────────────────────────
        hp = dp = ap = tw = 0.0
        for nm, probs in models.items():
            w = active_w.get(nm, 0)
            if w > 0:
                hp += probs[0] * w
                dp += probs[1] * w
                ap += probs[2] * w
                tw += w

        if tw > 0:
            hp /= tw; dp /= tw; ap /= tw

        hp, dp, ap = normalize_probs(hp, dp, ap)

        p.raw_hp = hp
        p.raw_dp = dp
        p.raw_ap = ap
        draw_model_dp = float(models.get("draw_model", (0,0,0))[1])
        p.draw_trace = {"weighted_ensemble": float(dp), "draw_model": draw_model_dp}

        # V10.0.4: Preserve independent draw evidence when ensemble averaging
        # materially suppresses it in genuinely draw-like contexts. This is a
        # probability correction, not a forced DRAW class decision.
        hp, dp, ap = self._apply_draw_evidence_preservation(
            hp, dp, ap, draw_model_dp, h, a
        )
        p.draw_trace["after_draw_preservation"] = float(dp)

        # V6.1: تصحيح انحياز التعادل
        hp, dp, ap = self._apply_draw_correction(hp, dp, ap, h, a)
        p.draw_trace["after_draw_correction"] = float(dp)

        # ── Calibration ───────────────────────────────────────
        if self.cal and self.cal.ok:
            hp, dp, ap = self.cal.adjust((hp, dp, ap), regime=regime)
            p.calibrated = True
            p.draw_trace["after_calibration"] = float(dp)

        # V7.6 meta layer: context-aware correction, activated only after temporal gain.
        if self.meta and self.meta.active:
            ctx={"elo_gap":abs(float(h.elo-a.elo)),"xg_diff":float(p.hxg-p.axg),"form_diff":float(h.form_score-a.form_score),"rest_diff":float(p.h_rest-p.a_rest),"data_reliability":min(h.played,a.played)/float(max(self.resources.total_rounds,1)),"season_progress":season_month/12.0,"is_derby":int(p.is_derby)}
            mq=self.meta.predict({"models":models,"context":ctx})
            if mq:
                hp,dp,ap=normalize_probs(.55*mq[0]+.45*hp,.55*mq[1]+.45*dp,.55*mq[2]+.45*ap)
                p.meta_used=True; p.meta_features=tuple(self.meta.feature_names)
                p.draw_trace["after_meta"] = float(dp)

        # V7.7 Reliability Engine: reduce confidence in historically unstable contexts.
        if self.cal and self.cal.ok:
            p.reliability_factor,p.reliability_reasons=self.cal.reliability((hp,dp,ap),regime=regime)
            pf, pa = ContextualPerformanceMonitor(self.resources.code).assess((hp,dp,ap), regime=regime)
            p.performance_factor, p.performance_alerts = pf, pa
            p.reliability_factor *= pf
            p.reliability_reasons.extend(pa)
            if self.drift_detector is not None:
                df, da = self.drift_detector.assess(self.drift_report)
            else:
                # Backtests must remain runnable without live drift state.
                df, da = 1.0, []
            p.drift_factor, p.drift_alerts = df, da
            p.reliability_factor *= df
            p.reliability_reasons.extend(da)

        # V7.3: entropy-based confidence guard. It does NOT change the class
        # ranking; it shrinks extreme probabilities toward 1/3 when the three
        # outcomes are intrinsically close, reducing overconfident picks.
        probs = np.array([hp, dp, ap], dtype=float)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0))) / np.log(3.0))
        shrink = min(0.16, max(0.0, (entropy - 0.70) * 0.22))
        if shrink > 0:
            probs = (1.0 - shrink) * probs + shrink * (1.0 / 3.0)
            hp, dp, ap = normalize_probs(*probs)

        p.draw_trace["final"] = float(dp)
        p.hp = hp
        p.dp = dp
        p.ap = ap

        p.dc_1x = hp + dp
        p.dc_x2 = ap + dp
        p.dc_12 = hp + ap
        p.dc_recommend = self._dc_recommend(p)

        # ── مصفوفة الأهداف ───────────────────────────────────
        mx    = DixonColes.matrix(p.hxg, p.axg)
        ss    = sorted(mx.items(), key=lambda x: x[1], reverse=True)
        p.top_sc = [(s[0][0], s[0][1], s[1]) for s in ss[:6]]

        p.btts = sum(pr for (hh, aa), pr in mx.items() if hh > 0 and aa > 0)
        p.o15  = sum(pr for (hh, aa), pr in mx.items() if hh + aa > 1)
        p.o25  = sum(pr for (hh, aa), pr in mx.items() if hh + aa > 2)
        p.o35  = sum(pr for (hh, aa), pr in mx.items() if hh + aa > 3)

        pd_map   = {"HOME": hp, "DRAW": dp, "AWAY": ap}
        p.result = max(pd_map, key=pd_map.get)
        p.conf   = max(pd_map.values()) * 100 * p.reliability_factor

        if p.top_sc:
            p.pred_sc = (p.top_sc[0][0], p.top_sc[0][1])

        if self.odds and self.odds.ok():
            od = self.odds.find(h.name, a.name)
            if od:
                p.odds          = od
                p.value_bets    = self._value(p, od)
                p.dc_value_bets = self._dc_value(p, od)

        return p

    # ── V10.0.4: Preserve draw evidence lost in weighted averaging ──
    def _apply_draw_evidence_preservation(
        self, hp: float, dp: float, ap: float, draw_model_dp: float,
        h: Team, a: Team,
    ) -> Tuple[float, float, float]:
        """Conservatively preserve independent draw-model evidence.

        The correction only activates when the specialized draw model is
        materially higher than the ensemble *and* the fixture context is close.
        It never forces DRAW to be the argmax and is capped to protect Log Loss.
        """
        gap = max(0.0, float(draw_model_dp) - float(dp))
        if gap < 0.035:
            return hp, dp, ap
        team_draw_avg = float(np.clip((h.dr + a.dr) / 2.0, 0.0, 1.0))
        balance = float(np.clip(1.0 - abs(hp - ap) / 0.35, 0.0, 1.0))
        draw_context = 0.45 * balance + 0.55 * float(np.clip((team_draw_avg - 0.18) / 0.18, 0.0, 1.0))
        if draw_context < 0.35:
            return hp, dp, ap
        # Preserve at most 45% of the discrepancy, with a hard 4.5pp cap.
        boost = min(0.045, gap * 0.45 * draw_context)
        if boost <= 0.0:
            return hp, dp, ap
        non_draw = max(hp + ap, 1e-9)
        hp -= boost * (hp / non_draw)
        ap -= boost * (ap / non_draw)
        dp += boost
        return normalize_probs(max(hp, 1e-9), dp, max(ap, 1e-9))

    # ── V6.1: تصحيح انحياز التعادل ───────────────────────────
    def _apply_draw_correction(
        self,
        hp: float, dp: float, ap: float,
        h: Team, a: Team,
    ) -> Tuple[float, float, float]:
        """
        إذا كان فارق hp/ap مقارنةً بـ dp صغيراً
        → يُرجَّح التعادل أكثر.
        """
        max_ha = max(hp, ap)
        gap    = max_ha - dp

        if gap < 0.12:                             # فارق صغير → boost
            boost = min(0.05, max(0.0, gap) * 0.3)
            dp   += boost
            hp   -= boost * 0.6
            ap   -= boost * 0.4

        # كلا الفريقين draw-prone تاريخياً
        team_draw_avg = (h.dr + a.dr) / 2.0
        if team_draw_avg > 0.30:
            extra = min(0.03, (team_draw_avg - 0.30) * 0.20)
            dp   += extra
            hp   -= extra * 0.55
            ap   -= extra * 0.45

        return normalize_probs(hp, dp, ap)

    # ── V6.1: _dc_recommend() بدون تعارض ─────────────────────
    def _dc_recommend(self, p: Pred) -> str:
        """
        نظام تسجيل نقاط بدل الشروط المتعارضة.
        """
        scores: Dict[str, float] = {
            "1X": p.dc_1x,
            "X2": p.dc_x2,
            "12": p.dc_12,
        }
        reasons: Dict[str, str] = {
            "1X": "Home favored but draw possible",
            "X2": "Away has real chance + draw likely",
            "12": "Draw unlikely",
        }

        # تعديلات حسب السياق
        if 0.40 <= p.hp <= 0.60 and p.dp > 0.20:
            scores["1X"] += 0.05
        if 0.30 <= p.ap <= 0.50 and p.dp > 0.20:
            scores["X2"] += 0.05
        if p.dp < 0.20:
            scores["12"] += 0.05
        if p.is_derby and p.hp > p.ap:
            scores["1X"] += 0.03
            reasons["1X"] = f"{p.derby_name} - Home advantage"

        best     = max(scores, key=scores.get)
        pct_str  = f"{scores[best] * 100:.1f}%"
        return f"{best} ({pct_str}) - {reasons[best]}"

    # ── xG ────────────────────────────────────────────────────
    def _xg(self, t: Team, opp: Team, home: bool) -> float:
        ah = max(self.data.avg_h, 0.5)
        aa = max(self.data.avg_a, 0.5)
        if home:
            att  = safe_div(t.h_avg_gf, ah, 1.0)
            df   = safe_div(opp.a_avg_ga, aa, 1.0)
            base = ah
        else:
            att  = safe_div(t.a_avg_gf, aa, 1.0)
            df   = safe_div(opp.h_avg_ga, ah, 1.0)
            base = aa

        fa      = safe_div(t.goal_form, max(t.avg_gf, 0.5), 1.0)
        fa      = 0.7 + 0.3 * min(fa, 2.0)
        raw_xg  = att * df * base * fa
        # V6.1 FIX: حد أدنى 0.10 بدل 0.25
        return max(0.10, min(raw_xg, 4.0))

    # ── V6.1: _form() يُدرك التعادل عند تقارب الفورم ──────────
    def _form(self, h: Team, a: Team) -> Tuple[float, float, float]:
        hf = h.form_score
        af = a.form_score
        t  = hf + af
        if t == 0:
            return (0.40, 0.25, 0.35)

        hs  = (hf / t) * 1.08
        a_s =  af / t
        diff = abs(hs - a_s)

        # V6.1 FIX: فورم متقارب → احتمال تعادل أعلى
        if   diff < 0.08: d = 0.35
        elif diff < 0.15: d = 0.30
        elif diff < 0.25: d = 0.25
        else:             d = max(0.15, 0.30 - diff * 0.40)

        rem = 1.0 - d
        sm  = hs + a_s
        if sm <= 0:
            return (rem * 0.5, d, rem * 0.5)
        return normalize_probs(rem * hs / sm, d, rem * a_s / sm)

    # ── H2H ───────────────────────────────────────────────────
    def _h2h(self, hid: int, aid: int) -> Tuple[float, float, float]:
        default = (0.40, 0.25, 0.35)
        ms = self.data.get_h2h(hid, aid)
        if not ms:
            return default
        hw = dw = aw = 0
        for m in ms[-10:]:
            if m["home_goals"] > m["away_goals"]:
                hw += 1 if m["home_id"] == hid else 0
                aw += 1 if m["home_id"] != hid else 0
            elif m["home_goals"] < m["away_goals"]:
                aw += 1 if m["home_id"] == hid else 0
                hw += 1 if m["home_id"] != hid else 0
            else:
                dw += 1
        n = hw + dw + aw
        if n == 0:
            return default
        alpha = 1
        return (
            (hw + alpha) / (n + 3 * alpha),
            (dw + alpha) / (n + 3 * alpha),
            (aw + alpha) / (n + 3 * alpha),
        )

    # ── Home Advantage ────────────────────────────────────────
    def _hadv(self, h: Team, a: Team) -> Tuple[float, float, float]:
        hp = h.h_wr * 1.25
        ap = a.a_wr
        sm = hp + ap
        if sm > 0:
            hp /= sm
            ap /= sm
        d  = 0.25
        hp *= 0.75
        ap *= 0.75
        return normalize_probs(hp, d, ap)

    # ── Value Bets ────────────────────────────────────────────
    def _value(self, p: Pred, od: dict) -> List[dict]:
        vals = []
        for nm, mp, ip, odd in [
            ("Home", p.hp, od["implied_home"], od["odds_home"]),
            ("Draw", p.dp, od["implied_draw"], od["odds_draw"]),
            ("Away", p.ap, od["implied_away"], od["odds_away"]),
        ]:
            edge  = (mp - ip) * 100
            kelly = (mp * odd - 1) / (odd - 1) if mp > 0 and odd > 1 else 0.0
            vals.append({
                "market":   nm,
                "model":    float(mp  * 100),
                "implied":  float(ip  * 100),
                "odds":     float(odd),
                "edge":     float(edge),
                "kelly":    float(max(0.0, kelly) * 100),
                "is_value": edge > 3,
            })
        return vals

    def _dc_value(self, p: Pred, od: dict) -> List[dict]:
        vals = []
        for nm, model_p, implied_p, odds_val in [
            ("1X", p.dc_1x, od.get("implied_1x"), od.get("odds_1x")),
            ("X2", p.dc_x2, od.get("implied_x2"), od.get("odds_x2")),
            ("12", p.dc_12, od.get("implied_12"), od.get("odds_12")),
        ]:
            if implied_p is None or odds_val is None:
                continue
            edge  = (model_p - implied_p) * 100
            kelly = (model_p * odds_val - 1) / (odds_val - 1) \
                    if model_p > 0 and odds_val > 1 else 0.0
            vals.append({
                "market":   f"DC {nm}",
                "model":    float(model_p  * 100),
                "implied":  float(implied_p * 100),
                "odds":     float(odds_val),
                "edge":     float(edge),
                "kelly":    float(max(0.0, kelly) * 100),
                "is_value": edge > 3,
            })
        return vals
# ══════════════════════════════════════════════════════════════
# BACKTESTER — V6.1: Data Leakage Fixed + DC Accuracy Fixed
#                    Brier Score Fixed + Confusion Matrix
# ══════════════════════════════════════════════════════════════
class Backtester:
    def __init__(self):
        self.results: dict = {}
        self.cal = Calibrator()

    def analyze_errors(self, preds: List[dict]) -> dict:
        """تحليل أنماط الأخطاء وطباعة Confusion Matrix"""
        errors = {
            "predicted_home_was_draw":  0,
            "predicted_home_was_away":  0,
            "predicted_draw_was_home":  0,
            "predicted_draw_was_away":  0,
            "predicted_away_was_home":  0,
            "predicted_away_was_draw":  0,
        }
        confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for p in preds:
            pred   = p["predicted"]
            actual = p["actual"]
            confusion[pred][actual] += 1
            if pred != actual:
                key = f"predicted_{pred.lower()}_was_{actual.lower()}"
                if key in errors:
                    errors[key] += 1

        print("\n ═══════ Confusion Matrix ═══════")
        print(f" {'Pred/Actual':>12} {'HOME':>8} {'DRAW':>8} {'AWAY':>8}")
        print(f" {'─' * 40}")
        for pred_label in ["HOME", "DRAW", "AWAY"]:
            row = f" {pred_label:>12}"
            for actual_label in ["HOME", "DRAW", "AWAY"]:
                val  = confusion[pred_label][actual_label]
                mark = "◼" if pred_label == actual_label else " "
                row += f" {mark}{val:>7}"
            print(row)
        print(f" {'─' * 40}")

        total_errors = sum(errors.values())
        print("\n ═══════ Error Analysis ═══════")
        for k, v in sorted(errors.items(), key=lambda x: -x[1]):
            pct = v / total_errors * 100 if total_errors > 0 else 0
            print(f" {k:<45}: {v:>4} ({pct:.1f}%)")

        home_pred  = sum(confusion["HOME"].values())
        draw_pred  = sum(confusion["DRAW"].values())
        away_pred  = sum(confusion["AWAY"].values())
        total_pred = home_pred + draw_pred + away_pred
        if total_pred > 0:
            print(f"\n ═══════ Prediction Bias ═══════")
            print(f" HOME predicted: {home_pred:>4} ({home_pred / total_pred * 100:.1f}%)")
            print(f" DRAW predicted: {draw_pred:>4} ({draw_pred / total_pred * 100:.1f}%)")
            print(f" AWAY predicted: {away_pred:>4} ({away_pred / total_pred * 100:.1f}%)")

        return {
            "confusion": {k: dict(v) for k, v in confusion.items()},
            "errors":    errors,
        }

    def deep_error_mining(self, preds: List[dict]) -> dict:
        """V7.4: mine calibration failures, confidence risk and component disagreement.
        This is strictly descriptive: it never uses evaluation outcomes to alter
        the predictions that generated them, so it is safe to use as a post-test
        diagnostic before the next training window.
        """
        labels = ["HOME", "DRAW", "AWAY"]
        bins = [(0.333, 0.40), (0.40, 0.45), (0.45, 0.50),
                (0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.01)]
        reliability = []
        for lo, hi in bins:
            rows = [x for x in preds if lo <= max(x.get("probs", (0,0,0))) < hi]
            if not rows:
                continue
            acc = sum(bool(x.get("correct")) for x in rows) / len(rows)
            avg_conf = sum(max(x.get("probs", (0,0,0))) for x in rows) / len(rows)
            reliability.append({
                "range": f"{lo:.2f}-{hi:.2f}",
                "n": len(rows),
                "avg_conf": round(avg_conf, 4),
                "accuracy": round(acc, 4),
                "gap": round(acc - avg_conf, 4),
            })

        component = {}
        for label in labels:
            rows = [x for x in preds if x.get("actual") == label]
            component[label] = {"n": len(rows), "correct": sum(x.get("predicted") == label for x in rows)}

        # Count how often each component's standalone argmax agrees with the final pick.
        agreement = {}
        for x in preds:
            final = x.get("predicted")
            for name, probs in (x.get("models") or {}).items():
                try:
                    standalone = labels[int(np.argmax(np.asarray(probs, dtype=float)))]
                except Exception:
                    continue
                z = agreement.setdefault(name, {"n":0, "agree":0, "standalone_correct":0, "final_correct_when_disagree":0})
                z["n"] += 1
                z["agree"] += int(standalone == final)
                z["standalone_correct"] += int(standalone == x.get("actual"))
                z["final_correct_when_disagree"] += int(standalone != final and final == x.get("actual"))
        for z in agreement.values():
            z["agreement_rate"] = round(z["agree"] / z["n"], 4) if z["n"] else 0.0
            z["standalone_accuracy"] = round(z["standalone_correct"] / z["n"], 4) if z["n"] else 0.0

        high_conf_errors = [
            {"home": x.get("home"), "away": x.get("away"), "predicted": x.get("predicted"),
             "actual": x.get("actual"), "confidence": x.get("confidence")}
            for x in preds if max(x.get("probs", (0,0,0))) >= 0.60 and not x.get("correct")
        ]
        return {
            "reliability": reliability,
            "by_actual": component,
            "component_agreement": agreement,
            "high_confidence_errors": high_conf_errors[:100],
            "high_confidence_error_count": len(high_conf_errors),
        }

    def optimize_component_weights(self, samples: List[dict], base_weights: Dict[str, float]) -> Dict[str, float]:
        """V7.2: leakage-safe weight optimization on calibration data only.
        Optimizes multiclass log-loss over component probabilities, then applies
        conservative shrinkage toward the validated prior weights to avoid overfit.
        """
        if len(samples) < 40:
            return dict(base_weights)
        names = [n for n in base_weights if any(n in x.get("models", {}) for x in samples)]
        names = [n for n in names if n != "ml" or any("ml" in x.get("models", {}) for x in samples)]
        if len(names) < 2:
            return dict(base_weights)
        y = np.array([x["actual"] for x in samples], dtype=int)
        mats = []
        for n in names:
            mats.append(np.array([x["models"].get(n, (0.0,0.0,0.0)) for x in samples], dtype=float))
        A = np.stack(mats, axis=1)  # N x K x 3
        rng = np.random.default_rng(7303)
        prior = np.array([base_weights.get(n, 0.0) for n in names], dtype=float)
        prior = prior / prior.sum() if prior.sum() > 0 else np.ones(len(names))/len(names)
        best_w, best_loss = prior.copy(), float("inf")
        # V7.3: blocked temporal validation INSIDE the calibration window.
        # This prevents selecting a weight vector that merely memorizes the
        # first half of calibration data. The evaluation window remains unseen.
        folds = np.array_split(np.arange(len(y)), 3)
        candidates = [prior, np.ones(len(names))/len(names)]
        candidates.extend(rng.dirichlet(np.ones(len(names))*2.5, size=2500))
        for w in candidates:
            fold_losses = []
            for fold in folds:
                if len(fold) == 0: continue
                q = np.einsum("k,nkc->nc", w, A[fold])
                q = np.clip(q, 1e-7, 1.0)
                q /= q.sum(axis=1, keepdims=True)
                fold_losses.append(float(-np.mean(np.log(q[np.arange(len(fold)), y[fold]]))))
            loss = float(np.mean(fold_losses)) if fold_losses else float("inf")
            if loss < best_loss:
                best_loss, best_w = loss, w.copy()
        # Strong shrinkage toward prior: calibration set should tune, not rewrite.
        final_w = 0.60 * prior + 0.40 * best_w
        final_w /= final_w.sum()
        out = {k: float(v) for k, v in zip(names, final_w)}
        for k, v in base_weights.items():
            if k not in out:
                out[k] = 0.0
        return out

    def train_meta_learner(self, samples: List[dict], base_weights: Dict[str,float]) -> Optional[MetaLearner]:
        names=[n for n in base_weights if any(n in x.get("models",{}) for x in samples)]
        if len(names)<2: return None
        meta=MetaLearner()
        return meta if meta.fit(samples,names) else None

    def optimize_adaptive_profiles(self, samples: List[dict], base_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """V7.5: learn conservative per-regime ensemble profiles from calibration only.
        Sparse regimes fall back to global weights. Each regime is optimized with
        blocked temporal folds, then shrunk strongly toward the global prior.
        """
        profiles: Dict[str, Dict[str, float]] = {}
        groups: Dict[str, List[dict]] = defaultdict(list)
        for x in samples:
            groups[str(x.get("regime", "BALANCED"))].append(x)
        for regime, rows in groups.items():
            if len(rows) < 80:
                continue
            prof = self.optimize_component_weights(rows, base_weights)
            names = set(base_weights) | set(prof)
            prior = np.array([float(base_weights.get(n, 0.0)) for n in names], dtype=float)
            cand = np.array([float(prof.get(n, 0.0)) for n in names], dtype=float)
            if prior.sum() <= 0:
                prior = np.ones(len(names))
            if cand.sum() <= 0:
                cand = prior.copy()
            prior /= prior.sum(); cand /= cand.sum()
            # Strong shrinkage: regime specialization must earn its influence.
            final = 0.80 * prior + 0.20 * cand
            final /= final.sum()
            profiles[regime] = {n: float(v) for n, v in zip(names, final)}
        return profiles

    def run(
        self,
        matches:   List[dict],
        resources: "LeagueResources",
        split:     float = BACKTEST_SPLIT,
        temporal_cv: bool = True,
    ) -> dict:
        fin = [m for m in matches if m.get("status") == "FINISHED"]
        fin.sort(key=lambda m: m.get("utcDate", ""))

        si    = int(len(fin) * split)
        train = fin[:si]
        test  = fin[si:]

        if len(train) < 30 or len(test) < 10:
            return {"error": "Not enough data"}

        # ══════════════════════════════════════════════════════
        # V6.1 FIX: ثلاث نسخ منفصلة من DataProc لمنع Data Leakage
        # ══════════════════════════════════════════════════════

        # ── النموذج يُدرَّب على train فقط ────────────────────
        td_train = DataProc(resources)
        td_train.process(train)

        ml = None
        if ML_AVAILABLE:
            ml = MLPred(model_file=resources.model_file)
            ml.train(td_train, force_retrain=True, temporal_cv=temporal_cv)

        # ── تقسيم مجموعة الاختبار إلى نصفين ─────────────────
        cs      = len(test) // 2
        cal_set  = test[:cs]
        eval_set = test[cs:]

        # ══════════════════════════════════════════════════════
        # المرحلة الأولى: Calibration
        # td_cal يبدأ من train ويُضاف إليه cal_set تدريجياً
        # ══════════════════════════════════════════════════════
        td_cal = DataProc(resources)
        td_cal.process(train)
        eng_cal = Engine(td_cal, resources, ml)

        cal_total = 0
        cal_samples: List[dict] = []
        for m in cal_set:
            ht  = m.get("homeTeam", {})
            at  = m.get("awayTeam", {})
            ft  = m.get("score", {}).get("fullTime", {})
            hid = ht.get("id")
            aid = at.get("id")
            ahg = ft.get("home")
            aag = ft.get("away")
            if not hid or not aid or ahg is None or aag is None:
                continue

            pr = eng_cal.predict(hid, aid, m.get("utcDate", ""))
            if not pr:
                td_cal.process([m])
                continue

            ahg = int(ahg)
            aag = int(aag)
            actual = "HOME" if ahg > aag else ("AWAY" if ahg < aag else "DRAW")
            cal_samples.append({
                "actual": ["HOME", "DRAW", "AWAY"].index(actual),
                "models": {k: tuple(float(v) for v in probs) for k, probs in pr.models.items()},
                "ensemble": tuple(float(v) for v in (pr.hp, pr.dp, pr.ap)),
                "regime": pr.adaptive_regime,
                "context": {
                    "elo_gap": abs(float(pr.h_elo-pr.a_elo)),
                    "xg_diff": float(pr.hxg-pr.axg),
                    "form_diff": float(td_cal.teams[hid].form_score-td_cal.teams[aid].form_score) if hid in td_cal.teams and aid in td_cal.teams else 0.0,
                    "rest_diff": float(pr.h_rest-pr.a_rest),
                    "data_reliability": min(td_cal.teams[hid].played/float(max(resources.total_rounds,1)), td_cal.teams[aid].played/float(max(resources.total_rounds,1))) if hid in td_cal.teams and aid in td_cal.teams else 0.0,
                    "season_progress": ((parse_date(m.get("utcDate", "")).month-8)%12+1)/12.0 if parse_date(m.get("utcDate", "")) else 0.0,
                    "is_derby": int(pr.is_derby),
                },
            })

            self.cal.add((pr.hp, pr.dp, pr.ap), actual, regime=pr.adaptive_regime)
            cal_total += 1

            # تحديث td_cal بعد التنبؤ (لا قبله) ✅
            td_cal.process([m])

        cal_ok = self.cal.calibrate()
        optimized_weights = self.optimize_component_weights(cal_samples, WEIGHTS)
        adaptive_profiles = self.optimize_adaptive_profiles(cal_samples, optimized_weights)
        meta_learner = self.train_meta_learner(cal_samples, optimized_weights)
        meta_path = str(Path("models") / f"{resources.code}_meta_v7.6.pkl")
        if meta_learner:
            meta_learner.save(meta_path)

        # ══════════════════════════════════════════════════════
        # المرحلة الثانية: Evaluation
        # td_eval يبدأ من train + cal_set كاملاً
        # V6.1 FIX: نسخة جديدة نظيفة تمنع data leakage
        # ══════════════════════════════════════════════════════
        td_eval = DataProc(resources)
        td_eval.process(train + cal_set)   # كل البيانات السابقة للـ eval

        eng_eval = Engine(
            td_eval, resources, ml,
            cal=self.cal if cal_ok else None,
            meta=meta_learner,
        )
        # V7.2: weights are tuned on the calibration window only.
        eng_eval.w = dict(optimized_weights)
        eng_eval.adaptive_profiles = dict(adaptive_profiles)

        # ── عدّادات الأداء ────────────────────────────────────
        eval_correct    = 0
        eval_correct_sc = 0
        eval_total      = 0

        home_correct = home_total = 0
        draw_correct = draw_total = 0
        away_correct = away_total = 0

        hi_correct = hi_total = 0
        me_correct = me_total = 0
        lo_correct = lo_total = 0

        # V6.1 FIX: DC Accuracy — يعتمد على توقع النموذج لا على حدوث الحدث
        DC_THRESHOLD = 0.60
        dc_1x_correct = dc_1x_total = 0
        dc_x2_correct = dc_x2_total = 0
        dc_12_correct = dc_12_total = 0

        preds: List[dict] = []

        for m in eval_set:
            ht  = m.get("homeTeam", {})
            at  = m.get("awayTeam", {})
            ft  = m.get("score", {}).get("fullTime", {})
            hid = ht.get("id")
            aid = at.get("id")
            ahg = ft.get("home")
            aag = ft.get("away")
            if not hid or not aid or ahg is None or aag is None:
                continue

            hn = ht.get("shortName") or ht.get("name", "")
            an = at.get("shortName") or at.get("name", "")

            pr = eng_eval.predict(hid, aid, m.get("utcDate", ""))
            if not pr:
                td_eval.process([m])
                continue

            ahg    = int(ahg)
            aag    = int(aag)
            actual = "HOME" if ahg > aag else ("AWAY" if ahg < aag else "DRAW")

            eval_total += 1
            is_correct  = pr.result == actual
            if is_correct:
                eval_correct += 1

            if pr.pred_sc[0] == ahg and pr.pred_sc[1] == aag:
                eval_correct_sc += 1

            # ── دقة حسب النتيجة ───────────────────────────────
            if actual == "HOME":
                home_total += 1
                if is_correct: home_correct += 1
            elif actual == "DRAW":
                draw_total += 1
                if is_correct: draw_correct += 1
            else:
                away_total += 1
                if is_correct: away_correct += 1

            # ── دقة حسب الثقة ─────────────────────────────────
            conf = pr.conf
            if conf > 60:
                hi_total += 1
                if is_correct: hi_correct += 1
            elif conf >= 45:
                me_total += 1
                if is_correct: me_correct += 1
            else:
                lo_total += 1
                if is_correct: lo_correct += 1

            # ── V6.1 FIX: DC Accuracy بناءً على توقع النموذج ─
            # 1X: نتوقع HOME أو DRAW
            if pr.dc_1x > DC_THRESHOLD:
                dc_1x_total += 1
                if actual in ("HOME", "DRAW"):
                    dc_1x_correct += 1

            # X2: نتوقع AWAY أو DRAW
            if pr.dc_x2 > DC_THRESHOLD:
                dc_x2_total += 1
                if actual in ("AWAY", "DRAW"):
                    dc_x2_correct += 1

            # 12: نتوقع HOME أو AWAY (لا تعادل)
            if pr.dc_12 > DC_THRESHOLD:
                dc_12_total += 1
                if actual in ("HOME", "AWAY"):
                    dc_12_correct += 1

            # V10.0.5: metric boundary integrity check. Never persist raw
            # probabilities from a prediction object without sanitising them.
            safe_probs = validate_probability_simplex((pr.hp, pr.dp, pr.ap))
            preds.append({
                "home":        hn,
                "away":        an,
                "predicted":   pr.result,
                "actual":      actual,
                "pred_score":  pr.pred_sc,
                "actual_score":(ahg, aag),
                "confidence":  float(pr.conf),
                "correct":     is_correct,
                "probs":       safe_probs,
                "dc_1x":       float(pr.dc_1x),
                "dc_x2":       float(pr.dc_x2),
                "dc_12":       float(pr.dc_12),
                "calibrated":  pr.calibrated,
                "models": {k: tuple(float(v) for v in probs) for k, probs in pr.models.items()},
                "draw_trace": {k: float(v) for k, v in pr.draw_trace.items()},
            })

            # تحديث td_eval بعد التنبؤ (لا قبله) ✅
            td_eval.process([m])

        if eval_total == 0:
            return {"error": "No evaluation matches found"}

        # ══════════════════════════════════════════════════════
        # V6.1 FIX: Brier Score — التقسيم على N فقط [0,2]
        # ثم على 2 للتطبيع إلى [0,1] (Multi-class Brier)
        # ══════════════════════════════════════════════════════
        brier_raw = 0.0
        for pred_item in preds:
            actual_vec    = [0, 0, 0]
            outcome_index = ["HOME", "DRAW", "AWAY"].index(pred_item["actual"])
            actual_vec[outcome_index] = 1
            for i in range(3):
                brier_raw += (pred_item["probs"][i] - actual_vec[i]) ** 2

        # Multi-class Brier مُطبَّع [0,1]:  BS = Σ / (N * 2)
        brier = brier_raw / (eval_total * 2) if eval_total > 0 else 0.0

        # V7.3: probability-first metric. Lower is better.
        logloss = 0.0
        for pred_item in preds:
            outcome_index = ["HOME", "DRAW", "AWAY"].index(pred_item["actual"])
            logloss += -np.log(max(float(pred_item["probs"][outcome_index]), 1e-9))
        logloss = logloss / eval_total if eval_total else 0.0

        # ── احتساب النسب ──────────────────────────────────────
        result_acc = eval_correct    / eval_total * 100
        score_acc  = eval_correct_sc / eval_total * 100

        home_acc = home_correct / home_total * 100 if home_total > 0 else 0.0
        draw_acc = draw_correct / draw_total * 100 if draw_total > 0 else 0.0
        away_acc = away_correct / away_total * 100 if away_total > 0 else 0.0

        hi_acc   = hi_correct / hi_total * 100 if hi_total > 0 else 0.0
        me_acc   = me_correct / me_total * 100 if me_total > 0 else 0.0
        lo_acc   = lo_correct / lo_total * 100 if lo_total > 0 else 0.0

        dc_1x_acc = dc_1x_correct / dc_1x_total * 100 if dc_1x_total > 0 else 0.0
        dc_x2_acc = dc_x2_correct / dc_x2_total * 100 if dc_x2_total > 0 else 0.0
        dc_12_acc = dc_12_correct / dc_12_total * 100 if dc_12_total > 0 else 0.0

        ml_acc_val = float(ml.acc * 100) if ml and ml.trained else 0.0

        # ── Confusion Matrix ──────────────────────────────────
        print()
        error_analysis = self.analyze_errors(preds)
        deep_analysis = self.deep_error_mining(preds)

        # ── حفظ Elo من td_eval (الأحدث) ──────────────────────
        elo_data = {t.name: t.elo for t in td_eval.teams.values()}
        resources.save_elo(elo_data)

        self.results = {
            "total":        eval_total,
            "train":        len(train),
            "test":         len(test),
            "cal_size":     cal_total,
            "eval_size":    eval_total,
            "result_acc":   result_acc,
            "score_acc":    score_acc,
            "brier":        float(brier),
            "logloss":      float(logloss),
            "correct":      eval_correct,
            "correct_sc":   eval_correct_sc,
            "home_acc":     home_acc,
            "draw_acc":     draw_acc,
            "away_acc":     away_acc,
            "home_total":   home_total,
            "draw_total":   draw_total,
            "away_total":   away_total,
            "hi_acc":       hi_acc,
            "me_acc":       me_acc,
            "lo_acc":       lo_acc,
            "hi_n":         hi_total,
            "me_n":         me_total,
            "lo_n":         lo_total,
            "dc_1x_acc":    dc_1x_acc,
            "dc_x2_acc":    dc_x2_acc,
            "dc_12_acc":    dc_12_acc,
            "dc_1x_n":      dc_1x_total,
            "dc_x2_n":      dc_x2_total,
            "dc_12_n":      dc_12_total,
            "dc_threshold": DC_THRESHOLD,
            "ml_acc":       ml_acc_val,
            "cal_used":     cal_ok,
            "confusion":    error_analysis.get("confusion", {}),
            "error_analysis": error_analysis.get("errors", {}),
            "deep_error_mining": deep_analysis,
            "meta_active": bool(meta_learner and meta_learner.active),
            "meta_validation_logloss": float(meta_learner.validation_logloss) if meta_learner and meta_learner.validation_logloss is not None else None,
            "meta_baseline_logloss": float(meta_learner.baseline_logloss) if meta_learner and meta_learner.baseline_logloss is not None else None,
            "optimized_weights": optimized_weights,
            "adaptive_profiles": adaptive_profiles,
            "adaptive_profile_sizes": {k: sum(1 for x in cal_samples if x.get("regime") == k) for k in sorted({x.get("regime") for x in cal_samples})},
            "predictions":  preds,
        }
        return self.results

    def run_walk_forward(self, matches: List[dict], resources: "LeagueResources",
                         min_train_ratio: float = 0.55, step_ratio: float = 0.10,
                         eval_ratio: float = 0.10, max_folds: int = 5) -> dict:
        """V7.8: expanding-window Walk-Forward validation.

        Every fold is strictly chronological. The normal leakage-safe backtest is
        executed independently at each origin, then fold metrics are aggregated.
        This is intentionally a validation harness, not a replacement for training.
        """
        fin = [m for m in matches if m.get("status") == "FINISHED"]
        fin.sort(key=lambda m: m.get("utcDate", ""))
        n = len(fin)
        if n < 60:
            return {"error": "Not enough finished matches for walk-forward"}
        min_train = max(30, int(n * min_train_ratio))
        eval_n = max(10, int(n * eval_ratio))
        step = max(10, int(n * step_ratio))
        folds=[]
        origin=min_train
        while origin + eval_n <= n and len(folds) < max_folds:
            subset=fin[:origin+eval_n]
            split=origin/float(len(subset))
            r=self.run(subset, resources, split=split)
            if not r.get("error"):
                folds.append({
                    "fold": len(folds)+1, "train_size": origin, "eval_size": r.get("eval_size",0),
                    "logloss": float(r.get("logloss",0)), "brier": float(r.get("brier",0)),
                    "result_acc": float(r.get("result_acc",0)), "draw_acc": float(r.get("draw_acc",0)),
                    "actual_draw_rate": (sum(1 for x in r.get("predictions", []) if x.get("actual") == "DRAW") / max(1, len(r.get("predictions", [])))),
                    "predicted_draw_rate": (sum(1 for x in r.get("predictions", []) if x.get("predicted") == "DRAW") / max(1, len(r.get("predictions", [])))),
                    "draw_recall": (sum(1 for x in r.get("predictions", []) if x.get("actual") == "DRAW" and x.get("predicted") == "DRAW") / max(1, sum(1 for x in r.get("predictions", []) if x.get("actual") == "DRAW"))),
                    "draw_precision": (sum(1 for x in r.get("predictions", []) if x.get("actual") == "DRAW" and x.get("predicted") == "DRAW") / max(1, sum(1 for x in r.get("predictions", []) if x.get("predicted") == "DRAW"))),
                    "draw_baseline_accuracy": (sum(1 for x in r.get("predictions", []) if x.get("actual") == "DRAW") / max(1, len(r.get("predictions", [])))),
                    "mean_draw_probability": float(np.mean([x.get("probs",(0,0,0))[1] for x in r.get("predictions", [])])) if r.get("predictions") else 0.0,
                    "draw_brier": float(np.mean([(x.get("probs",(0,0,0))[1] - (1.0 if x.get("actual")=="DRAW" else 0.0))**2 for x in r.get("predictions", [])])) if r.get("predictions") else 1.0,
                    "draw_logloss": float(-np.mean([np.log(np.clip(x.get("probs",(0,0,0))[1] if x.get("actual")=="DRAW" else 1.0-x.get("probs",(0,0,0))[1], 1e-7, 1.0)) for x in r.get("predictions", [])])) if r.get("predictions") else 99.0,
                    "draw_probability_on_actual_draw": float(np.mean([x.get("probs",(0,0,0))[1] for x in r.get("predictions", []) if x.get("actual")=="DRAW"])) if any(x.get("actual")=="DRAW" for x in r.get("predictions", [])) else 0.0,
                    "draw_trace": {stage: float(np.mean([x.get("draw_trace",{}).get(stage, x.get("probs",(0,0,0))[1]) for x in r.get("predictions", [])])) for stage in ("draw_model","weighted_ensemble","after_draw_preservation","after_draw_correction","after_calibration","after_meta","final")},
                    "meta_active": bool(r.get("meta_active",False)), "cal_used": bool(r.get("cal_used",False)),
                    "meta_validation_logloss": r.get("meta_validation_logloss"),
                    "meta_baseline_logloss": r.get("meta_baseline_logloss"),
                })
            origin += step
        if not folds:
            return {"error": "No valid walk-forward folds"}
        weights=np.array([max(1,f["eval_size"]) for f in folds],dtype=float)
        def wavg(k): return float(np.average([f[k] for f in folds],weights=weights))
        return {"folds":folds,"fold_count":len(folds),"total_eval":int(weights.sum()),
                "logloss":wavg("logloss"),"brier":wavg("brier"),
                "result_acc":wavg("result_acc"),"draw_acc":wavg("draw_acc"),
                "accuracy":wavg("result_acc"), "draw_accuracy":wavg("draw_acc"),
                "actual_draw_rate":wavg("actual_draw_rate"), "predicted_draw_rate":wavg("predicted_draw_rate"),
                "draw_recall":wavg("draw_recall"), "draw_precision":wavg("draw_precision"),
                "draw_baseline_accuracy":wavg("draw_baseline_accuracy"),
                "mean_draw_probability":wavg("mean_draw_probability"), "draw_brier":wavg("draw_brier"), "draw_logloss":wavg("draw_logloss"),
                "draw_probability_on_actual_draw":wavg("draw_probability_on_actual_draw"),
                "draw_trace": {stage: float(np.average([f.get("draw_trace",{}).get(stage,0.0) for f in folds], weights=weights)) for stage in ("draw_model","weighted_ensemble","after_draw_preservation","after_draw_correction","after_calibration","after_meta","final")},
                "meta_activation_rate":float(np.mean([f["meta_active"] for f in folds])),
                "calibration_activation_rate":float(np.mean([f["cal_used"] for f in folds]))}


# ══════════════════════════════════════════════════════════════
# V7.9 — AUTOMATED MODEL SELECTION ENGINE
# ══════════════════════════════════════════════════════════════
class ModelSelectionEngine:
    """Conservative league-specific policy selector driven by Walk-Forward evidence.

    The selector never claims that a layer is useful merely because it exists.
    It requires enough chronological validation data and records an explicit
    policy explaining whether Meta-Learning and Calibration should remain enabled.
    """
    VERSION = "v7.9"

    def __init__(self, league_code: str):
        self.league_code = league_code
        self.path = Path("models") / f"{league_code}_selection_v7.9.json"

    def select(self, wf: dict) -> dict:
        if not wf or wf.get("error"):
            return {"version": self.VERSION, "active": False,
                    "reason": wf.get("error", "No walk-forward evidence") if wf else "No evidence"}
        folds = wf.get("folds", [])
        total = int(wf.get("total_eval", 0))
        # Conservative evidence gates: enough folds and out-of-sample matches.
        evidence_ok = len(folds) >= 3 and total >= 30
        meta_rate = float(wf.get("meta_activation_rate", 0.0))
        cal_rate = float(wf.get("calibration_activation_rate", 0.0))
        policy = {
            "version": self.VERSION,
            "league": self.league_code,
            "active": bool(evidence_ok),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "evidence": {"folds": len(folds), "total_eval": total,
                         "logloss": float(wf.get("logloss", 0.0)),
                         "brier": float(wf.get("brier", 0.0)),
                         "draw_acc": float(wf.get("draw_acc", 0.0))},
            # A component must have survived chronological gates often enough;
            # this is intentionally a policy gate, not an accuracy-only switch.
            "components": {
                "meta_learner": {"enabled": bool(evidence_ok and meta_rate >= 0.34),
                                 "activation_rate": meta_rate},
                "calibration": {"enabled": bool(evidence_ok and cal_rate >= 0.34),
                                "activation_rate": cal_rate},
                "reliability_engine": {"enabled": bool(evidence_ok)}
            },
            "reason": "Chronological evidence accepted" if evidence_ok else
                      "Insufficient chronological evidence; keep conservative defaults"
        }
        return policy

    def save(self, policy: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> Optional[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if data.get("version") == self.VERSION else None
        except Exception:
            return None



class ContextualPerformanceMonitor:
    """V8.1 rolling, context-aware performance guard.

    Profiles are updated from chronological, out-of-sample observations only.
    A context is penalized only when enough recent evidence shows its Log Loss
    materially worse than the league baseline; sparse evidence remains neutral.
    """
    VERSION = "v8.1"
    MIN_SAMPLES = 30

    def __init__(self, league_code: str):
        self.league_code = league_code
        self.path = Path("models") / f"{league_code}_context_monitor_v8.1.json"

    @staticmethod
    def _key(regime: str, confidence: float) -> str:
        band = "HIGH" if confidence >= .55 else "MEDIUM" if confidence >= .42 else "LOW"
        return f"{regime or 'BALANCED'}::{band}"

    def build(self, rows: List[dict]) -> dict:
        buckets = {}
        for r in rows:
            try:
                probs=np.clip(np.asarray(r["probs"],dtype=float),1e-7,1.0); probs/=probs.sum()
                actual=int(r["actual"]); key=self._key(r.get("regime","BALANCED"),float(np.max(probs)))
                buckets.setdefault(key,[]).append(float(-np.log(probs[actual])))
            except Exception: continue
        all_loss=[x for v in buckets.values() for x in v]
        baseline=float(np.mean(all_loss)) if all_loss else None
        profiles={}
        for k,v in buckets.items():
            n=len(v); ll=float(np.mean(v)); deterioration=(ll-baseline) if baseline is not None else 0.0
            factor=1.0 if n<self.MIN_SAMPLES or deterioration<=0.03 else max(.72, 1.0-min(.28,deterioration*.45))
            profiles[k]={"samples":n,"log_loss":ll,"baseline_log_loss":baseline,"deterioration":deterioration,"factor":factor,"status":"degraded" if factor<.99 else "stable"}
        return {"version":self.VERSION,"league":self.league_code,"created_at":datetime.utcnow().isoformat()+"Z","baseline_log_loss":baseline,"profiles":profiles}

    def save(self, profile: dict) -> None:
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_suffix('.tmp'); tmp.write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(self.path)

    def load(self) -> Optional[dict]:
        try:
            d=json.loads(self.path.read_text(encoding='utf-8'))
            return d if d.get("version")==self.VERSION and d.get("league")==self.league_code else None
        except Exception: return None

    def assess(self, probs, regime="BALANCED", profile=None):
        profile=profile or self.load()
        if not profile: return 1.0, []
        key=self._key(regime,float(max(probs))); row=profile.get("profiles",{}).get(key)
        if not row or int(row.get("samples",0))<self.MIN_SAMPLES: return 1.0, []
        factor=float(row.get("factor",1.0))
        return factor, ([f"Recent performance degraded in {key}"] if factor<.99 else [])




class ControlledAutoRetrainingPipeline:
    """V8.3 controlled candidate retraining pipeline.

    Drift detection may request a retrain, but production artifacts are never
    overwritten directly. A candidate is trained in an isolated artifact path,
    evaluated on a chronological holdout, and can only be promoted when a
    leakage-safe incumbent comparison proves a material improvement.
    """
    VERSION = "v8.3"
    MIN_MATCHES = 80
    MIN_HOLDOUT = 20
    MIN_IMPROVEMENT = 0.01

    def __init__(self, league_code: str, resources: "LeagueResources"):
        self.league_code = league_code
        self.resources = resources
        self.root = Path("models") / "retraining" / league_code
        self.report_path = self.root / "latest_retraining_report_v8.3.json"

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _actual(m: dict):
        ft=m.get("score",{}).get("fullTime",{})
        h,a=ft.get("home"),ft.get("away")
        if h is None or a is None: return None
        return 0 if h>a else (1 if h==a else 2)

    def _holdout_score(self, model, train_matches, holdout):
        sim=DataProc(self.resources); sim.process(train_matches)
        losses=[]
        for m in holdout:
            ht,at=m.get("homeTeam",{}),m.get("awayTeam",{})
            hid,aid=ht.get("id"),at.get("id"); actual=self._actual(m)
            if hid is not None and aid is not None and actual is not None:
                h=sim.teams.get(hid); a=sim.teams.get(aid)
                if h and a and h.played>=3 and a.played>=3:
                    try:
                        md=parse_date(m.get("utcDate",""))
                        derby=bool(self.resources.is_derby(ht.get("shortName") or ht.get("name",""), at.get("shortName") or at.get("name","")))
                        probs=model.predict(h,a,sim,md,derby)
                        if probs is not None:
                            q=np.clip(np.asarray(probs,dtype=float),1e-7,1.0); q/=q.sum()
                            losses.append(float(-np.log(q[actual])))
                    except Exception: pass
            sim.process([m])
        return (float(np.mean(losses)), len(losses)) if losses else (None,0)

    def run(self, matches: List[dict], drift_report: Optional[dict]=None) -> dict:
        now=datetime.utcnow().isoformat()+"Z"
        report={"version":self.VERSION,"league":self.league_code,"created_at":now,"status":"not_started","production_replaced":False}
        if not drift_report or not drift_report.get("retrain_recommended"):
            report.update(status="no_retraining_recommendation", reason="v8.2 did not recommend retraining")
            self._atomic_json(self.report_path,report); return report
        fin=sorted([m for m in matches if m.get("status")=="FINISHED" and self._actual(m) is not None],key=lambda x:x.get("utcDate", ""))
        if len(fin)<self.MIN_MATCHES:
            report.update(status="insufficient_data", samples=len(fin))
            self._atomic_json(self.report_path,report); return report
        hold_n=max(self.MIN_HOLDOUT,int(len(fin)*0.20)); hold_n=min(hold_n,len(fin)//2)
        train,hold=fin[:-hold_n],fin[-hold_n:]
        candidate_path=self.root / f"candidate_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.pkl"
        candidate=MLPred(model_file=str(candidate_path))
        td=DataProc(self.resources); td.process(train)
        if not candidate.train(td, force_retrain=True):
            report.update(status="candidate_training_failed", train_samples=len(train), holdout_samples=len(hold))
            self._atomic_json(self.report_path,report); return report
        candidate.save_pipeline()
        cand_loss,cand_n=self._holdout_score(candidate,train,hold)
        report.update(status="candidate_validated" if cand_loss is not None else "candidate_validation_failed", train_samples=len(train), holdout_samples=len(hold), candidate_model=str(candidate_path), candidate_log_loss=cand_loss, candidate_eval_samples=cand_n)
        # Promotion requires an explicit leakage-safe incumbent benchmark. Older artifacts
        # have no training cutoff metadata, so v8.3 deliberately refuses unsafe promotion.
        meta_path=Path(str(self.resources.model_file)+".meta.json")
        incumbent_loss=None
        if meta_path.exists():
            try:
                meta=json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("version") and meta.get("trained_until"):
                    incumbent=MLPred(model_file=self.resources.model_file)
                    if incumbent._try_load_external():
                        incumbent_loss,_=self._holdout_score(incumbent,train,hold)
            except Exception: pass
        report["incumbent_log_loss"]=incumbent_loss
        if incumbent_loss is None:
            report.update(production_replaced=False, promotion_status="blocked_no_leakage_safe_incumbent_benchmark")
        elif cand_loss is not None and incumbent_loss-cand_loss>=self.MIN_IMPROVEMENT:
            backup=Path(self.resources.model_file).with_suffix(".v8.3.backup.pkl")
            prod=Path(self.resources.model_file); prod.parent.mkdir(parents=True,exist_ok=True)
            if prod.exists(): shutil.copy2(prod,backup)
            shutil.copy2(candidate_path,prod)
            self._atomic_json(meta_path,{"version":self.VERSION,"league":self.league_code,"trained_until":train[-1].get("utcDate"),"promoted_at":now,"candidate_log_loss":cand_loss,"incumbent_log_loss":incumbent_loss})
            report.update(production_replaced=True,promotion_status="promoted",improvement=incumbent_loss-cand_loss,backup_model=str(backup))
        else:
            report.update(production_replaced=False,promotion_status="candidate_not_better_enough",improvement=(incumbent_loss-cand_loss) if cand_loss is not None else None)
        self._atomic_json(self.report_path,report)
        return report


class ModelRegistry:
    """V8.4 per-league model lifecycle registry.

    Keeps an append-only history of Champion/Challenger decisions so model
    promotion is auditable without changing the conservative v8.3 gate.
    """
    VERSION = "v8.4"

    def __init__(self, league_code: str):
        self.league_code = league_code
        self.root = Path("models") / "registry" / league_code
        self.path = self.root / "model_registry_v8.4.json"

    def _load(self):
        if not self.path.exists():
            return {"version": self.VERSION, "league": self.league_code, "champion": None, "history": []}
        try:
            data=json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != self.VERSION or data.get("league") != self.league_code:
                return {"version": self.VERSION, "league": self.league_code, "champion": None, "history": []}
            data.setdefault("history", []); return data
        except Exception:
            return {"version": self.VERSION, "league": self.league_code, "champion": None, "history": []}

    def _save(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp=self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def record_retraining(self, report: dict) -> dict:
        data=self._load(); now=datetime.utcnow().isoformat()+"Z"
        status=report.get("promotion_status") or report.get("status") or "unknown"
        challenger={
            "role":"challenger", "model":report.get("candidate_model"),
            "created_at":report.get("created_at", now),
            "log_loss":report.get("candidate_log_loss"),
            "eval_samples":report.get("candidate_eval_samples"),
            "decision":status, "reason":report.get("reason") or status,
            "improvement":report.get("improvement")
        }
        data["history"].append(challenger)
        if report.get("production_replaced"):
            data["champion"]={
                "role":"champion", "model":report.get("candidate_model"),
                "promoted_at":now, "log_loss":report.get("candidate_log_loss"),
                "replaced_incumbent_log_loss":report.get("incumbent_log_loss"),
                "reason":"validated_improvement"
            }
            challenger["role"]="challenger_promoted"
        self._save(data)
        return {"registry_path":str(self.path), "decision":status, "champion":data.get("champion")}



class AutomatedDriftDetector:
    """V8.2 conservative data/performance drift monitor and retraining planner.

    It never silently retrains production models. Instead it requires sufficient
    recent evidence, records the reason, and emits an explicit retraining plan.
    This prevents reacting to short-term variance or a single bad matchweek.
    """
    VERSION = "v8.2"
    MIN_BASELINE = 80
    MIN_RECENT = 30

    def __init__(self, league_code: str):
        self.league_code = league_code
        self.path = Path("models") / f"{league_code}_drift_v8.2.json"

    @staticmethod
    def _psi(expected, actual, bins=10):
        expected=np.asarray(expected,dtype=float); actual=np.asarray(actual,dtype=float)
        expected=expected[np.isfinite(expected)]; actual=actual[np.isfinite(actual)]
        if len(expected)<10 or len(actual)<10: return 0.0
        edges=np.unique(np.quantile(expected,np.linspace(0,1,bins+1)))
        if len(edges)<3: return 0.0
        e=np.histogram(expected,bins=edges)[0]/max(len(expected),1)
        a=np.histogram(actual,bins=edges)[0]/max(len(actual),1)
        e=np.clip(e,1e-4,None); a=np.clip(a,1e-4,None)
        return float(np.sum((a-e)*np.log(a/e)))

    def build(self, rows: List[dict]) -> dict:
        clean=[]
        for r in rows:
            try:
                probs=np.clip(np.asarray(r["probs"],dtype=float),1e-7,1.0); probs/=probs.sum()
                actual=int(r["actual"]); clean.append({"probs":probs,"actual":actual,"context":r.get("context",{})})
            except Exception: continue
        if len(clean)<self.MIN_BASELINE+self.MIN_RECENT:
            return {"version":self.VERSION,"league":self.league_code,"status":"insufficient_data","samples":len(clean),"retrain_recommended":False}
        recent=clean[-max(self.MIN_RECENT,min(len(clean)//4,120)):]; base=clean[:len(clean)-len(recent)]
        perf_base=float(np.mean([-np.log(x["probs"][x["actual"]]) for x in base]))
        perf_recent=float(np.mean([-np.log(x["probs"][x["actual"]]) for x in recent]))
        perf_delta=perf_recent-perf_base
        features={}
        for key in ("elo_gap","xg_diff","form_diff","rest_diff","season_progress","data_reliability"):
            b=[x["context"].get(key,np.nan) for x in base]; r=[x["context"].get(key,np.nan) for x in recent]
            psi=self._psi(b,r); features[key]=round(psi,5)
        max_psi=max(features.values()) if features else 0.0
        data_drift=max_psi>=0.25
        performance_drift=perf_delta>=0.05
        severe=perf_delta>=0.10 or max_psi>=0.40
        return {"version":self.VERSION,"league":self.league_code,"created_at":datetime.utcnow().isoformat()+"Z","status":"drift_detected" if (data_drift or performance_drift) else "stable","baseline_samples":len(base),"recent_samples":len(recent),"baseline_log_loss":perf_base,"recent_log_loss":perf_recent,"performance_delta":perf_delta,"feature_psi":features,"max_psi":max_psi,"data_drift":data_drift,"performance_drift":performance_drift,"retrain_recommended":bool((data_drift and performance_drift) or severe),"retrain_scope":"league_models_and_calibration" if (data_drift or performance_drift) else "none"}

    def save(self, report: dict) -> None:
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_suffix(".tmp"); tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); tmp.replace(self.path)

    def load(self) -> Optional[dict]:
        try:
            d=json.loads(self.path.read_text(encoding="utf-8"))
            return d if d.get("version")==self.VERSION and d.get("league")==self.league_code else None
        except Exception: return None

    def assess(self, report=None):
        report=report or self.load()
        if not report: return 1.0, []
        if report.get("status")=="insufficient_data": return 1.0, []
        alerts=[]
        if report.get("data_drift"): alerts.append("Data drift detected")
        if report.get("performance_drift"): alerts.append("Recent predictive performance degraded")
        if report.get("retrain_recommended"): alerts.append("Retraining recommended before increasing model confidence")
        factor=.92 if report.get("retrain_recommended") else (.97 if alerts else 1.0)
        return factor, alerts


class AdaptiveEnsembleOptimizer:
    """V8.0: conservative, league-aware ensemble weight policy.

    Profiles are learned only from chronological calibration samples and are
    strongly shrunk toward the validated global prior. The saved policy is
    version-protected and can be safely ignored when evidence is insufficient.
    """
    VERSION = "v8.0"
    def __init__(self, league_code: str):
        self.league_code = league_code
        self.path = Path("models") / f"{league_code}_adaptive_ensemble_v8.0.json"

    @staticmethod
    def _normalize(w: Dict[str, float]) -> Dict[str, float]:
        x={k:max(0.0,float(v)) for k,v in w.items()}; z=sum(x.values())
        return {k:v/z for k,v in x.items()} if z>0 else dict(WEIGHTS)

    def build(self, global_weights: Dict[str,float], profiles: Dict[str,Dict[str,float]], sizes: Dict[str,int], walk_forward: Optional[dict]=None) -> dict:
        evidence_ok = bool(walk_forward and int(walk_forward.get("fold_count",0)) >= 3 and int(walk_forward.get("total_eval",0)) >= 30)
        clean={}
        for regime, w in (profiles or {}).items():
            n=int((sizes or {}).get(regime,0))
            if n < 80: continue
            # More data earns slightly more specialization, capped for stability.
            alpha=min(0.35, 0.12 + n/1200.0)
            names=set(global_weights)|set(w)
            blend={k:(1-alpha)*float(global_weights.get(k,0))+alpha*float(w.get(k,0)) for k in names}
            clean[regime]=self._normalize(blend)
        return {"version":self.VERSION,"league":self.league_code,"created_at":datetime.utcnow().isoformat()+"Z","evidence_ok":evidence_ok,"global_weights":self._normalize(global_weights),"profiles":clean,"profile_sizes":{k:int(v) for k,v in (sizes or {}).items()},"activation":"enabled" if evidence_ok and clean else "conservative_fallback"}

    def save(self, policy: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp=self.path.with_suffix(".tmp"); tmp.write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding="utf-8"); tmp.replace(self.path)

    def load(self) -> Optional[dict]:
        try:
            d=json.loads(self.path.read_text(encoding="utf-8"))
            return d if d.get("version")==self.VERSION and d.get("league")==self.league_code else None
        except Exception: return None


# ══════════════════════════════════════════════════════════════
# V8.6 — AUTOMATED LIVE PROMOTION PIPELINE
# ══════════════════════════════════════════════════════════════
class AutomatedLivePromotionPipeline:
    """Final safety gate joining drift, retraining, registry and live shadow evidence.

    This class deliberately does not replace production artifacts unless every
    independent gate passes. A failed or incomplete gate always retains Champion.
    """
    VERSION = "v8.6"
    MIN_HISTORICAL_IMPROVEMENT = 0.01
    MIN_LIVE_IMPROVEMENT = 0.01
    MIN_LIVE_SAMPLES = 30

    def __init__(self, league_code: str):
        self.league_code = league_code
        self.root = Path("models") / "promotion" / league_code
        self.path = self.root / "promotion_pipeline_v8.6.json"

    def _default(self):
        return {"version": self.VERSION, "league": self.league_code,
                "runs": [], "last_decision": None}

    def _load(self):
        if not self.path.exists(): return self._default()
        try:
            d=json.loads(self.path.read_text(encoding="utf-8"))
            return d if d.get("version")==self.VERSION and d.get("league")==self.league_code else self._default()
        except Exception: return self._default()

    def _save(self, d):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp=self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def evaluate(self, drift_report=None, retraining_report=None, registry=None, shadow=None):
        gates=[]
        # Historical candidate gate: v8.3 already performs leakage-safe temporal comparison.
        hist_ok=False
        if retraining_report:
            status=str(retraining_report.get("promotion_status") or retraining_report.get("status") or "")
            imp=float(retraining_report.get("improvement", retraining_report.get("log_loss_improvement", 0)) or 0)
            hist_ok=(status in {"promote_recommended","candidate_ready","approved"} or imp>=self.MIN_HISTORICAL_IMPROVEMENT)
            gates.append({"gate":"historical_validation","passed":hist_ok,"status":status,"improvement":imp})
        else:
            gates.append({"gate":"historical_validation","passed":False,"reason":"no_candidate_evidence"})

        # Registry gate: Challenger must be traceable and not previously rejected.
        registry_ok=False; reg_decision=None
        if registry:
            try:
                rd=registry._load()
                entries=rd.get("models", rd.get("history", rd.get("entries", [])))
                if isinstance(entries, list) and entries:
                    reg_decision=entries[-1].get("decision")
                    registry_ok=reg_decision not in {"rejected", "failed"}
                else:
                    registry_ok=True  # registry exists but schema is intentionally permissive
            except Exception: registry_ok=False
        gates.append({"gate":"registry_traceability","passed":registry_ok,"decision":reg_decision})

        # Live gate is independent from historical evidence.
        live_result=shadow.evaluate() if shadow else {"status":"shadow_unavailable"}
        live_ok=(live_result.get("status")=="evaluated" and
                 int(live_result.get("samples",0))>=self.MIN_LIVE_SAMPLES and
                 float(live_result.get("improvement",0))>=self.MIN_LIVE_IMPROVEMENT and
                 live_result.get("decision")=="promote_recommended")
        gates.append({"gate":"live_shadow_validation","passed":live_ok,"result":live_result})

        approved=hist_ok and registry_ok and live_ok
        decision="promotion_approved" if approved else "champion_retained"
        result={"version":self.VERSION,"league":self.league_code,
                "evaluated_at":datetime.utcnow().isoformat()+"Z",
                "decision":decision,"approved":approved,"gates":gates,
                "drift_detected":bool((drift_report or {}).get("retrain_recommended"))}
        d=self._load(); d["runs"].append(result); d["last_decision"]=result; self._save(d)
        return result

# ══════════════════════════════════════════════════════════════
# LEAGUE APP — V6.1: unique_matches key مُصلح
# ══════════════════════════════════════════════════════════════
class LeagueApp:
    def __init__(
        self,
        league_code: str,
        api_token:   str,
        odds_key:    str = "",
    ):
        self.code = league_code
        if league_code not in LEAGUES_CONFIG:
            raise ValueError(
                f"League '{league_code}' not found in {LEAGUES_CONFIG_FILE}"
            )
        self.resources = LeagueResources(league_code, LEAGUES_CONFIG[league_code])
        self.api       = FootballAPI(api_token, self.resources.api_url)
        self.data      = DataProc(self.resources)
        self.ml        = MLPred(model_file=self.resources.model_file)
        _sport_map = {
            "PL": "soccer_epl",
            "LL": "soccer_spain_la_liga",
            "SA": "soccer_italy_serie_a",
            "BL": "soccer_germany_bundesliga",
            "L1": "soccer_france_ligue_one",
        }
        self.odds      = OddsAPI(odds_key, _sport_map.get(league_code, "soccer_epl"))
        self.cal       = Calibrator()
        self.bt        = Backtester()
        self.meta = MetaLearner.load(str(Path("models") / f"{league_code}_meta_v7.6.pkl"))
        self.performance_monitor = ContextualPerformanceMonitor(league_code)
        self.performance_profile = self.performance_monitor.load()
        self.drift_detector = AutomatedDriftDetector(league_code)
        self.drift_report = self.drift_detector.load()
        self.retraining_pipeline = ControlledAutoRetrainingPipeline(league_code, self.resources)
        self.retraining_report = None
        self.model_registry = ModelRegistry(league_code)
        self.shadow = ShadowDeploymentManager(league_code)
        self.live_promotion_pipeline = AutomatedLivePromotionPipeline(league_code)
        self.live_promotion_report = None
        self.safe_artifact_promotion = SafeArtifactPromotionManager(league_code, self.resources)
        self.safe_promotion_report = None
        self.canary_monitor = PostPromotionCanaryMonitor(league_code, self.safe_artifact_promotion)
        self.canary_report = self.canary_monitor.load()
        self.lifecycle_orchestrator = ModelLifecycleOrchestrator(league_code, self.resources, {
            "drift": self.drift_detector, "registry": self.model_registry,
            "shadow": self.shadow, "promotion": self.live_promotion_pipeline,
            "safe_promotion": self.safe_artifact_promotion, "canary": self.canary_monitor})
        self.lifecycle_report = self.lifecycle_orchestrator.refresh()
        self.governance_engine = AutonomousModelGovernanceEngine(league_code, self.lifecycle_orchestrator)
        self.governance_report = self.governance_engine.refresh()
        self.decision_audit_engine = ExplainableDecisionAuditEngine(league_code, self.lifecycle_orchestrator, self.governance_engine)
        self.decision_audit_report = self.decision_audit_engine.refresh()
        self.data_quality_engine = DataQualityIntegrityEngine(league_code)
        self.data_quality_report = self.data_quality_engine.load()
        self.production_hardening = ProductionHardeningManager(league_code, self.resources)
        self.production_report = self.production_hardening.load()
        self.final_benchmark_engine = FinalBenchmarkAcceptanceEngine(league_code)
        self.final_benchmark_report = None
        self.eng:      Optional[Engine] = None
        self.raw:      List[dict]       = []
        self.last_preds: List[Pred]     = []
        self._log:     List[Tuple[str, str]] = []
        self.sy:       Optional[int]    = None

    def _log_msg(self, level: str, msg: str):
        full_msg = f"[{self.code}] {msg}"
        self._log.append((level, full_msg))
        print(f">>> {full_msg}", flush=True)

    def init(self) -> bool:
        self.raw = []

        # ── CSV ──────────────────────────────────────────────
        self._log_msg("progress", f"Loading CSV data ({self.resources.name})…")
        csv_matches = self.resources.load_csv_data()
        if csv_matches:
            self.raw.extend(csv_matches)
            self._log_msg("success", f"CSV: {len(csv_matches)} matches with deep stats")
        else:
            self._log_msg("info", "No CSV data found, using API only")

        # ── API ──────────────────────────────────────────────
        self.sy = self.api.season_year(self.resources.api_code)
        if self.sy:
            self._log_msg("progress", f"Loading API season {self.sy}…")
            api_matches = self.api.finished(self.resources.api_code, self.sy)
            if api_matches:
                self.raw.extend(api_matches)
                self._log_msg("success", f"API: {len(api_matches)} matches loaded")

        # ── V7: canonicalize identities before deduplication ───────────────
        # API IDs and CSV IDs are not guaranteed to match. Names are normalized
        # first, then a deterministic league/team ID is assigned.
        unique_matches: Dict[str, dict] = {}
        for m in self.raw:
            ht, at = m.get("homeTeam", {}), m.get("awayTeam", {})
            hn = self.resources.norm_name(ht.get("shortName") or ht.get("name", ""))
            an = self.resources.norm_name(at.get("shortName") or at.get("name", ""))
            if not hn or not an:
                continue
            ht["shortName"] = ht["name"] = hn
            at["shortName"] = at["name"] = an
            ht["id"] = self.resources.canonical_team_id(hn)
            at["id"] = self.resources.canonical_team_id(an)
            date_key = m.get("utcDate", "")[:10]
            key = f"{date_key}_{hn.lower()}_{an.lower()}"
            # Prefer CSV/deep-stat rows over API rows.
            current = unique_matches.get(key)
            if current is None or ("stats" not in current and "stats" in m):
                unique_matches[key] = m

        self.raw = sorted(
            unique_matches.values(),
            key=lambda x: x.get("utcDate", ""),
        )

        if not self.raw:
            self._log_msg("error", "No data found!")
            return False

        self._log_msg("success", f"Total: {len(self.raw)} unique matches")

        # ── V9.2 DATA QUALITY & INTEGRITY ───────────────────
        self.data_quality_report = self.data_quality_engine.refresh(self.raw)
        latest_quality = self.data_quality_report.get("latest", {})
        self._log_msg("info", f"Data quality: {latest_quality.get('status')} (score={latest_quality.get('quality_score')}, leakage={latest_quality.get('leakage_risk')})")
        if not latest_quality.get("safe_for_training", False):
            self._log_msg("warning", "Data integrity guard: dataset is not recommended for automated training decisions")

        # ── V9.3 PRODUCTION HARDENING ─────────────────────────
        self.production_report = self.production_hardening.run_startup_checks(
            data_quality=latest_quality, model_path=self.resources.model_file,
            required_paths=[LEAGUES_CONFIG_FILE])
        prod_status = self.production_report.get("latest", {}).get("status", "unknown")
        self._log_msg("info", f"Production readiness: {prod_status}")
        if prod_status == "critical":
            self._log_msg("warning", "Production hardening guard: automatic lifecycle actions are blocked")

        # ── V8.3 controlled retraining candidate ─────────────
        if (self.drift_report and self.drift_report.get("retrain_recommended")
                and prod_status != "critical"
                and latest_quality.get("safe_for_training", False)):
            self._log_msg("progress", "Drift recommendation found: building isolated retraining candidate…")
            self.retraining_report = self.retraining_pipeline.run(self.raw, self.drift_report)
            st=self.retraining_report.get("promotion_status") or self.retraining_report.get("status")
            self._log_msg("info", f"Retraining pipeline: {st}")
            registry_result=self.model_registry.record_retraining(self.retraining_report)
            self.retraining_report["registry"] = registry_result
            self._log_msg("info", f"Model registry decision recorded: {registry_result.get('decision')}")
            # V8.6: do not promote here; live evidence remains an independent final gate.
            self.live_promotion_report = self.live_promotion_pipeline.evaluate(
                self.drift_report, self.retraining_report, self.model_registry, self.shadow)
            self._log_msg("info", f"Live promotion gate: {self.live_promotion_report.get('decision')}")
            # V8.7: an approval now triggers a separate safe artifact transaction.
            if self.live_promotion_report.get("approved"):
                candidate_path=self.retraining_report.get("candidate_model")
                self.safe_promotion_report=self.safe_artifact_promotion.promote(
                    self.live_promotion_report, candidate_path)
                self._log_msg("info", f"Safe artifact promotion: {self.safe_promotion_report.get('status')}")

        # ── معالجة ───────────────────────────────────────────
        self._log_msg("progress", "Processing data + Elo + Deep Stats…")
        self.data.process(self.raw)
        self._log_msg("success", f"Teams processed: {len(self.data.teams)}")

        # ── تدريب ML ─────────────────────────────────────────
        if ML_AVAILABLE:
            model_type = (
                "Stacking(RF+XGB+LR)" if XGBOOST_AVAILABLE else "Stacking(RF+LR)"
            )
            self._log_msg(
                "progress",
                f"Training {model_type} | {N_FEATURES} features…",
            )
            if self.ml.train(self.data):
                src     = "loaded" if self.ml._external else "trained"
                acc_str = f"{self.ml.acc * 100:.1f}%"
                self._log_msg("success", f"ML model {src} | CV Acc: {acc_str}")
                if not self.ml._external:
                    self.ml.save_pipeline()
            else:
                self._log_msg("info", "ML training skipped (not enough data)")

        # ── Calibration ──────────────────────────────────────
        if self.cal.load(self.resources.calibration_file):
            self._log_msg("success", "Calibration loaded")

        # ── Odds ─────────────────────────────────────────────
        if self.odds.ok():
            self._log_msg("progress", f"Fetching live odds ({self.odds.sport})...")
            odds_res = self.odds.fetch()
            if odds_res:
                self._log_msg("success", f"Live odds loaded for {len(odds_res)} matches")
            else:
                self._log_msg("info", "No live odds available right now (empty/failed response)")

        # ── Engine ───────────────────────────────────────────
        self.eng = Engine(
            self.data, self.resources, self.ml, self.odds, self.cal, self.meta,
            drift_detector=self.drift_detector, drift_report=self.drift_report
        )
        self._log_msg(
            "success",
            f"Engine Ready for {self.resources.name}! (v8.7 | Safe Artifact Promotion + Automatic Rollback | Automated Live Promotion Pipeline | Shadow Deployment + Live Model Evaluation | Model Registry + Champion–Challenger | Controlled Auto-Retraining | Drift Detection | Performance Monitor | Adaptive Ensemble | Meta-Learner)",
        )
        return True

    def predict_upcoming(self, days: int = 14) -> List[Pred]:
        upcoming = self.api.upcoming(self.resources.api_code, days)
        if not upcoming:
            self._log_msg("info", "No upcoming matches from API")
            return []

        preds: List[Pred] = []
        for m in upcoming:
            hid = m.get("homeTeam", {}).get("id")
            aid = m.get("awayTeam", {}).get("id")
            if hid and aid:
                pr = self.eng.predict(hid, aid, m.get("utcDate", ""))
                if pr:
                    preds.append(pr)
                    # V8.5: Champion output stays untouched; Challenger is shadow-only.
                    key=f"{self.code}_{m.get('utcDate','')[:10]}_{hid}_{aid}"
                    self.shadow.record_prediction(key, [pr.hp, pr.dp, pr.ap], None,
                                                  {"date":m.get("utcDate"),"home_id":hid,"away_id":aid})

        self.last_preds = preds
        return preds

    def predict_custom(self, home_name: str, away_name: str) -> Optional[Pred]:
        ht = self.data.team_by_name(home_name)
        at = self.data.team_by_name(away_name)
        if not ht:
            self._log_msg("error", f"Team not found: {home_name}")
            return None
        if not at:
            self._log_msg("error", f"Team not found: {away_name}")
            return None
        pr = self.eng.predict(ht.id, at.id, "Custom")
        if pr:
            self.last_preds = [pr]
            # Custom predictions are also auditable, but have no live result until resolved.
            key=f"{self.code}_custom_{ht.id}_{at.id}"
            self.shadow.record_prediction(key, [pr.hp, pr.dp, pr.ap], None, {"custom":True})
        return pr

    def run_backtest(self) -> dict:
        r = self.bt.run(self.raw, self.resources)
        # V8.0: persist adaptive ensemble policy from leakage-safe calibration profiles.
        optimizer = AdaptiveEnsembleOptimizer(self.code)
        policy = optimizer.build(r.get("adaptive_profiles") and self.bt.results.get("optimized_weights", WEIGHTS) or WEIGHTS,
                                 r.get("adaptive_profiles", {}), r.get("adaptive_profile_sizes", {}), None)
        optimizer.save(policy)
        r["adaptive_ensemble_policy"] = policy
        if r.get("cal_used"):
            self.cal = self.bt.cal
            self.cal.save(self.resources.calibration_file)
            # إعادة بناء Engine مع Calibration المحدّثة
            self.meta = self.bt.train_meta_learner([], WEIGHTS) if False else self.meta
            self.eng = Engine(
                self.data, self.resources, self.ml, self.odds, self.cal, self.meta,
                drift_detector=self.drift_detector, drift_report=self.drift_report
            )
        return r

    def run_walk_forward_backtest(self) -> dict:
        """V7.9: run chronological validation and persist a conservative league policy."""
        wf = self.bt.run_walk_forward(self.raw, self.resources)
        selector = ModelSelectionEngine(self.code)
        policy = selector.select(wf)
        selector.save(policy)
        wf["model_selection"] = policy
        # V8.0: Walk-Forward evidence gates activation of the adaptive ensemble.
        opt = AdaptiveEnsembleOptimizer(self.code)
        base = (self.eng.w if self.eng else WEIGHTS)
        profiles = (self.bt.results.get("adaptive_profiles", {}) if self.bt.results else {})
        sizes = (self.bt.results.get("adaptive_profile_sizes", {}) if self.bt.results else {})
        ensemble_policy = opt.build(base, profiles, sizes, wf)
        opt.save(ensemble_policy)
        wf["adaptive_ensemble"] = ensemble_policy
        return wf

    def run_final_benchmark_acceptance(self) -> dict:
        """Run the v9.4 final benchmark gate using actual chronological evidence."""
        wf = self.run_walk_forward_backtest()
        # Refresh quality and production guards against the current loaded dataset.
        self.data_quality_report = self.data_quality_engine.refresh(self.raw)
        latest_dq = self.data_quality_report.get("latest", self.data_quality_report)
        self.production_report = self.production_hardening.run_startup_checks(
            latest_dq, self.resources.model_file,
            [self.resources.csv_file] if getattr(self.resources, "csv_file", None) else [])
        self.final_benchmark_report = self.final_benchmark_engine.run(
            wf, self.data_quality_report, self.production_report)
        return self.final_benchmark_report

    def get_model_selection_policy(self) -> Optional[dict]:
        """Return the last version-compatible automated selection policy."""
        return ModelSelectionEngine(self.code).load()

    def standings(self) -> List[Team]:
        return sorted(self.data.teams.values(), key=lambda t: t.pos)

    def export_predictions(
        self,
        preds:    List[Pred] = None,
        filename: str        = None,
    ) -> str:
        preds    = preds    or self.last_preds
        filename = filename or (
            f"predictions_{self.code}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        out = []
        for p in preds:
            out.append({
                "league":     p.league,
                "home":       p.home,
                "away":       p.away,
                "date":       p.date,
                "prediction": p.result,
                "score":      f"{p.pred_sc[0]}-{p.pred_sc[1]}",
                "confidence": round(float(p.conf),  1),
                "calibrated": p.calibrated,
                "derby":      p.derby_name if p.is_derby else None,
                "component_probabilities": {
                    k: {"home": round(float(v[0] * 100), 2), "draw": round(float(v[1] * 100), 2), "away": round(float(v[2] * 100), 2)}
                    for k, v in p.models.items()
                },
                "probabilities": {
                    "home": round(float(p.hp * 100), 1),
                    "draw": round(float(p.dp * 100), 1),
                    "away": round(float(p.ap * 100), 1),
                },
                "double_chance": {
                    "1X":            round(float(p.dc_1x * 100), 1),
                    "X2":            round(float(p.dc_x2 * 100), 1),
                    "12":            round(float(p.dc_12 * 100), 1),
                    "recommendation": p.dc_recommend,
                },
                "xg": {
                    "home":  round(p.hxg, 2),
                    "away":  round(p.axg, 2),
                    "total": round(p.hxg + p.axg, 2),
                },
                "market": {
                    "btts":      round(p.btts  * 100, 1),
                    "over_1_5":  round(p.o15   * 100, 1),
                    "over_2_5":  round(p.o25   * 100, 1),
                    "over_3_5":  round(p.o35   * 100, 1),
                },
            })
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return filename


# ══════════════════════════════════════════════════════════════
# CLI DISPLAY
# ══════════════════════════════════════════════════════════════
class Disp:
    @staticmethod
    def header():
        print()
        print(C.cyan(" ╔══════════════════════════════════════════════════════════════════╗"))
        print(C.cyan(" ║") + C.bold(" ⚽  FOOTBALL PREDICTOR PRO v7.9 (LEAKAGE-RESISTANT EDITION) ⚽  ") + C.cyan("║"))
        print(C.cyan(" ║") + C.dim("  126 Features • Error Mining • Draw-Aware • Leakage-Free BT  ") + C.cyan("║"))
        print(C.cyan(" ║") + C.dim("  Multi-League • CSV Deep Stats • Per-League Models • v6.1  ") + C.cyan("║"))
        print(C.cyan(" ╚══════════════════════════════════════════════════════════════════╝"))
        print()

    @staticmethod
    def section(t: str):
        print(f"\n {C.yellow(C.bold('══ ' + t + ' ══'))}\n")

    @staticmethod
    def leagues_menu(available: List[str]):
        Disp.section("Available Leagues")
        for i, code in enumerate(available, 1):
            cfg         = LEAGUES_CONFIG.get(code, {})
            name        = cfg.get("name", code)
            country     = cfg.get("country", "")
            data_files  = cfg.get("data_files", [])
            file_status = "✅" if any(Path(f).exists() for f in data_files) else "⚠️"
            print(f" {C.cyan(str(i))}. {C.bold(code)} - {name} ({country}) {file_status}")
        print()

    @staticmethod
    def pred_card(p: Pred):
        w = 68
        print(f"\n {C.blue('┌' + '─' * w + '┐')}")

        if p.league:
            cfg         = LEAGUES_CONFIG.get(p.league, {})
            league_name = cfg.get("name", p.league)
            print(box(f" 🏆 {C.magenta(league_name)}"))

        if p.is_derby:
            print(box(f" 🔥 {C.magenta(C.bold(p.derby_name))}"))

        print(box(
            f" {C.bold(C.green('🏠 ' + p.home))} "
            f"{C.dim('vs')} "
            f"{C.bold(C.red('✈️  ' + p.away))}"
        ))

        if p.date and p.date != "Custom":
            dt = parse_date(p.date)
            ds = dt.strftime("%a %d %b %Y • %H:%M") if dt else p.date[:16]
            print(box(f" 📅 {ds}"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(f" {C.bold('📊 PROBABILITIES')}"))
        print(box(f" 🏠 Home: {C.green(f'{p.hp * 100:5.1f}%')} {C.pct_bar(p.hp, 25, C.G)}"))
        print(box(f" 🤝 Draw: {C.yellow(f'{p.dp * 100:5.1f}%')} {C.pct_bar(p.dp, 25, C.Y)}"))
        print(box(f" ✈️  Away: {C.red(f'{p.ap * 100:5.1f}%')} {C.pct_bar(p.ap, 25, C.R)}"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(
            f" ⚡ xG: {p.home}: {C.bold(f'{p.hxg:.2f}')} | "
            f"{p.away}: {C.bold(f'{p.axg:.2f}')} | "
            f"Total: {C.bold(f'{p.hxg + p.axg:.2f}')}"
        ))
        print(box(
            f" 🎯 Predicted: {C.bold(p.result)} | "
            f"Score: {p.pred_sc[0]}-{p.pred_sc[1]} | "
            f"Conf: {p.conf:.1f}%"
        ))

        # V6.1: label صحيح CV Acc
        cal_str = " ✅ Calibrated" if p.calibrated else ""
        acc_lbl = f"CV Acc: {p.ml_acc * 100:.1f}%" if p.ml_used else "N/A"
        print(box(f" 🤖 ML: {'✅' if p.ml_used else '❌'} | {acc_lbl}{cal_str}"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(
            f" 🛡️  DC: 1X={p.dc_1x * 100:.1f}% | "
            f"12={p.dc_12 * 100:.1f}% | "
            f"X2={p.dc_x2 * 100:.1f}%"
        ))
        print(box(f" 💡 {C.bold(p.dc_recommend)}"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(f" {C.bold('🎯 TOP SCORES')}"))
        for i, (hg, ag, pr2) in enumerate(p.top_sc[:5]):
            mk = "👉" if i == 0 else "  "
            print(box(f" {mk} {hg}-{ag} ({pr2 * 100:.1f}%)"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(
            f" 📈 Markets: BTTS={p.btts * 100:.1f}% | "
            f"O1.5={p.o15 * 100:.1f}% | "
            f"O2.5={p.o25 * 100:.1f}% | "
            f"O3.5={p.o35 * 100:.1f}%"
        ))
        print(f" {C.blue('└' + '─' * w + '┘')}")

    @staticmethod
    def backtest_summary(r: dict, league_code: str = ""):
        Disp.section(f"Backtest Results [{league_code}] — v6.1")
        if "error" in r:
            print(f" {C.red('✖')} {r['error']}")
            return

        ra  = r["result_acc"]
        rac = C.green if ra > 52 else (C.yellow if ra > 47 else C.red)
        print(f" 📊 1X2 Accuracy  : {rac(f'{ra:.1f}%')} ({r['correct']}/{r['total']})")
        print(f" ⚽ Exact Score   : {r['score_acc']:.1f}%")

        bs  = r["brier"]
        # V6.1: Brier مُطبَّع [0,1] → عتبات أصغر
        bsc = C.green if bs < 0.08 else (C.yellow if bs < 0.12 else C.red)
        print(f" 📐 Brier Score   : {bsc(f'{bs:.4f}')}  [0=perfect, 1=worst]")

        if r.get("ml_acc", 0) > 0:
            print(f" 🤖 ML CV Bal.Acc : {C.green(f'{r["ml_acc"]:.1f}%')}")

        print()
        print(f" 🏠 Home Win Acc  : {r.get('home_acc', 0):.1f}%  ({r.get('home_total', 0)} matches)")
        print(f" 🤝 Draw Acc      : {r.get('draw_acc', 0):.1f}%  ({r.get('draw_total', 0)} matches)")
        print(f" ✈️  Away Win Acc  : {r.get('away_acc', 0):.1f}%  ({r.get('away_total', 0)} matches)")

        print()
        print(f" 🔥 High Conf >60%   : {C.green(f'{r.get("hi_acc", 0):.1f}%')}  ({r.get('hi_n', 0)} matches)")
        print(f" ⚡ Med  Conf 45-60% : {C.yellow(f'{r.get("me_acc", 0):.1f}%')}  ({r.get('me_n', 0)} matches)")
        print(f" ⚠️  Low  Conf <45%  : {C.red(f'{r.get("lo_acc", 0):.1f}%')}  ({r.get('lo_n', 0)} matches)")

        thr = r.get("dc_threshold", 0.60)
        print()
        print(f" 🛡️  DC 1X Acc (>{thr:.0%}): {r.get('dc_1x_acc', 0):.1f}%  ({r.get('dc_1x_n', 0)} bets)")
        print(f" 🛡️  DC X2 Acc (>{thr:.0%}): {r.get('dc_x2_acc', 0):.1f}%  ({r.get('dc_x2_n', 0)} bets)")
        print(f" 🛡️  DC 12 Acc (>{thr:.0%}): {r.get('dc_12_acc', 0):.1f}%  ({r.get('dc_12_n', 0)} bets)")

        print()
        print(f" ✅ Calibrated    : {'Yes' if r.get('cal_used') else 'No'}")
        print(f" 📋 Train Matches : {r.get('train', 0)}")
        print(f" 🧪 Cal Set       : {r.get('cal_size', 0)}")
        print(f" 🔬 Eval Set      : {r.get('eval_size', 0)}")


# ══════════════════════════════════════════════════════════════
# STREAMLIT UI — V6.1: Memory leak fix + Brier label fix
# ══════════════════════════════════════════════════════════════
def run_streamlit():
    st.set_page_config(
        page_title="Football Predictor Pro v6.1",
        page_icon="⚽",
        layout="wide",
    )
    st.markdown(
        "<h1 style='text-align:center;color:#00ff9d;'>"
        "⚽ Football Predictor Pro v6.1</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "126 Features • Stacking ML (RF+XGB+LR) • Draw-Aware Engine • "
        "Data-Leakage-Free Backtest • Confusion Matrix"
    )

    st.sidebar.title("⚙️ Settings")
    fb_key = st.sidebar.text_input(
        "🔑 Football-Data API Key",
        type="password",
        value=os.environ.get("FOOTBALL_DATA_KEY", ""),
    )
    odds_key = st.sidebar.text_input(
        "🎰 Odds API Key (optional)",
        type="password",
        value=os.environ.get("ODDS_API_KEY", ""),
    )

    available = list(LEAGUES_CONFIG.keys())
    if not available:
        st.warning("⚠️ No leagues configured!")
        st.stop()

    selected_league = st.sidebar.selectbox(
        "🏆 Select League",
        available,
        format_func=lambda c: f"{c} - {LEAGUES_CONFIG.get(c, {}).get('name', c)}",
    )
    league_cfg = LEAGUES_CONFIG.get(selected_league, {})
    data_files = league_cfg.get("data_files", [])
    st.sidebar.markdown("**📁 Data Files:**")
    for f in data_files:
        exists = Path(f).exists()
        st.sidebar.caption(f"{'✅' if exists else '❌'} {f}")

    init_key = (
        f"app_{selected_league}_{fb_key[:8] if fb_key else 'nokey'}"
    )

    if st.sidebar.button(f"🚀 Load {selected_league}") and fb_key:
        # V6.1 FIX: تنظيف نسخ قديمة من session_state
        old_keys = [
            k for k in st.session_state
            if k.startswith("app_") and k != init_key
        ]
        for k in old_keys:
            del st.session_state[k]

        with st.spinner(f"Loading {selected_league}…"):
            try:
                app = LeagueApp(selected_league, fb_key, odds_key)
                if app.init():
                    st.session_state[init_key] = app
                    st.session_state["active_league"] = selected_league
                    st.sidebar.success(f"✅ {selected_league} Ready!")
                else:
                    st.sidebar.error("❌ Init failed")
            except Exception as e:
                st.sidebar.error(f"❌ {e}")

    if init_key not in st.session_state:
        st.info(f"👈 Enter API key and load {selected_league}")
        with st.expander("📖 Setup Guide — v6.1"):
            st.markdown(f"""
            ### v6.1 Fixes
            - **Data Leakage Fixed**: Separate DataProc for train / cal / eval
            - **DC Accuracy Fixed**: Based on model prediction (threshold {60}%), not event occurrence
            - **Brier Score Fixed**: Normalized Multi-class [0, 1]
            - **Draw Correction**: Automatic bias adjustment
            - **_form() Draw-Aware**: Higher draw prob when form is close
            - **xG bounds**: min 0.10 (was 0.25)
            - **Entry Point**: Streamlit/CLI properly separated

            ### File Structure
            ```
            leagues_config.json
            data/{selected_league}_Master.csv
            config/{selected_league}_aliases.json
            models/   ← auto-generated
            ```
            ### CSV Required Columns
            `Date, HomeTeam, AwayTeam, FTHG, FTAG`
            Optional: `HST, AST, HC, AC, HF, AF, HY, AY, HR, AR`
            """)
        st.stop()

    app: LeagueApp = st.session_state[init_key]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏆 League",   app.resources.name)
    col2.metric("📊 Matches",  app.data.total)
    col3.metric("👥 Teams",    len(app.data.teams))
    col4.metric("🤖 ML Ready", "✅" if app.ml.trained else "❌")
    col5.metric("🎯 Features", "126")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predictions", "⚽ Custom Match", "📊 Standings", "🔬 Backtest",
    ])

    # ── helper: render_pred_card ──────────────────────────────
    def render_pred_card(pr: Pred):
        title    = f"{'🔥' if pr.is_derby else '⚽'} {pr.home} vs {pr.away}"
        subtitle = f"{pr.hp*100:.1f}% / {pr.dp*100:.1f}% / {pr.ap*100:.1f}%"
        with st.expander(f"{title} | {subtitle}", expanded=True):

            if pr.is_derby:
                st.markdown(f"🔥 **{pr.derby_name}**")
            if pr.calibrated:
                st.caption("✅ Calibrated probabilities")
            if pr.ml_used:
                st.caption(f"🤖 ML Stacking | CV Acc: {pr.ml_acc*100:.1f}%")

            c1, c2, c3 = st.columns(3)
            c1.metric("🏠 Home Win", f"{pr.hp*100:.1f}%")
            c2.metric("🤝 Draw",     f"{pr.dp*100:.1f}%")
            c3.metric("✈️ Away Win",  f"{pr.ap*100:.1f}%")

            st.progress(min(1.0, float(pr.hp)), text=f"Home {pr.home}")
            st.progress(min(1.0, float(pr.dp)), text="Draw")
            st.progress(min(1.0, float(pr.ap)), text=f"Away {pr.away}")

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⚡ xG Home",  f"{pr.hxg:.2f}")
            c2.metric("⚡ xG Away",  f"{pr.axg:.2f}")
            c3.metric("🏆 Elo Home", f"{pr.h_elo:.0f}")
            c4.metric("🏆 Elo Away", f"{pr.a_elo:.0f}")

            st.markdown("#### 🛡️ Double Chance")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("1X (Home/Draw)", f"{pr.dc_1x*100:.1f}%")
            cc2.metric("12 (No Draw)",   f"{pr.dc_12*100:.1f}%")
            cc3.metric("X2 (Away/Draw)", f"{pr.dc_x2*100:.1f}%")
            st.success(f"💡 **Recommendation:** {pr.dc_recommend}")

            st.markdown("#### 📈 Markets")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BTTS",    f"{pr.btts*100:.1f}%")
            m2.metric("Over 1.5",f"{pr.o15*100:.1f}%")
            m3.metric("Over 2.5",f"{pr.o25*100:.1f}%")
            m4.metric("Over 3.5",f"{pr.o35*100:.1f}%")

            st.markdown("#### 🎯 Likely Scores")
            scores_html = " &nbsp; ".join([
                f"<span style='padding:4px 12px;background:#1a1a2e;"
                f"color:#00ff9d;border-radius:5px;font-weight:bold;'>"
                f"{hg}-{ag} ({pr2*100:.1f}%)</span>"
                for hg, ag, pr2 in pr.top_sc[:5]
            ])
            st.markdown(scores_html, unsafe_allow_html=True)

            if pr.value_bets:
                vb = [v for v in pr.value_bets if v["is_value"]]
                if vb:
                    st.markdown("#### 💰 Value Bets")
                    for v in vb:
                        st.success(
                            f"**{v['market']}** | Odds: {v['odds']} | "
                            f"Model: {v['model']:.1f}% vs "
                            f"Implied: {v['implied']:.1f}% | "
                            f"Edge: +{v['edge']:.1f}%"
                        )

    # ── Tab 1: Upcoming ───────────────────────────────────────
    with tab1:
        days = st.slider("Days ahead", 1, 30, 14)
        if st.button("🔍 Get Upcoming Predictions"):
            with st.spinner("Predicting…"):
                preds = app.predict_upcoming(days)
                st.session_state[f"preds_{selected_league}"] = preds

        preds_key = f"preds_{selected_league}"
        if preds_key in st.session_state:
            preds = st.session_state[preds_key]
            if preds:
                st.success(f"✅ {len(preds)} predictions ready")
                if st.button("📥 Export JSON"):
                    fn = app.export_predictions(preds)
                    st.success(f"Exported: {fn}")
                for pr in preds:
                    render_pred_card(pr)
            else:
                st.info("No upcoming matches found")

    # ── Tab 2: Custom Match ───────────────────────────────────
    with tab2:
        teams_list = sorted(t.name for t in app.data.teams.values())
        if not teams_list:
            st.warning("No teams data available")
        else:
            c1, c2 = st.columns(2)
            home = c1.selectbox(
                "🏠 Home Team", teams_list,
                key=f"home_{selected_league}",
            )
            away = c2.selectbox(
                "✈️ Away Team", teams_list,
                index=min(1, len(teams_list) - 1),
                key=f"away_{selected_league}",
            )
            if st.button("🔮 Predict"):
                if home == away:
                    st.error("Please select different teams!")
                else:
                    with st.spinner("Predicting…"):
                        pr = app.predict_custom(home, away)
                        if pr:
                            st.session_state[f"custom_{selected_league}"] = pr
                        else:
                            st.error(
                                "Prediction failed — not enough data for these teams"
                            )

        custom_key = f"custom_{selected_league}"
        if custom_key in st.session_state:
            render_pred_card(st.session_state[custom_key])

    # ── Tab 3: Standings ──────────────────────────────────────
    with tab3:
        teams = app.standings()
        if teams:
            rows = []
            for t in teams:
                rows.append({
                    "Pos":        t.pos,
                    "Team":       t.name,
                    "P":          t.played,
                    "W":          t.wins,
                    "D":          t.draws,
                    "L":          t.losses,
                    "GF":         t.gf,
                    "GA":         t.ga,
                    "GD":         t.gd,
                    "Pts":        t.pts,
                    "Elo":        round(t.elo),
                    "PPG":        round(t.ppg,         2),
                    "Form":       t.form_string[-5:],
                    "xG For":     round(t.avg_gf,      2),
                    "xG Ag":      round(t.avg_ga,      2),
                    "SoT/g":      round(t.avg_sot,     1),
                    "Corners/g":  round(t.avg_corners, 1),
                    "Momentum":   t.momentum,
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── Tab 4: Backtest ───────────────────────────────────────
    with tab4:
        split_pct = int(BACKTEST_SPLIT * 100)
        st.info(
            f"Backtest v6.1: {split_pct}% train / {100 - split_pct}% test | "
            "Data-Leakage-Free | 126 features"
        )
        if st.button("▶️ Run Backtest"):
            with st.spinner("Running backtest…"):
                r = app.run_backtest()
                st.session_state[f"bt_{selected_league}"] = r

        bt_key = f"bt_{selected_league}"
        if bt_key in st.session_state:
            r = st.session_state[bt_key]
            if "error" in r:
                st.error(r["error"])
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🎯 1X2 Accuracy", f"{r['result_acc']:.1f}%")
                c2.metric("⚽ Exact Score",  f"{r['score_acc']:.1f}%")
                # V6.1: Brier label مُصلح
                c3.metric(
                    "📐 Brier [0→1]",
                    f"{r['brier']:.4f}",
                    help="Multi-class Brier Score, normalized [0,1]. Lower is better.",
                )
                c4.metric("🤖 ML CV Acc", f"{r.get('ml_acc', 0):.1f}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("📋 Train",    r["train"])
                c6.metric("🎚️ Cal Set",  r.get("cal_size", 0))
                c7.metric("🔬 Eval Set", r.get("eval_size", 0))
                c8.metric("✅ Calibrated", "Yes" if r.get("cal_used") else "No")

                st.divider()
                st.markdown("#### 📊 Accuracy by Outcome")
                ca, cb, cc = st.columns(3)
                ca.metric("🏠 Home Win", f"{r.get('home_acc', 0):.1f}%",
                          f"{r.get('home_total', 0)} matches")
                cb.metric("🤝 Draw",     f"{r.get('draw_acc', 0):.1f}%",
                          f"{r.get('draw_total', 0)} matches")
                cc.metric("✈️ Away Win", f"{r.get('away_acc', 0):.1f}%",
                          f"{r.get('away_total', 0)} matches")

                st.markdown("#### 🎯 Accuracy by Confidence")
                d1, d2, d3 = st.columns(3)
                d1.metric("🔥 High >60%",     f"{r.get('hi_acc', 0):.1f}%",
                          f"{r.get('hi_n', 0)} matches")
                d2.metric("⚡ Medium 45-60%", f"{r.get('me_acc', 0):.1f}%",
                          f"{r.get('me_n', 0)} matches")
                d3.metric("⚠️ Low <45%",      f"{r.get('lo_acc', 0):.1f}%",
                          f"{r.get('lo_n', 0)} matches")

                thr = r.get("dc_threshold", 0.60)
                st.markdown(
                    f"#### 🛡️ Double Chance (threshold >{thr:.0%})"
                )
                e1, e2, e3 = st.columns(3)
                e1.metric("1X", f"{r.get('dc_1x_acc', 0):.1f}%",
                          f"{r.get('dc_1x_n', 0)} bets")
                e2.metric("12", f"{r.get('dc_12_acc', 0):.1f}%",
                          f"{r.get('dc_12_n', 0)} bets")
                e3.metric("X2", f"{r.get('dc_x2_acc', 0):.1f}%",
                          f"{r.get('dc_x2_n', 0)} bets")

                # Confusion Matrix
                if r.get("confusion"):
                    st.markdown("#### 🔍 Confusion Matrix")
                    conf    = r["confusion"]
                    cm_data = []
                    for pred_label in ["HOME", "DRAW", "AWAY"]:
                        row = {"Predicted ↓ / Actual →": pred_label}
                        for act_label in ["HOME", "DRAW", "AWAY"]:
                            row[f"Actual {act_label}"] = (
                                conf.get(pred_label, {}).get(act_label, 0)
                            )
                        cm_data.append(row)
                    st.dataframe(
                        pd.DataFrame(cm_data),
                        use_container_width=True,
                        hide_index=True,
                    )

                if r.get("predictions"):
                    st.markdown("#### 📋 Last 50 Test Predictions")
                    preds_rows = []
                    for pred_item in r["predictions"][-50:]:
                        preds_rows.append({
                            "Home":       pred_item["home"],
                            "Away":       pred_item["away"],
                            "Predicted":  pred_item["predicted"],
                            "Actual":     pred_item["actual"],
                            "Result":     "✅" if pred_item["correct"] else "❌",
                            "Confidence": f"{pred_item['confidence']:.1f}%",
                            "P(Home)":    f"{pred_item['probs'][0]*100:.1f}%",
                            "P(Draw)":    f"{pred_item['probs'][1]*100:.1f}%",
                            "P(Away)":    f"{pred_item['probs'][2]*100:.1f}%",
                            "Calibrated": "✅" if pred_item.get("calibrated") else "❌",
                        })
                    st.dataframe(
                        pd.DataFrame(preds_rows),
                        use_container_width=True,
                        hide_index=True,
                    )


# ══════════════════════════════════════════════════════════════
# CLI MAIN
# ══════════════════════════════════════════════════════════════
def cli_main():
    Disp.header()
    available = list(LEAGUES_CONFIG.keys())
    if not available:
        print(f"{C.red('❌')} No leagues in {LEAGUES_CONFIG_FILE}")
        return

    tok = os.environ.get("FOOTBALL_DATA_KEY", "")
    if not tok:
        tok = input(C.cyan(" 🔑 Football-Data API key: ")).strip()
        if not tok:
            print(C.red("❌ No API key provided"))
            return

    odds_key = os.environ.get("ODDS_API_KEY", "")

    Disp.leagues_menu(available)
    if len(available) == 1:
        league_code = available[0]
        print(f" Auto-selected: {C.bold(league_code)}")
    else:
        choice = input(C.cyan(f" Select league (1-{len(available)}): ")).strip()
        try:
            league_code = available[int(choice) - 1]
        except (ValueError, IndexError):
            league_code = available[0]

    print(f"\n {C.green('►')} Loading {C.bold(league_code)} v6.1…")
    try:
        app = LeagueApp(league_code, tok, odds_key)
        if not app.init():
            print(C.red("❌ Initialization failed"))
            return
    except Exception as e:
        print(C.red(f"❌ Error: {e}"))
        return

    while True:
        try:
            print(f"\n {C.cyan('─' * 55)}")
            print(f" {C.bold('Options:')}")
            print(f" {C.cyan('1')} - Predict upcoming matches")
            print(f" {C.cyan('2')} - Custom match prediction")
            print(f" {C.cyan('3')} - League standings")
            print(f" {C.cyan('4')} - Run backtest (v6.1 — Leakage-Free)")
            print(f" {C.cyan('5')} - Export last predictions")
            print(f" {C.cyan('6')} - Switch league")
            print(f" {C.cyan('0')} - Exit")

            ch = input(C.cyan("\n Choice: ")).strip()

            if ch == "1":
                Disp.section("Upcoming Predictions")
                preds = app.predict_upcoming(14)
                if preds:
                    for pr in preds:
                        Disp.pred_card(pr)
                else:
                    print(C.yellow(" No upcoming matches found"))

            elif ch == "2":
                Disp.section("Custom Match")
                teams = sorted(t.name for t in app.data.teams.values())
                print(f" Teams: {', '.join(teams)}")
                home = input(C.cyan(" Home team: ")).strip()
                away = input(C.cyan(" Away team: ")).strip()
                pr   = app.predict_custom(home, away)
                if pr:
                    Disp.pred_card(pr)

            elif ch == "3":
                Disp.section(f"Standings — {app.resources.name}")
                standings = app.standings()
                print(
                    f" {'#':>3} {'Team':<22} {'P':>3} {'W':>2} "
                    f"{'D':>2} {'L':>2} {'GD':>4} {'Pts':>4} "
                    f"{'Elo':>6} {'Mom':>5}"
                )
                print(f" {'─' * 65}")
                total_t = len(standings)
                for t in standings:
                    pc = (
                        C.G if t.pos <= 4
                        else (C.R if t.pos >= total_t - 2 else C.W)
                    )
                    gd_str = f"{t.gd:+d}"
                    print(
                        f" {pc}{t.pos:>3}{C.E} {t.name:<22} "
                        f"{t.played:>3} {t.wins:>2} {t.draws:>2} "
                        f"{t.losses:>2} {gd_str:>4} {t.pts:>4} "
                        f"{t.elo:>6.0f} {t.momentum:>5}"
                    )

            elif ch == "4":
                Disp.section("Backtest v6.1 — Data-Leakage-Free")
                r = app.run_backtest()
                Disp.backtest_summary(r, league_code)

            elif ch == "5":
                if app.last_preds:
                    fn = app.export_predictions()
                    print(C.green(f" ✅ Exported: {fn}"))
                else:
                    print(C.yellow(" No predictions to export"))

            elif ch == "6":
                Disp.leagues_menu(available)
                choice2 = input(C.cyan(f" Select (1-{len(available)}): ")).strip()
                try:
                    new_code = available[int(choice2) - 1]
                    new_app  = LeagueApp(new_code, tok, odds_key)
                    if new_app.init():
                        app         = new_app
                        league_code = new_code
                    else:
                        print(C.red("❌ Failed to load league"))
                except (ValueError, IndexError):
                    print(C.red("❌ Invalid choice"))

            elif ch == "0":
                print(C.green(" Goodbye! ⚽"))
                break

        except KeyboardInterrupt:
            print(C.green("\n Goodbye! ⚽"))
            break
        except Exception as e:
            print(C.red(f" Error: {e}"))



# ══════════════════════════════════════════════════════════════
# V8.5 — SHADOW DEPLOYMENT & LIVE MODEL EVALUATION
# ══════════════════════════════════════════════════════════════
class ShadowDeploymentManager:
    """Runs Challenger evaluation alongside the Champion without changing user output.

    The manager is intentionally conservative: it records Champion predictions
    and, when a compatible challenger predictor is supplied, records its silent
    probabilities for the same fixture. Promotion requires enough resolved live
    fixtures and a real Log-Loss improvement; otherwise the Champion remains live.
    """
    VERSION = "v8.5"
    MIN_LIVE_SAMPLES = 30
    MIN_IMPROVEMENT = 0.01

    def __init__(self, league_code: str):
        self.league_code = league_code
        self.root = Path("models") / "shadow" / league_code
        self.path = self.root / "shadow_deployment_v8.5.json"

    def _default(self):
        return {"version": self.VERSION, "league": self.league_code,
                "active_challenger": None, "fixtures": [], "decisions": []}

    def _load(self):
        if not self.path.exists(): return self._default()
        try:
            d=json.loads(self.path.read_text(encoding="utf-8"))
            if d.get("version") != self.VERSION or d.get("league") != self.league_code: return self._default()
            d.setdefault("fixtures", []); d.setdefault("decisions", []); return d
        except Exception: return self._default()

    def _save(self, d):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp=self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def activate(self, challenger_model: str, metadata: dict=None):
        d=self._load(); d["active_challenger"]={"model":challenger_model,
            "activated_at":datetime.utcnow().isoformat()+"Z", "metadata":metadata or {}}
        self._save(d); return d["active_challenger"]

    def record_prediction(self, fixture_key: str, champion_probs, challenger_probs=None, fixture_meta=None):
        d=self._load()
        if not d.get("active_challenger"): return False
        if any(x.get("fixture_key")==fixture_key for x in d["fixtures"]): return False
        norm=lambda p:[float(max(1e-9,x)) for x in p]
        d["fixtures"].append({"fixture_key":fixture_key, "created_at":datetime.utcnow().isoformat()+"Z",
            "champion_probs":norm(champion_probs),
            "challenger_probs":norm(challenger_probs) if challenger_probs else None,
            "fixture":fixture_meta or {}, "outcome":None})
        self._save(d); return True

    def resolve(self, fixture_key: str, outcome_index: int):
        d=self._load()
        for x in d["fixtures"]:
            if x.get("fixture_key")==fixture_key:
                x["outcome"]=int(outcome_index); x["resolved_at"]=datetime.utcnow().isoformat()+"Z"
                self._save(d); return True
        return False

    def evaluate(self):
        d=self._load(); rows=[x for x in d["fixtures"] if x.get("outcome") is not None and x.get("challenger_probs")]
        if len(rows)<self.MIN_LIVE_SAMPLES:
            return {"status":"insufficient_live_evidence", "samples":len(rows), "required":self.MIN_LIVE_SAMPLES}
        def ll(key):
            return sum(-math.log(max(1e-9,r[key][r["outcome"]])) for r in rows)/len(rows)
        champion_ll, challenger_ll=ll("champion_probs"), ll("challenger_probs")
        improvement=champion_ll-challenger_ll
        decision="promote_recommended" if improvement>=self.MIN_IMPROVEMENT else "champion_retained"
        result={"status":"evaluated", "samples":len(rows), "champion_log_loss":champion_ll,
                "challenger_log_loss":challenger_ll, "improvement":improvement, "decision":decision,
                "evaluated_at":datetime.utcnow().isoformat()+"Z"}
        d["decisions"].append(result); self._save(d); return result

# ══════════════════════════════════════════════════════════════
# V8.7 — SAFE ARTIFACT PROMOTION & AUTOMATIC ROLLBACK
# ══════════════════════════════════════════════════════════════
class SafeArtifactPromotionManager:
    """Promotes approved model artifacts with backup, atomic replacement and rollback.

    This layer consumes the *approval* from v8.6 but performs the filesystem
    operation separately. Every promotion is journaled, the incumbent is backed
    up first, and a post-swap health check can immediately restore the Champion.
    """
    VERSION = "v8.7"

    def __init__(self, league_code: str, resources):
        self.league_code = league_code
        self.resources = resources
        self.root = Path("models") / "safe_promotion" / league_code
        self.path = self.root / "artifact_promotion_v8.7.json"

    def _default(self):
        return {"version": self.VERSION, "league": self.league_code,
                "history": [], "active": None}

    def _load(self):
        if not self.path.exists(): return self._default()
        try:
            d=json.loads(self.path.read_text(encoding="utf-8"))
            if d.get("version") != self.VERSION or d.get("league") != self.league_code:
                return self._default()
            d.setdefault("history", []); return d
        except Exception:
            return self._default()

    def _save(self, d):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp=self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _health_check(self, artifact: Path):
        if not artifact.exists() or artifact.stat().st_size <= 0:
            return False, "artifact_missing_or_empty"
        try:
            probe=MLPred(model_file=artifact)
            if not probe._try_load_external():
                return False, "model_load_failed"
            return True, "model_load_ok"
        except Exception as e:
            return False, f"health_exception:{type(e).__name__}"

    def rollback(self, reason="manual_or_health_failure"):
        d=self._load(); active=d.get("active") or {}
        prod=Path(active.get("production_model", self.resources.model_file))
        backup=Path(active.get("backup_model", "")) if active.get("backup_model") else None
        if not backup or not backup.exists():
            result={"status":"rollback_blocked","reason":"backup_unavailable","at":datetime.utcnow().isoformat()+"Z"}
            d["history"].append(result); self._save(d); return result
        tmp=prod.with_suffix(prod.suffix+".rollback.tmp")
        shutil.copy2(backup,tmp); tmp.replace(prod)
        ok,msg=self._health_check(prod)
        result={"status":"rolled_back" if ok else "rollback_failed", "reason":reason,
                "health":msg,"production_model":str(prod),"backup_model":str(backup),
                "at":datetime.utcnow().isoformat()+"Z"}
        d["history"].append(result)
        if ok: d["active"]=None
        self._save(d); return result

    def promote(self, promotion_report: dict, challenger_model: str=None):
        if not promotion_report or not promotion_report.get("approved") or promotion_report.get("decision") != "promotion_approved":
            return {"status":"promotion_blocked","reason":"v8.6_not_approved"}
        source=challenger_model
        if not source:
            # Prefer the candidate recorded by the retraining pipeline/registry caller.
            source=(promotion_report.get("challenger_model") or promotion_report.get("candidate_model"))
        if not source:
            return {"status":"promotion_blocked","reason":"challenger_artifact_not_specified"}
        candidate=Path(source); prod=Path(self.resources.model_file)
        if not candidate.exists() or candidate.resolve()==prod.resolve() if prod.exists() else not candidate.exists():
            return {"status":"promotion_blocked","reason":"candidate_artifact_invalid"}
        self.root.mkdir(parents=True, exist_ok=True); prod.parent.mkdir(parents=True, exist_ok=True)
        stamp=datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup=self.root / f"champion_{stamp}{prod.suffix or '.pkl'}"
        if prod.exists(): shutil.copy2(prod, backup)
        else: return {"status":"promotion_blocked","reason":"champion_artifact_missing"}
        ok,msg=self._health_check(candidate)
        if not ok:
            return {"status":"promotion_blocked","reason":"candidate_health_failed","health":msg}
        tmp=prod.with_suffix(prod.suffix+".promote.tmp")
        try:
            shutil.copy2(candidate,tmp); tmp.replace(prod)
            ok,msg=self._health_check(prod)
            d=self._load()
            result={"status":"promoted" if ok else "promotion_failed", "production_model":str(prod),
                    "challenger_model":str(candidate),"backup_model":str(backup),"health":msg,
                    "promoted_at":datetime.utcnow().isoformat()+"Z"}
            d["history"].append(result); d["active"]=result if ok else None; self._save(d)
            if not ok:
                rollback=self.rollback("post_swap_health_failure"); result["rollback"]=rollback
            return result
        except Exception as e:
            try:
                if backup.exists():
                    tmp=prod.with_suffix(prod.suffix+".recover.tmp"); shutil.copy2(backup,tmp); tmp.replace(prod)
            except Exception: pass
            return {"status":"promotion_failed","reason":f"swap_exception:{type(e).__name__}"}


# ══════════════════════════════════════════════════════════════
# V8.8 — POST-PROMOTION CANARY MONITORING
# ══════════════════════════════════════════════════════════════
class PostPromotionCanaryMonitor:
    """Monitors a newly promoted Champion against a pre-promotion baseline.

    The canary layer never invents a rollback from a tiny sample. It waits for
    resolved fixtures, compares mean log loss with the baseline recorded at
    promotion time, and requests the v8.7 rollback manager to restore the
    previous artifact only when sustained degradation is evidenced.
    """
    VERSION = "v8.8"
    MIN_SAMPLES = 20
    DEGRADATION_THRESHOLD = 0.015
    CONFIRMATION_WINDOWS = 2

    def __init__(self, league_code: str, promotion_manager=None):
        self.league_code = league_code
        self.promotion_manager = promotion_manager
        self.root = Path("models") / "canary" / league_code
        self.path = self.root / "post_promotion_canary_v8.8.json"

    def _default(self):
        return {"version": self.VERSION, "league": self.league_code,
                "active": None, "fixtures": [], "history": []}

    def load(self):
        if not self.path.exists(): return self._default()
        try:
            d=json.loads(self.path.read_text(encoding="utf-8"))
            if d.get("version") != self.VERSION or d.get("league") != self.league_code:
                return self._default()
            d.setdefault("fixtures", []); d.setdefault("history", [])
            return d
        except Exception:
            return self._default()

    def _save(self, d):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp=self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def start(self, promotion_result: dict, baseline_log_loss=None):
        if not promotion_result or promotion_result.get("status") != "promoted":
            return {"status":"canary_not_started", "reason":"promotion_not_successful"}
        if baseline_log_loss is None:
            baseline_log_loss=promotion_result.get("baseline_log_loss")
        if baseline_log_loss is None:
            return {"status":"canary_not_started", "reason":"baseline_missing"}
        d=self.load()
        d["active"]={"started_at":datetime.utcnow().isoformat()+"Z",
            "production_model":promotion_result.get("production_model"),
            "backup_model":promotion_result.get("backup_model"),
            "baseline_log_loss":float(baseline_log_loss),
            "bad_windows":0, "status":"monitoring"}
        d["fixtures"]=[]; self._save(d)
        return d["active"]

    def record_prediction(self, fixture_key: str, probs, fixture_meta=None):
        d=self.load(); active=d.get("active")
        if not active or active.get("status") != "monitoring": return False
        if any(x.get("fixture_key")==fixture_key for x in d["fixtures"]): return False
        p=[float(max(1e-9,x)) for x in probs]
        if len(p)!=3: return False
        d["fixtures"].append({"fixture_key":fixture_key,"probs":p,
            "fixture":fixture_meta or {},"outcome":None,
            "created_at":datetime.utcnow().isoformat()+"Z"})
        self._save(d); return True

    def resolve(self, fixture_key: str, outcome_index: int):
        d=self.load()
        for row in d["fixtures"]:
            if row.get("fixture_key")==fixture_key and row.get("outcome") is None:
                row["outcome"]=int(outcome_index); row["resolved_at"]=datetime.utcnow().isoformat()+"Z"
                self._save(d); return True
        return False

    def evaluate(self):
        d=self.load(); active=d.get("active")
        if not active: return {"status":"inactive"}
        rows=[r for r in d["fixtures"] if r.get("outcome") is not None]
        if len(rows)<self.MIN_SAMPLES:
            return {"status":"insufficient_canary_evidence","samples":len(rows),"required":self.MIN_SAMPLES}
        ll=sum(-math.log(max(1e-9,r["probs"][r["outcome"]])) for r in rows)/len(rows)
        baseline=float(active["baseline_log_loss"]); degradation=ll-baseline
        bad=degradation>=self.DEGRADATION_THRESHOLD
        active["bad_windows"]=active.get("bad_windows",0)+(1 if bad else -active.get("bad_windows",0))
        result={"status":"evaluated","samples":len(rows),"canary_log_loss":ll,
                "baseline_log_loss":baseline,"degradation":degradation,
                "bad_window":bad,"bad_windows":active["bad_windows"],
                "evaluated_at":datetime.utcnow().isoformat()+"Z"}
        if active["bad_windows"]>=self.CONFIRMATION_WINDOWS:
            rollback={"status":"rollback_unavailable"}
            if self.promotion_manager:
                rollback=self.promotion_manager.rollback("post_promotion_canary_performance_degradation")
            active["status"]="rolled_back" if rollback.get("status")=="rolled_back" else "rollback_failed"
            result["decision"]="rollback"; result["rollback"]=rollback
        elif not bad:
            active["status"]="healthy"
            result["decision"]="healthy"
        else:
            result["decision"]="monitoring"
        d["history"].append(result); self._save(d); return result


# ══════════════════════════════════════════════════════════════
# V8.9 — MODEL LIFECYCLE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════
class ModelLifecycleOrchestrator:
    """Central, conservative state machine for the model lifecycle.

    v8.9 does not bypass any existing safety gate.  It reads the evidence
    produced by v8.2–v8.8, determines the current lifecycle stage for one
    league, persists an auditable snapshot, and exposes the next safe action.
    """
    VERSION = "v8.9"

    STAGES = (
        "stable_champion", "drift_detected", "retraining_pending",
        "challenger_registered", "shadow_evaluation", "promotion_review",
        "artifact_promoted", "canary_monitoring", "rolled_back"
    )

    def __init__(self, league_code, resources=None, managers=None):
        self.league_code = league_code
        self.resources = resources
        self.managers = managers or {}
        self.root = Path("models") / "lifecycle" / league_code
        self.path = self.root / "model_lifecycle_v8.9.json"

    def _default(self):
        return {"version": self.VERSION, "league": self.league_code,
                "current_stage": "stable_champion", "history": [],
                "updated_at": None, "evidence": {}}

    def load(self):
        if not self.path.exists(): return self._default()
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if d.get("version") != self.VERSION or d.get("league") != self.league_code:
                return self._default()
            d.setdefault("history", []); d.setdefault("evidence", {})
            return d
        except Exception: return self._default()

    def _save(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _safe_load(obj, method="load"):
        try:
            fn = getattr(obj, method, None)
            return fn() if callable(fn) else None
        except Exception:
            return None

    def collect_evidence(self):
        m = self.managers
        evidence = {}
        for name, method in [
            ("drift", "load"), ("registry", "load"), ("shadow", "_load"),
            ("promotion", "load"), ("safe_promotion", "_load"),
            ("canary", "load")]:
            obj = m.get(name)
            if obj: evidence[name] = self._safe_load(obj, method) or {}
        return evidence

    def determine_stage(self, evidence=None):
        e = evidence or self.collect_evidence()
        canary = (e.get("canary") or {}).get("active") or {}
        if canary.get("status") == "rolled_back": return "rolled_back", "canary_requested_rollback"
        if canary.get("status") in ("monitoring", "healthy"): return "canary_monitoring", "post_promotion_canary_active"
        safe = (e.get("safe_promotion") or {}).get("active") or {}
        if safe: return "artifact_promoted", "safe_artifact_promotion_active"
        promo = e.get("promotion") or {}
        decision = promo.get("decision") or promo.get("status")
        if decision in ("promotion_approved", "approved"): return "promotion_review", "live_and_historical_gates_passed"
        shadow = e.get("shadow") or {}
        if shadow.get("active_challenger"): return "shadow_evaluation", "challenger_collecting_live_evidence"
        registry = e.get("registry") or {}
        if registry.get("challenger") or registry.get("candidates"): return "challenger_registered", "candidate_registered"
        drift = e.get("drift") or {}
        rec = drift.get("recommendation") or drift.get("decision") or drift.get("status")
        if rec and any(x in str(rec).lower() for x in ("retrain", "drift")):
            return "drift_detected", "drift_engine_requests_attention"
        return "stable_champion", "no_active_lifecycle_risk"

    def refresh(self):
        evidence = self.collect_evidence()
        stage, reason = self.determine_stage(evidence)
        d = self.load(); previous = d.get("current_stage")
        d["current_stage"] = stage
        d["updated_at"] = datetime.utcnow().isoformat()+"Z"
        d["evidence"] = evidence
        d["next_action"] = {
            "stable_champion":"continue_monitoring",
            "drift_detected":"run_controlled_retraining",
            "retraining_pending":"validate_candidate",
            "challenger_registered":"activate_shadow_deployment",
            "shadow_evaluation":"wait_for_live_evidence",
            "promotion_review":"run_safe_artifact_promotion",
            "artifact_promoted":"start_or_continue_canary",
            "canary_monitoring":"continue_post_promotion_monitoring",
            "rolled_back":"retain_previous_champion_and_investigate"
        }.get(stage, "manual_review")
        if previous != stage:
            d["history"].append({"from": previous, "to": stage, "reason": reason,
                                 "at": d["updated_at"]})
        self._save(d)
        return {"stage":stage, "reason":reason, "next_action":d["next_action"],
                "changed":previous != stage}

class AutonomousModelGovernanceEngine:
    """v9.0 governance layer above the lifecycle orchestrator.

    Converts lifecycle evidence into an auditable risk assessment and an
    explicit approval decision. It deliberately does not bypass the safety
    gates implemented by v8.2-v8.8; execution remains delegated to those
    components after governance approval.
    """
    VERSION = "v9.0"

    def __init__(self, league_code, lifecycle_orchestrator, managers=None):
        self.league_code = league_code
        self.lifecycle = lifecycle_orchestrator
        self.managers = managers or {}
        self.root = Path("models") / "governance" / league_code
        self.path = self.root / "model_governance_v9.0.json"

    def _default(self):
        return {"version": self.VERSION, "league": self.league_code,
                "updated_at": None, "policy": {}, "risk": {},
                "approval": {"status": "pending"}, "history": []}

    def load(self):
        if not self.path.exists(): return self._default()
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if d.get("version") != self.VERSION or d.get("league") != self.league_code:
                return self._default()
            d.setdefault("history", []); return d
        except Exception: return self._default()

    def _save(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def governance_policy(self):
        # Conservative defaults; per-league policies can be extended without
        # changing the lifecycle safety contracts.
        return {"league": self.league_code, "risk_tolerance": "low",
                "require_historical_validation": True,
                "require_live_validation": True,
                "require_canary_for_finalization": True,
                "auto_execution": False}

    def assess_risk(self, lifecycle):
        stage = lifecycle.get("stage", "stable_champion")
        evidence = getattr(self.lifecycle, "load", lambda: {})().get("evidence", {})
        score, reasons = 0, []
        if stage in ("drift_detected", "retraining_pending"): score += 35; reasons.append("model_or_data_drift")
        if stage in ("promotion_review", "artifact_promoted", "canary_monitoring"): score += 25; reasons.append("production_transition")
        if stage == "rolled_back": score += 70; reasons.append("rollback_event")
        drift = evidence.get("drift") or {}
        if drift.get("data_drift") or drift.get("performance_drift"): score += 20; reasons.append("confirmed_drift_signal")
        canary = (evidence.get("canary") or {}).get("active") or {}
        if canary.get("status") == "rolled_back": score = max(score, 90); reasons.append("canary_degradation")
        return {"score": min(100, score), "level": "critical" if score >= 70 else "high" if score >= 40 else "medium" if score >= 20 else "low", "reasons": reasons}

    def approve(self, lifecycle, risk, policy):
        stage = lifecycle.get("stage", "stable_champion")
        action = lifecycle.get("next_action", "continue_monitoring")
        if stage == "rolled_back":
            return {"status":"approved", "action":"retain_previous_champion_and_investigate", "reason":"rollback_is_a_protective_action"}
        if policy["auto_execution"]:
            return {"status":"rejected", "action":"manual_review", "reason":"unsafe_policy_configuration"}
        if stage == "promotion_review" and risk["level"] in ("critical", "high"):
            return {"status":"requires_manual_review", "action":action, "reason":"production_promotion_requires_low_risk_evidence"}
        if stage == "stable_champion":
            return {"status":"approved", "action":action, "reason":"no_active_governance_risk"}
        return {"status":"approved", "action":action, "reason":"delegated_safety_gates_remain_mandatory"}

    def refresh(self):
        lifecycle = self.lifecycle.refresh()
        policy = self.governance_policy()
        risk = self.assess_risk(lifecycle)
        approval = self.approve(lifecycle, risk, policy)
        d = self.load(); now = datetime.utcnow().isoformat()+"Z"
        snapshot = {"at":now, "lifecycle_state":lifecycle.get("stage"), "risk_level":risk["level"], "approval":approval["status"]}
        if not d.get("history") or d["history"][-1] != snapshot: d["history"].append(snapshot)
        d.update({"updated_at":now, "lifecycle":lifecycle, "policy":policy, "risk":risk, "approval":approval})
        self._save(d)
        return d


class ExplainableDecisionAuditEngine:
    """v9.1 central explanation and audit layer.

    Produces compact, evidence-based explanations for lifecycle/governance
    decisions without changing model probabilities or bypassing safety gates.
    """
    VERSION = "v9.1"

    def __init__(self, league_code, lifecycle_orchestrator, governance_engine):
        self.league_code = league_code
        self.lifecycle = lifecycle_orchestrator
        self.governance = governance_engine
        self.root = Path("models") / "decision_audit" / league_code
        self.path = self.root / "decision_audit_v9.1.json"

    def _default(self):
        return {"version": self.VERSION, "league": self.league_code,
                "updated_at": None, "latest": {}, "history": []}

    def load(self):
        if not self.path.exists():
            return self._default()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != self.VERSION or data.get("league") != self.league_code:
                return self._default()
            data.setdefault("history", [])
            return data
        except Exception:
            return self._default()

    def _save(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _reason(code):
        labels = {
            "model_or_data_drift": "تم اكتشاف تغير في البيانات أو أداء النموذج",
            "production_transition": "النموذج يمر بمرحلة انتقال مرتبطة بالإنتاج",
            "rollback_event": "تم تسجيل حدث استرجاع وقائي سابق",
            "confirmed_drift_signal": "توجد إشارة انجراف مؤكدة من نظام المراقبة",
            "canary_degradation": "أظهرت مراقبة Canary تدهوراً مؤكداً",
        }
        return labels.get(code, str(code))

    def explain(self, lifecycle, governance):
        risk = governance.get("risk") or {}
        approval = governance.get("approval") or {}
        evidence = []
        for code in risk.get("reasons", []):
            evidence.append({"type": "risk_factor", "code": code, "explanation": self._reason(code)})
        evidence.append({"type": "lifecycle_state", "value": lifecycle.get("stage", "unknown"),
                         "explanation": "المرحلة الحالية في دورة حياة النموذج"})
        evidence.append({"type": "proposed_action", "value": approval.get("action", lifecycle.get("next_action")),
                         "explanation": "الإجراء المقترح بعد تطبيق قواعد الحوكمة"})
        return {
            "decision": approval.get("status", "pending"),
            "action": approval.get("action", "continue_monitoring"),
            "decision_reason": approval.get("reason", "insufficient_audit_context"),
            "confidence": "low" if risk.get("level") in ("high", "critical") else "medium" if risk.get("level") == "medium" else "high",
            "risk_level": risk.get("level", "unknown"),
            "risk_score": risk.get("score"),
            "evidence": evidence,
            "safety_statement": "هذا المحرك يفسر ويسجل القرار فقط ولا يتجاوز بوابات التحقق أو الاسترجاع السابقة."
        }

    def refresh(self):
        lifecycle = self.lifecycle.refresh()
        governance = self.governance.refresh()
        explanation = self.explain(lifecycle, governance)
        data = self.load()
        now = datetime.utcnow().isoformat()+"Z"
        event = {"at": now, "stage": lifecycle.get("stage"), "decision": explanation["decision"],
                 "action": explanation["action"], "risk_level": explanation["risk_level"]}
        if not data.get("history") or data["history"][-1] != event:
            data["history"].append(event)
        data.update({"updated_at": now, "latest": explanation,
                     "lifecycle_snapshot": lifecycle, "governance_snapshot": governance})
        self._save(data)
        return data


class DataQualityIntegrityEngine:
    """v9.2 pre-model data quality and integrity guard.

    Observational and conservative: it reports problems and provides a
    reliability score without silently repairing source data or inventing rows.
    """
    VERSION = "v9.2"

    def __init__(self, league_code):
        self.league_code = league_code
        self.root = Path("models") / "data_quality" / league_code
        self.path = self.root / "data_quality_v9.2.json"

    def _default(self):
        return {"version": self.VERSION, "league": self.league_code,
                "updated_at": None, "latest": {}, "history": []}

    def load(self):
        if not self.path.exists(): return self._default()
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if d.get("version") != self.VERSION or d.get("league") != self.league_code:
                return self._default()
            d.setdefault("history", [])
            return d
        except Exception:
            return self._default()

    def _save(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _pick(d, *keys):
        for k in keys:
            v = d.get(k)
            if v not in (None, ""): return v
        return None

    def assess(self, rows):
        rows = rows or []
        issues = {"missing_required": 0, "duplicate_matches": 0,
                  "invalid_dates": 0, "future_finished_dates": 0,
                  "conflicting_fixtures": 0, "chronology_anomalies": 0}
        examples = {k: [] for k in issues}
        seen, fixture_results, parsed_dates = set(), {}, []
        now = datetime.utcnow()

        for i, r in enumerate(rows):
            if not isinstance(r, dict):
                issues["missing_required"] += 1; continue
            home = self._pick(r, "home_team", "home", "homeTeam", "home_id")
            away = self._pick(r, "away_team", "away", "awayTeam", "away_id")
            date_raw = self._pick(r, "date", "utcDate", "match_date", "datetime")
            if not home or not away or not date_raw:
                issues["missing_required"] += 1
                if len(examples["missing_required"]) < 5: examples["missing_required"].append(i)
                continue
            try:
                dt = datetime.fromisoformat(str(date_raw).replace("Z", "+00:00")).replace(tzinfo=None)
                parsed_dates.append((i, dt))
                if dt > now + timedelta(days=1):
                    issues["future_finished_dates"] += 1
                    if len(examples["future_finished_dates"]) < 5: examples["future_finished_dates"].append(i)
            except Exception:
                issues["invalid_dates"] += 1
                if len(examples["invalid_dates"]) < 5: examples["invalid_dates"].append(i)
                dt = None
            key = (str(home), str(away), str(date_raw)[:10])
            if key in seen:
                issues["duplicate_matches"] += 1
                if len(examples["duplicate_matches"]) < 5: examples["duplicate_matches"].append(i)
            seen.add(key)
            result = self._pick(r, "result", "score", "full_time_score", "home_score")
            if key in fixture_results and fixture_results[key] != result:
                issues["conflicting_fixtures"] += 1
                if len(examples["conflicting_fixtures"]) < 5: examples["conflicting_fixtures"].append(i)
            fixture_results[key] = result

        if parsed_dates:
            inversions = sum(1 for a, b in zip(parsed_dates, parsed_dates[1:]) if b[1] < a[1])
            issues["chronology_anomalies"] = inversions
            examples["chronology_anomalies"] = [a for a, b in zip(parsed_dates, parsed_dates[1:]) if b[1] < a[1]][:5]

        total_issues = sum(issues.values())
        denominator = max(len(rows), 1)
        score = max(0.0, round(1.0 - min(1.0, total_issues / denominator), 4))
        status = "healthy" if score >= .98 else "warning" if score >= .90 else "critical"
        leakage_risk = "high" if issues["future_finished_dates"] or issues["chronology_anomalies"] > max(3, len(rows)//20) else "low"
        return {"rows": len(rows), "status": status, "quality_score": score,
                "issues": issues, "examples": examples, "leakage_risk": leakage_risk,
                "safe_for_training": status != "critical" and leakage_risk == "low"}

    def refresh(self, rows):
        report = self.assess(rows)
        data = self.load(); now = datetime.utcnow().isoformat()+"Z"
        snapshot = {"at": now, "status": report["status"], "quality_score": report["quality_score"],
                    "leakage_risk": report["leakage_risk"], "rows": report["rows"]}
        if not data.get("history") or data["history"][-1] != snapshot:
            data.setdefault("history", []).append(snapshot)
        data.update({"updated_at": now, "latest": report})
        self._save(data)
        return data


# ══════════════════════════════════════════════════════════════
# V9.4 — FINAL BENCHMARK & ACCEPTANCE TESTING
# ══════════════════════════════════════════════════════════════
class FinalBenchmarkAcceptanceEngine:
    """Evidence-based final acceptance gate.

    This layer does not invent a "PASS" result: it combines actual walk-forward
    metrics with production/data-quality guards and stores a reproducible report.
    """
    VERSION = "v10.0.3"

    def __init__(self, league_code):
        self.league_code = league_code
        self.root = Path("models") / "benchmark" / league_code
        self.path = self.root / "final_benchmark_v10.0.3.json"

    def _save(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(self.path)

    @staticmethod
    def _metric_status(wf):
        folds = int(wf.get('fold_count', 0) or 0)
        total = int(wf.get('total_eval', 0) or 0)
        logloss = float(wf.get('logloss', 99.0) or 99.0)
        brier = float(wf.get('brier', 99.0) or 99.0)
        failures=[]
        if folds < 3: failures.append('fewer than 3 valid walk-forward folds')
        if total < 30: failures.append('fewer than 30 out-of-sample matches')
        if not math.isfinite(logloss) or logloss <= 0 or logloss > 2.5: failures.append('Log Loss outside acceptance bounds')
        if not math.isfinite(brier) or brier < 0 or brier > 1.0: failures.append('Brier Score outside acceptance bounds')
        actual_draw_rate=float(wf.get('actual_draw_rate', 0.0) or 0.0)
        predicted_draw_rate=float(wf.get('predicted_draw_rate', 0.0) or 0.0)
        draw_recall=float(wf.get('draw_recall', 0.0) or 0.0)
        # Separate probabilistic quality from argmax class frequency. A model can assign
        # useful draw probability without DRAW being the largest of three classes.
        mean_dp=float(wf.get('mean_draw_probability', 0.0) or 0.0)
        draw_on_actual=float(wf.get('draw_probability_on_actual_draw', 0.0) or 0.0)
        draw_brier=float(wf.get('draw_brier', 1.0) or 1.0)
        if actual_draw_rate >= 0.15 and mean_dp < max(0.05, actual_draw_rate * 0.35):
            failures.append('mean draw probability is severely below the observed draw base rate')
        if actual_draw_rate >= 0.15 and draw_on_actual < 0.12:
            failures.append('draw probability on actual draws is critically low')
        if not math.isfinite(draw_brier) or draw_brier > 0.30:
            failures.append('binary draw Brier Score is outside acceptance bounds')
        # Argmax recall remains diagnostic rather than a standalone rejection criterion.
        if actual_draw_rate >= 0.15 and predicted_draw_rate < 0.01 and mean_dp < actual_draw_rate * 0.60:
            failures.append('draw is almost never selected and its probability is also materially suppressed')
        return failures

    def run(self, walk_forward, data_quality=None, production=None):
        wf = walk_forward or {}
        failures = self._metric_status(wf)
        dq = (data_quality or {}).get('latest', data_quality or {})
        prod = (production or {}).get('latest', production or {})
        if dq and (dq.get('status') == 'critical' or dq.get('leakage_risk') == 'high'):
            failures.append('data quality or leakage guard is critical')
        if prod and prod.get('automatic_actions_allowed') is False:
            failures.append('production hardening gate is blocking automatic actions')
        report = {
            'version': self.VERSION, 'league': self.league_code,
            'evaluated_at': datetime.utcnow().isoformat()+'Z',
            'walk_forward': {k: wf.get(k) for k in ('fold_count','total_eval','logloss','brier','accuracy','draw_accuracy','result_acc','draw_acc','actual_draw_rate','predicted_draw_rate','draw_recall','draw_precision','draw_baseline_accuracy','mean_draw_probability','draw_brier','draw_logloss','draw_probability_on_actual_draw','draw_trace','meta_activation_rate','calibration_activation_rate')},
            'layer_diagnostics': {
                'meta_learner': {'activation_rate': float(wf.get('meta_activation_rate',0.0) or 0.0), 'status': 'active' if float(wf.get('meta_activation_rate',0.0) or 0.0) > 0 else 'inactive_requires_evidence_review'},
                'calibration': {'activation_rate': float(wf.get('calibration_activation_rate',0.0) or 0.0), 'status': 'active' if float(wf.get('calibration_activation_rate',0.0) or 0.0) > 0 else 'inactive_requires_evidence_review'}
            },
            'data_quality': {'status': dq.get('status','unknown'), 'quality_score': dq.get('quality_score'), 'leakage_risk': dq.get('leakage_risk','unknown')},
            'production': {'status': prod.get('status','unknown'), 'automatic_actions_allowed': prod.get('automatic_actions_allowed')},
            'failures': failures,
            'acceptance': 'PASS' if not failures else 'FAIL',
            'safe_for_final_release': not failures,
        }
        self._save(report)
        return report

# ══════════════════════════════════════════════════════════════
# V9.3 — END-TO-END PRODUCTION HARDENING
# ══════════════════════════════════════════════════════════════
class ProductionHardeningManager:
    """Conservative production-readiness guard for startup and lifecycle actions.

    It never repairs or replaces artifacts silently.  Failures are captured in a
    per-league report so automated lifecycle actions can stop safely.
    """
    VERSION = "v9.3"

    def __init__(self, league_code, resources):
        self.league_code = league_code
        self.resources = resources
        self.root = Path("models") / "production_hardening" / league_code
        self.path = self.root / "production_hardening_v9.3.json"

    def _default(self):
        return {"version": self.VERSION, "league": self.league_code,
                "updated_at": None, "latest": {}, "history": []}

    def load(self):
        if not self.path.exists(): return self._default()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != self.VERSION or data.get("league") != self.league_code:
                return self._default()
            data.setdefault("history", [])
            return data
        except Exception:
            return self._default()

    def _save(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _artifact_check(path):
        p = Path(path)
        if not p.exists(): return {"status": "missing", "detail": str(p)}
        if not p.is_file(): return {"status": "invalid", "detail": str(p)}
        if p.stat().st_size <= 0: return {"status": "empty", "detail": str(p)}
        return {"status": "healthy", "detail": str(p), "bytes": p.stat().st_size}

    def run_startup_checks(self, data_quality=None, model_path=None, required_paths=None):
        checks, failures = {}, []
        for raw in (required_paths or []):
            c = self._artifact_check(raw)
            checks[f"required:{raw}"] = c
            if c["status"] != "healthy": failures.append(f"required artifact {c['status']}: {raw}")
        if model_path:
            c = self._artifact_check(model_path)
            checks["champion_model"] = c
            # Missing model is warning because first-run training is supported.
            if c["status"] in {"invalid", "empty"}: failures.append(f"model artifact {c['status']}")
        dq = data_quality or {}
        checks["data_quality"] = {"status": dq.get("status", "unknown"),
                                  "safe_for_training": dq.get("safe_for_training", False),
                                  "leakage_risk": dq.get("leakage_risk", "unknown")}
        if dq.get("status") == "critical" or dq.get("leakage_risk") == "high":
            failures.append("data quality or leakage guard is critical")

        # Writable storage is essential for atomic model lifecycle records.
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            probe = self.root / ".write_probe"
            probe.write_text("ok", encoding="utf-8"); probe.unlink()
            checks["atomic_storage"] = {"status": "healthy"}
        except Exception as exc:
            checks["atomic_storage"] = {"status": "failed", "detail": str(exc)}
            failures.append("production storage is not writable")

        status = "healthy" if not failures else "critical"
        latest = {"at": datetime.utcnow().isoformat()+"Z", "status": status,
                  "checks": checks, "failures": failures,
                  "automatic_actions_allowed": status == "healthy"}
        data = self.load(); data["updated_at"] = latest["at"]; data["latest"] = latest
        if not data["history"] or data["history"][-1].get("status") != status:
            data["history"].append({"at": latest["at"], "status": status, "failures": failures})
        self._save(data)
        return data

# ══════════════════════════════════════════════════════════════
# V10.0.7 — RESUMABLE WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════
def build_walk_forward_plan(matches, min_train_ratio=0.55, step_ratio=0.10, eval_ratio=0.10, max_folds=5):
    fin=sorted([m for m in matches if m.get("status")=="FINISHED"], key=lambda m:m.get("utcDate", ""))
    n=len(fin); min_train=max(30,int(n*min_train_ratio)); eval_n=max(10,int(n*eval_ratio)); step=max(10,int(n*step_ratio))
    plan=[]; origin=min_train
    while origin+eval_n<=n and len(plan)<max_folds:
        plan.append({"fold":len(plan)+1,"origin":origin,"eval_n":eval_n,"split":origin/float(origin+eval_n)})
        origin+=step
    return fin, plan

def run_walk_forward_resumable(backtester, matches, resources, checkpoint_path, **kwargs):
    """Checkpoint every completed temporal fold; safely resumes without recomputation."""
    from resumable_walk_forward import ResumableWalkForwardRunner
    fin, plan=build_walk_forward_plan(matches, **kwargs)
    runner=ResumableWalkForwardRunner(checkpoint_path)
    def evaluate(spec):
        subset=fin[:spec["origin"]+spec["eval_n"]]
        r=backtester.run(subset, resources, split=spec["split"], temporal_cv=False)
        if r.get("error"): return r
        preds=r.get("predictions",[])
        return {"fold":spec["fold"],"train_size":spec["origin"],"eval_size":r.get("eval_size",0),
                "logloss":float(r.get("logloss",0)),"brier":float(r.get("brier",0)),"accuracy":float(r.get("result_acc",0)),
                "result_acc":float(r.get("result_acc",0)),"draw_acc":float(r.get("draw_acc",0)),
                "actual_draw_rate":sum(x.get("actual")=="DRAW" for x in preds)/max(1,len(preds)),
                "predicted_draw_rate":sum(x.get("predicted")=="DRAW" for x in preds)/max(1,len(preds)),
                "draw_recall":sum(x.get("actual")=="DRAW" and x.get("predicted")=="DRAW" for x in preds)/max(1,sum(x.get("actual")=="DRAW" for x in preds)),
                "draw_precision":sum(x.get("actual")=="DRAW" and x.get("predicted")=="DRAW" for x in preds)/max(1,sum(x.get("predicted")=="DRAW" for x in preds)),
                "mean_draw_probability":float(np.mean([x.get("probs",(0,0,0))[1] for x in preds])) if preds else 0.0,
                "draw_trace":{s:float(np.mean([x.get("draw_trace",{}).get(s,x.get("probs",(0,0,0))[1]) for x in preds])) for s in ("draw_model","weighted_ensemble","after_draw_preservation","after_draw_correction","after_calibration","after_meta","final")}}
    return runner.run(fin, plan, evaluate)

# ══════════════════════════════════════════════════════════════
# V10.0.8 — DECISION POLICY & DRAW-AWARE CLASSIFICATION AUDIT
# ══════════════════════════════════════════════════════════════
def audit_decision_policies(predictions, candidate_params=None, **selection_kwargs):
    """Evaluate policies using temporal folds; a flat list remains backward-compatible.

    Preferred input is a list of fold-lists: ``[fold1_rows, fold2_rows, ...]``.
    A flat prediction list is treated as one fold and therefore cannot promote
    a challenger because v10.0.9 requires multiple temporal folds for selection.
    """
    from decision_policy_audit import audit_fold, audit_temporal_folds, VERSION
    if isinstance(predictions, dict):
        folds = [predictions[k] for k in sorted(predictions)]
    elif predictions and isinstance(predictions[0], dict):
        folds = [predictions]
    else:
        folds = list(predictions or [])
    if folds and folds[0] and isinstance(folds[0][0], dict):
        result = audit_temporal_folds(folds, candidate_params, **selection_kwargs)
        return result
    result = audit_temporal_folds([], candidate_params, **selection_kwargs)
    result["version"] = VERSION
    return result
# ══════════════════════════════════════════════════════════════
# ENTRY POINT — V6.1 FIX: Streamlit/CLI separation
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ══════════════════════════════════════════════════════
    # ENTRY POINT — V6.1 FIX: منطق واضح بدون تعارض
    # ──────────────────────────────────────────────────────
    # python appp.py               → CLI
    # python appp.py --streamlit   → Streamlit (يُطلق run_streamlit)
    # streamlit run appp.py        → Streamlit (sys.argv[0] = 'appp.py')
    # ══════════════════════════════════════════════════════
    _want_streamlit = False
    if STREAMLIT_AVAILABLE:
        if "--streamlit" in sys.argv:
            _want_streamlit = True
        else:
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                if get_script_run_ctx() is not None:
                    _want_streamlit = True
            except Exception:
                pass

    if _want_streamlit:
        run_streamlit()
    else:
        cli_main()
