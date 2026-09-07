#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   🌍 FOOTBALL DATA DOWNLOADER v2.0 (MULTI-LEAGUE EDITION)           ║
║                                                                        ║
║   ✅ Premier League (England)     → E0_Master.csv                    ║
║   ✅ La Liga (Spain)              → SP1_Master.csv                   ║
║   ✅ Serie A (Italy)              → I1_Master.csv                    ║
║   ✅ Bundesliga (Germany)         → D1_Master.csv                    ║
║   ✅ Ligue 1 (France)             → F1_Master.csv                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ══════════════════════════════════════════════════════════════
# إعدادات الدوريات
# ══════════════════════════════════════════════════════════════

LEAGUES = {
    "E0": {
        "name": "Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
        "country": "England",
        "source_url": "https://www.football-data.co.uk/englandm.php",
        "pattern": re.compile(r'mmz4281/(\d{4})/E0\.csv', re.IGNORECASE),
        "output_file": "data/PL_Master.csv",
        "download_dir": "historical_data/england",
        "file_prefix": "E0",
        "config_data_files": ["data/PL_Master.csv"],
    },
    "SP1": {
        "name": "La Liga 🇪🇸",
        "country": "Spain",
        "source_url": "https://www.football-data.co.uk/spainm.php",
        "pattern": re.compile(r'mmz4281/(\d{4})/SP1\.csv', re.IGNORECASE),
        "output_file": "data/LL_Master.csv",
        "download_dir": "historical_data/spain",
        "file_prefix": "SP1",
        "config_data_files": ["data/LL_Master.csv"],
    },
    "I1": {
        "name": "Serie A 🇮🇹",
        "country": "Italy",
        "source_url": "https://www.football-data.co.uk/italym.php",
        "pattern": re.compile(r'mmz4281/(\d{4})/I1\.csv', re.IGNORECASE),
        "output_file": "data/SA_Master.csv",
        "download_dir": "historical_data/italy",
        "file_prefix": "I1",
        "config_data_files": ["data/SA_Master.csv"],
    },
    "D1": {
        "name": "Bundesliga 🇩🇪",
        "country": "Germany",
        "source_url": "https://www.football-data.co.uk/germanym.php",
        "pattern": re.compile(r'mmz4281/(\d{4})/D1\.csv', re.IGNORECASE),
        "output_file": "data/BL_Master.csv",
        "download_dir": "historical_data/germany",
        "file_prefix": "D1",
        "config_data_files": ["data/BL_Master.csv"],
    },
    "F1": {
        "name": "Ligue 1 🇫🇷",
        "country": "France",
        "source_url": "https://www.football-data.co.uk/francem.php",
        "pattern": re.compile(r'mmz4281/(\d{4})/F1\.csv', re.IGNORECASE),
        "output_file": "data/L1_Master.csv",
        "download_dir": "historical_data/france",
        "file_prefix": "F1",
        "config_data_files": ["data/L1_Master.csv"],
    },
}

# ══════════════════════════════════════════════════════════════
# إعدادات عامة
# ══════════════════════════════════════════════════════════════

BASE_URL       = "https://www.football-data.co.uk/"
MIN_SEASON_YEAR = 2014   # تجاهل ما قبل 2014/15
REQUEST_DELAY  = 0.8   # ثانية بين كل طلب
REQUEST_TIMEOUT = 30   # ثانية

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

# الأعمدة المطلوبة من كل ملف CSV
IMPORTANT_COLS = [
    'Date', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR',
    'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HF', 'AF',
    'HY', 'AY', 'HR', 'AR'
]

# ══════════════════════════════════════════════════════════════
# إعداد المجلدات
# ══════════════════════════════════════════════════════════════

def setup_directories():
    """إنشاء جميع المجلدات المطلوبة"""
    dirs = ["data", "historical_data", "config", "models"]
    for league_cfg in LEAGUES.values():
        dirs.append(league_cfg["download_dir"])
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("📁 تم إعداد مجلدات المشروع ✅")

# ══════════════════════════════════════════════════════════════
# أدوات مساعدة
# ══════════════════════════════════════════════════════════════

def season_label(folder: str) -> str:
    """تحويل '2324' إلى '23/24' للعرض"""
    if len(folder) == 4:
        return f"{folder[:2]}/{folder[2:]}"
    return folder

def is_current_season(folder: str) -> bool:
    now = datetime.now()
    start_year = now.year if now.month >= 7 else now.year - 1
    return folder == f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"

def is_season_valid(folder: str) -> bool:
    """التحقق من أن الموسم ضمن النطاق المطلوب"""
    try:
        start_year_2d = int(folder[:2])
        # تحويل السنة إلى 4 أرقام لتجنب مشاكل الألفية والتسعينات
        if start_year_2d >= 80:
            full_year = 1900 + start_year_2d
        else:
            full_year = 2000 + start_year_2d
            
        return full_year >= MIN_SEASON_YEAR
    except (ValueError, IndexError):
        return False

def smart_read_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    قارئ ذكي يجرب ترميزات متعددة
    ويُنظف أسماء الأعمدة تلقائياً
    """
    encodings = ['utf-8', 'windows-1252', 'latin1', 'unicode_escape']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            # تنظيف المسافات في أسماء الأعمدة
            df.columns = df.columns.str.strip()
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    print(f"    ⚠️ فشل قراءة: {file_path}")
    return None

def safe_request(url: str, retries: int = 3) -> Optional[requests.Response]:
    """طلب HTTP مع إعادة المحاولة عند الفشل"""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            print(f"    ⏱️ انتهت المهلة (محاولة {attempt}/{retries})")
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 429 or (code is not None and code >= 500):
                wait = min(30, 2 ** attempt)
                print(f"    ⏳ HTTP {code} — إعادة المحاولة بعد {wait}s")
                if attempt < retries:
                    time.sleep(wait)
                    continue
            print(f"    ❌ HTTP Error: {e}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"    🔌 خطأ في الاتصال (محاولة {attempt}/{retries})")
        if attempt < retries:
            time.sleep(2 * attempt)
    return None

# ══════════════════════════════════════════════════════════════
# تحميل دوري واحد
# ══════════════════════════════════════════════════════════════

def download_league(league_code: str) -> List[str]:
    """
    تحميل جميع مواسم دوري واحد
    يعيد قائمة بمسارات الملفات المُحمَّلة
    """
    cfg = LEAGUES[league_code]
    print(f"\n{'═'*60}")
    print(f"  🏆 {cfg['name']} ({cfg['country']})")
    print(f"{'═'*60}")
    print(f"  🌐 المصدر: {cfg['source_url']}")

    # جلب صفحة الروابط
    resp = safe_request(cfg["source_url"])
    if not resp:
        print(f"  ❌ فشل الاتصال بصفحة {cfg['name']}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    links = soup.find_all('a')

    csv_files: List[str] = []
    download_count = 0
    skip_count = 0
    fail_count = 0

    for link in links:
        href = link.get('href', '')
        if not href:
            continue

        match = cfg["pattern"].search(href)
        if not match:
            continue

        season_folder = match.group(1)   # مثال: '2324'

        # تصفية المواسم القديمة
        if not is_season_valid(season_folder):
            continue

        full_url  = urljoin(BASE_URL, href)
        file_name = f"{cfg['file_prefix']}_{season_folder}.csv"
        file_path = os.path.join(cfg["download_dir"], file_name)

        csv_files.append(file_path)

        # لا نتخطى الموسم الجاري: قد تتغير النتائج يومياً.
        if os.path.exists(file_path) and os.path.getsize(file_path) > 100 and not is_current_season(season_folder):
            skip_count += 1
            print(f"  ⏭️  موسم {season_label(season_folder)} → موجود مسبقاً")
            continue

        print(f"  📥 تحميل موسم {season_label(season_folder)} ...", end=" ", flush=True)
        csv_resp = safe_request(full_url)

        if csv_resp and csv_resp.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(csv_resp.content)
            size_kb = len(csv_resp.content) / 1024
            print(f"✅ ({size_kb:.1f} KB)")
            download_count += 1
            time.sleep(REQUEST_DELAY)
        else:
            print(f"❌ فشل")
            fail_count += 1

    print(f"\n  📊 النتيجة: ✅ جديد={download_count} | ⏭️ موجود={skip_count} | ❌ فشل={fail_count}")
    return csv_files

# ══════════════════════════════════════════════════════════════
# دمج ملفات دوري واحد
# ══════════════════════════════════════════════════════════════

def merge_league_files(
    file_list: List[str],
    output_path: str,
    league_code: str
) -> Optional[pd.DataFrame]:
    """
    دمج جميع مواسم دوري واحد في ملف Master واحد
    يعيد الـ DataFrame النهائي
    """
    cfg = LEAGUES[league_code]
    print(f"\n  🔄 دمج مواسم {cfg['name']}...")

    all_dfs: List[pd.DataFrame] = []

    for file_path in sorted(file_list):
        if not os.path.exists(file_path):
            continue

        df = smart_read_csv(file_path)
        if df is None:
            continue

        # اختيار الأعمدة المتاحة فقط
        available = [c for c in IMPORTANT_COLS if c in df.columns]
        if len(available) < 5:
            print(f"    ⚠️ أعمدة غير كافية في: {Path(file_path).name}")
            continue

        df_clean = df[available].copy()

        # إزالة الصفوف الفارغة
        df_clean = df_clean.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG'])
        df_clean = df_clean[df_clean['HomeTeam'].astype(str).str.strip() != '']

        if not df_clean.empty:
            # إضافة عمود المصدر لتتبع الموسم
            df_clean['_source_file'] = Path(file_path).name
            all_dfs.append(df_clean)
            season = Path(file_path).stem.split('_')[-1]
            print(f"    ✅ {Path(file_path).name}: {len(df_clean)} مباراة")

    if not all_dfs:
        print(f"    ❌ لا توجد بيانات صالحة لـ {cfg['name']}")
        return None

    master_df = pd.concat(all_dfs, ignore_index=True)

    # إزالة عمود المصدر قبل الحفظ
    if '_source_file' in master_df.columns:
        master_df = master_df.drop(columns=['_source_file'])

    # إزالة التكرار بهوية مستقرة حتى لو اختلفت كتابة الاسم بين المصادر.
    before = len(master_df)
    master_df['_dedup_date'] = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
    master_df['_dedup_home'] = master_df['HomeTeam'].astype(str).str.strip().str.casefold()
    master_df['_dedup_away'] = master_df['AwayTeam'].astype(str).str.strip().str.casefold()
    master_df = master_df.drop_duplicates(subset=['_dedup_date','_dedup_home','_dedup_away'], keep='last')
    master_df = master_df.drop(columns=['_dedup_date','_dedup_home','_dedup_away'])
    after = len(master_df)
    if before != after:
        print(f"    🧹 حُذف {before - after} تكرار")

    # حفظ الملف
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"    💾 تم الحفظ: {output_path} ({len(master_df)} مباراة)")

    return master_df

# ══════════════════════════════════════════════════════════════
# تقرير الإحصائيات
# ══════════════════════════════════════════════════════════════

def print_league_stats(df: pd.DataFrame, league_code: str):
    """طباعة إحصائيات سريعة عن قاعدة البيانات"""
    cfg = LEAGUES[league_code]
    print(f"\n  📈 إحصائيات {cfg['name']}:")
    print(f"     • إجمالي المباريات : {len(df):,}")

    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dropna()
        if not dates.empty:
            print(f"     • من تاريخ       : {dates.min().strftime('%d/%m/%Y')}")
            print(f"     • إلى تاريخ      : {dates.max().strftime('%d/%m/%Y')}")

    if 'HomeTeam' in df.columns:
        n_teams = df['HomeTeam'].nunique()
        print(f"     • عدد الفرق       : {n_teams}")

    # نسبة التوفر لكل عمود إحصائي مهم
    stat_cols = ['HST', 'AST', 'HC', 'AC', 'HY', 'AY']
    available_stats = [c for c in stat_cols if c in df.columns]
    if available_stats:
        coverage = df[available_stats].notna().mean().mean() * 100
        print(f"     • تغطية الإحصائيات: {coverage:.1f}%")

# ══════════════════════════════════════════════════════════════
# تحديث leagues_config.json
# ══════════════════════════════════════════════════════════════

def update_leagues_config(results: Dict[str, Optional[pd.DataFrame]]):
    """
    تحديث ملف leagues_config.json بعد اكتمال التحميل
    يُضيف أو يُحدِّث إعدادات كل دوري
    """
    import json

    config_file = "leagues_config.json"

    # API codes لكل دوري (football-data.org)
    api_codes = {
        "E0":  {"api_code": "PL",  "sport": "soccer_epl",  "ha": 65, "avg_h": 1.53, "avg_a": 1.16},
        "SP1": {"api_code": "PD",  "sport": "soccer_spain_la_liga", "ha": 60, "avg_h": 1.48, "avg_a": 1.12},
        "I1":  {"api_code": "SA",  "sport": "soccer_italy_serie_a", "ha": 58, "avg_h": 1.45, "avg_a": 1.10},
        "D1":  {"api_code": "BL1", "sport": "soccer_germany_bundesliga", "ha": 62, "avg_h": 1.68, "avg_a": 1.30},
        "F1":  {"api_code": "FL1", "sport": "soccer_france_ligue_one", "ha": 57, "avg_h": 1.40, "avg_a": 1.08},
    }

    # تحميل الملف الموجود أو بناء هيكل جديد
    if Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            "leagues": {},
            "global_settings": {
                "elo_init": 1500,
                "elo_k": 32,
                "form_n": 8,
                "backtest_split": 0.70,
                "min_samples_per_class": 10,
                "n_features": 126
            }
        }

    # اسم الدوري المختصر لكل كود
    league_names = {
        "E0":  ("PL",  "Premier League",  "England"),
        "SP1": ("LL",  "La Liga",         "Spain"),
        "I1":  ("SA",  "Serie A",         "Italy"),
        "D1":  ("BL",  "Bundesliga",      "Germany"),
        "F1":  ("L1",  "Ligue 1",         "France"),
    }

    updated = []
    for league_code, df in results.items():
        if df is None:
            continue

        short_code, full_name, country = league_names[league_code]
        api_info = api_codes[league_code]
        league_cfg = LEAGUES[league_code]

        config["leagues"][short_code] = {
            "name": full_name,
            "country": country,
            "api_code": api_info["api_code"],
            "api_url": "https://api.football-data.org/v4",
            "data_files": league_cfg["config_data_files"],
            "model_file": f"models/{short_code}_model_v7.pkl",
            "calibration_file": f"models/{short_code}_calibration_v7.pkl",
            "elo_file": f"models/{short_code}_elo_v7.pkl",
            "teams_map_file": f"config/{short_code}_teams_map.json",
            "aliases_file": f"config/{short_code}_aliases.json",
            "rivalries_file": f"config/{short_code}_rivalries.json",
            "home_advantage": api_info["ha"],
            "avg_home_goals": api_info["avg_h"],
            "avg_away_goals": api_info["avg_a"]
        }
        updated.append(full_name)

    # حفظ الملف المحدَّث
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ تم تحديث {config_file}")
    print(f"   الدوريات المُضافة/المُحدَّثة: {', '.join(updated)}")

# ══════════════════════════════════════════════════════════════
# إنشاء ملفات config الأساسية (aliases + rivalries)
# ══════════════════════════════════════════════════════════════

def create_league_configs():
    """إنشاء ملفات aliases.json و rivalries.json لكل دوري"""
    import json

    # ─── Aliases ─────────────────────────────────────────────
    aliases_data = {

        # ── PL ──
        "PL": {
            "manchester united": "Man United",
            "manchester city": "Man City",
            "tottenham hotspur": "Tottenham",
            "tottenham": "Tottenham",
            "newcastle united": "Newcastle",
            "west ham united": "West Ham",
            "wolverhampton wanderers": "Wolves",
            "wolverhampton": "Wolves",
            "nottingham forest": "Nottm Forest",
            "leicester city": "Leicester",
            "brighton & hove albion": "Brighton",
            "brighton": "Brighton",
            "crystal palace": "Crystal Palace",
            "aston villa": "Aston Villa",
            "arsenal": "Arsenal",
            "chelsea": "Chelsea",
            "liverpool": "Liverpool",
            "everton": "Everton",
            "brentford": "Brentford",
            "fulham": "Fulham",
            "bournemouth": "Bournemouth",
            "luton town": "Luton",
            "sheffield united": "Sheffield Utd",
            "burnley": "Burnley",
            "ipswich town": "Ipswich",
            "southampton": "Southampton",
            "sunderland": "Sunderland",
        },

        # ── LL (La Liga) ──
        "LL": {
            "real madrid": "Real Madrid",
            "real madrid cf": "Real Madrid",
            "fc barcelona": "Barcelona",
            "barcelona": "Barcelona",
            "atletico madrid": "Atletico Madrid",
            "club atletico de madrid": "Atletico Madrid",
            "athletic bilbao": "Athletic Club",
            "athletic club": "Athletic Club",
            "real sociedad": "Real Sociedad",
            "real betis": "Real Betis",
            "real betis balompie": "Real Betis",
            "villarreal cf": "Villarreal",
            "villarreal": "Villarreal",
            "sevilla fc": "Sevilla",
            "sevilla": "Sevilla",
            "valencia cf": "Valencia",
            "valencia": "Valencia",
            "rayo vallecano": "Rayo Vallecano",
            "getafe cf": "Getafe",
            "getafe": "Getafe",
            "celta vigo": "Celta Vigo",
            "rc celta": "Celta Vigo",
            "osasuna": "Osasuna",
            "ca osasuna": "Osasuna",
            "girona fc": "Girona",
            "girona": "Girona",
            "alaves": "Alaves",
            "deportivo alaves": "Alaves",
            "real valladolid": "Valladolid",
            "espanyol": "Espanyol",
            "rcd espanyol": "Espanyol",
            "las palmas": "Las Palmas",
            "ud las palmas": "Las Palmas",
            "mallorca": "Mallorca",
            "rcd mallorca": "Mallorca",
            "leganes": "Leganes",
            "cd leganes": "Leganes",
        },

        # ── SA (Serie A) ──
        "SA": {
            "inter milan": "Inter",
            "fc internazionale milano": "Inter",
            "internazionale": "Inter",
            "ac milan": "Milan",
            "juventus fc": "Juventus",
            "juventus": "Juventus",
            "ssc napoli": "Napoli",
            "napoli": "Napoli",
            "as roma": "Roma",
            "roma": "Roma",
            "ss lazio": "Lazio",
            "lazio": "Lazio",
            "atalanta bc": "Atalanta",
            "atalanta": "Atalanta",
            "acf fiorentina": "Fiorentina",
            "fiorentina": "Fiorentina",
            "torino fc": "Torino",
            "torino": "Torino",
            "bologna fc": "Bologna",
            "bologna": "Bologna",
            "udinese calcio": "Udinese",
            "udinese": "Udinese",
            "us sassuolo": "Sassuolo",
            "sassuolo": "Sassuolo",
            "hellas verona": "Verona",
            "verona": "Verona",
            "cagliari calcio": "Cagliari",
            "cagliari": "Cagliari",
            "genoa cfc": "Genoa",
            "genoa": "Genoa",
            "venezia fc": "Venezia",
            "venezia": "Venezia",
            "empoli fc": "Empoli",
            "empoli": "Empoli",
            "como 1907": "Como",
            "como": "Como",
            "parma calcio": "Parma",
            "parma": "Parma",
            "monza": "Monza",
            "ac monza": "Monza",
            "lecce": "Lecce",
            "us lecce": "Lecce",
        },

        # ── BL (Bundesliga) ──
        "BL": {
            "fc bayern münchen": "Bayern Munich",
            "fc bayern munich": "Bayern Munich",
            "bayern munich": "Bayern Munich",
            "borussia dortmund": "Dortmund",
            "bvb": "Dortmund",
            "bayer 04 leverkusen": "Leverkusen",
            "bayer leverkusen": "Leverkusen",
            "rb leipzig": "RB Leipzig",
            "rasenballsport leipzig": "RB Leipzig",
            "borussia mönchengladbach": "M'gladbach",
            "borussia monchengladbach": "M'gladbach",
            "eintracht frankfurt": "Frankfurt",
            "vfb stuttgart": "Stuttgart",
            "stuttgarter kickers": "Stuttgart",
            "sc freiburg": "Freiburg",
            "freiburg": "Freiburg",
            "union berlin": "Union Berlin",
            "1. fc union berlin": "Union Berlin",
            "vfl wolfsburg": "Wolfsburg",
            "wolfsburg": "Wolfsburg",
            "tsg hoffenheim": "Hoffenheim",
            "hoffenheim": "Hoffenheim",
            "sv werder bremen": "Werder Bremen",
            "werder bremen": "Werder Bremen",
            "1. fc heidenheim": "Heidenheim",
            "heidenheim": "Heidenheim",
            "fc augsburg": "Augsburg",
            "augsburg": "Augsburg",
            "vfl bochum": "Bochum",
            "bochum": "Bochum",
            "fc st. pauli": "St. Pauli",
            "holstein kiel": "Holstein Kiel",
            "1. fsv mainz 05": "Mainz",
            "mainz 05": "Mainz",
            "mainz": "Mainz",
            "hamburger sv": "Hamburg",
            "hamburg": "Hamburg",
        },

        # ── L1 (Ligue 1) ──
        "L1": {
            "paris saint-germain": "PSG",
            "paris sg": "PSG",
            "psg": "PSG",
            "olympique de marseille": "Marseille",
            "marseille": "Marseille",
            "olympique lyonnais": "Lyon",
            "lyon": "Lyon",
            "as monaco": "Monaco",
            "monaco": "Monaco",
            "losc lille": "Lille",
            "lille": "Lille",
            "stade rennais fc": "Rennes",
            "rennes": "Rennes",
            "rc lens": "Lens",
            "lens": "Lens",
            "ogc nice": "Nice",
            "nice": "Nice",
            "stade de reims": "Reims",
            "reims": "Reims",
            "montpellier hsc": "Montpellier",
            "montpellier": "Montpellier",
            "rc strasbourg alsace": "Strasbourg",
            "strasbourg": "Strasbourg",
            "toulouse fc": "Toulouse",
            "toulouse": "Toulouse",
            "nantes": "Nantes",
            "fc nantes": "Nantes",
            "girondins de bordeaux": "Bordeaux",
            "bordeaux": "Bordeaux",
            "stade brestois 29": "Brest",
            "brest": "Brest",
            "le havre ac": "Le Havre",
            "le havre": "Le Havre",
            "angers sco": "Angers",
            "angers": "Angers",
            "clermont foot": "Clermont",
            "clermont": "Clermont",
            "metz": "Metz",
            "fc metz": "Metz",
            "auxerre": "Auxerre",
            "aj auxerre": "Auxerre",
            "saint-etienne": "St Etienne",
            "as saint-etienne": "St Etienne",
        },
    }

    # ─── Rivalries ───────────────────────────────────────────
    rivalries_data = {

        "PL": [
            {"teams": ["Arsenal", "Tottenham"],     "name": "North London Derby"},
            {"teams": ["Liverpool", "Everton"],     "name": "Merseyside Derby"},
            {"teams": ["Man United", "Man City"],   "name": "Manchester Derby"},
            {"teams": ["Man United", "Liverpool"],  "name": "Northwest Derby"},
            {"teams": ["Chelsea", "Arsenal"],       "name": "London Derby"},
            {"teams": ["Chelsea", "Tottenham"],     "name": "London Derby"},
            {"teams": ["West Ham", "Tottenham"],    "name": "London Derby"},
            {"teams": ["Crystal Palace", "Brighton"], "name": "M23 Derby"},
            {"teams": ["Nottm Forest", "Leicester"], "name": "East Midlands Derby"},
            {"teams": ["Wolves", "Aston Villa"],    "name": "West Midlands Derby"},
        ],

        "LL": [
            {"teams": ["Real Madrid", "Barcelona"],       "name": "El Clásico"},
            {"teams": ["Real Madrid", "Atletico Madrid"], "name": "Madrid Derby"},
            {"teams": ["Barcelona", "Espanyol"],          "name": "Derbi Barceloní"},
            {"teams": ["Barcelona", "Atletico Madrid"],   "name": "La Liga Top Clash"},
            {"teams": ["Sevilla", "Real Betis"],          "name": "Derbi Sevillano"},
            {"teams": ["Athletic Club", "Real Sociedad"], "name": "Basque Derby"},
            {"teams": ["Valencia", "Villarreal"],         "name": "Derbi de la Comunitat"},
        ],

        "SA": [
            {"teams": ["Inter", "Milan"],           "name": "Derby della Madonnina"},
            {"teams": ["Inter", "Juventus"],        "name": "Derby d'Italia"},
            {"teams": ["Roma", "Lazio"],             "name": "Derby della Capitale"},
            {"teams": ["Juventus", "Torino"],       "name": "Derby della Mole"},
            {"teams": ["Napoli", "Roma"],            "name": "South vs Capital Derby"},
            {"teams": ["Fiorentina", "Juventus"],   "name": "Derby d'Italia"},
            {"teams": ["Genoa", "Sampdoria"],       "name": "Derby della Lanterna"},
        ],

        "BL": [
            {"teams": ["Bayern Munich", "Dortmund"], "name": "Der Klassiker"},
            {"teams": ["Bayern Munich", "M'gladbach"], "name": "Bayern vs Gladbach"},
            {"teams": ["Dortmund", "Schalke"],      "name": "Revierderby"},
            {"teams": ["Hamburg", "Werder Bremen"], "name": "Nordderby"},
            {"teams": ["Frankfurt", "Darmstadt"],   "name": "Hessenderby"},
            {"teams": ["Stuttgart", "Frankfurt"],   "name": "Südwestderby"},
            {"teams": ["Union Berlin", "Hertha"],   "name": "Berlin Derby"},
        ],

        "L1": [
            {"teams": ["PSG", "Marseille"],         "name": "Le Classique"},
            {"teams": ["PSG", "Lyon"],              "name": "Paris vs Lyon"},
            {"teams": ["Marseille", "Lyon"],        "name": "Olympico"},
            {"teams": ["Marseille", "Nice"],        "name": "Côte d'Azur Derby"},
            {"teams": ["Lyon", "Saint-Etienne"],   "name": "Derby du Rhône"},
            {"teams": ["Lens", "Lille"],            "name": "Derby du Nord"},
            {"teams": ["Monaco", "Nice"],           "name": "Côte d'Azur Derby"},
        ],
    }

    # خريطة short_code → اسم الملف
    file_map = {
        "PL": "PL", "LL": "LL",
        "SA": "SA", "BL": "BL", "L1": "L1"
    }

    for short_code in file_map:
        # حفظ aliases
        alias_file = f"config/{short_code}_aliases.json"
        if not Path(alias_file).exists():
            with open(alias_file, 'w', encoding='utf-8') as f:
                json.dump(aliases_data.get(short_code, {}), f,
                          indent=2, ensure_ascii=False)
            print(f"  📄 {alias_file} ← أُنشئ")

        # حفظ rivalries
        rivalry_file = f"config/{short_code}_rivalries.json"
        if not Path(rivalry_file).exists():
            with open(rivalry_file, 'w', encoding='utf-8') as f:
                json.dump(rivalries_data.get(short_code, []), f,
                          indent=2, ensure_ascii=False)
            print(f"  📄 {rivalry_file} ← أُنشئ")

        # teams_map فارغ (يُملأ يدوياً أو لاحقاً)
        map_file = f"config/{short_code}_teams_map.json"
        if not Path(map_file).exists():
            with open(map_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            print(f"  📄 {map_file} ← أُنشئ (فارغ)")

# ══════════════════════════════════════════════════════════════
# التقرير النهائي
# ══════════════════════════════════════════════════════════════

def print_final_report(results: Dict[str, Optional[pd.DataFrame]]):
    print(f"\n{'═'*60}")
    print("  📋 التقرير النهائي")
    print(f"{'═'*60}")

    total_matches = 0
    success = []
    failed  = []

    for code, df in results.items():
        cfg = LEAGUES[code]
        if df is not None:
            n = len(df)
            total_matches += n
            success.append((cfg['name'], n, cfg['output_file']))
        else:
            failed.append(cfg['name'])

    for name, n, out in success:
        print(f"  ✅ {name:<25} {n:>6,} مباراة → {out}")

    for name in failed:
        print(f"  ❌ {name:<25} فشل التحميل")

    print(f"\n  🌟 الإجمالي: {total_matches:,} مباراة في {len(success)} دوري")
    print(f"{'═'*60}\n")

# ══════════════════════════════════════════════════════════════
# نقطة الدخول الرئيسية
# ══════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🌍 FOOTBALL DATA DOWNLOADER v2.0 — Multi-League Edition    ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 1. إعداد المجلدات
    setup_directories()

    # 2. إنشاء ملفات config
    print("\n📁 إنشاء ملفات الإعداد...")
    create_league_configs()

    # 3. اختيار الدوريات
    print("\n" + "═"*60)
    print("  اختر الدوريات للتحميل:")
    print("  0 - جميع الدوريات (الكل)")
    for i, (code, cfg) in enumerate(LEAGUES.items(), 1):
        print(f"  {i} - {cfg['name']}")
    print("  أو أدخل أرقام متعددة مفصولة بفاصلة (مثل: 1,3,5)")
    print("═"*60)

    choice = input("\n  اختيارك (0 = الكل): ").strip()

    if choice == "0" or choice == "":
        selected_codes = list(LEAGUES.keys())
    else:
        codes_list = list(LEAGUES.keys())
        selected_codes = []
        for part in choice.split(","):
            part = part.strip()
            try:
                idx = int(part) - 1
                if 0 <= idx < len(codes_list):
                    selected_codes.append(codes_list[idx])
            except ValueError:
                # قبول الكود مباشرة مثل "E0"
                up = part.upper()
                if up in LEAGUES:
                    selected_codes.append(up)

    if not selected_codes:
        print("❌ لم يتم اختيار أي دوري")
        return

    print(f"\n  ✅ الدوريات المختارة: {', '.join(LEAGUES[c]['name'] for c in selected_codes)}\n")

    # 4. التحميل والدمج
    results: Dict[str, Optional[pd.DataFrame]] = {}

    for league_code in selected_codes:
        # تحميل الملفات
        files = download_league(league_code)

        # دمج الملفات
        cfg = LEAGUES[league_code]
        df = None
        if files:
            df = merge_league_files(files, cfg["output_file"], league_code)
            if df is not None:
                print_league_stats(df, league_code)

        results[league_code] = df

    # 5. تحديث leagues_config.json
    update_leagues_config(results)

    # 6. التقرير النهائي
    print_final_report(results)

    print("🚀 جاهز للتدريب! شغّل football_predictor.py الآن.\n")


if __name__ == "__main__":
    main()