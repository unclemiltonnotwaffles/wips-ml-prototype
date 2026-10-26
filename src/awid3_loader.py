"""
awid3_loader.py — reads AWID3 CSVs and returns the SAME schema as
`synthetic.generate_traffic()`.

WHY this file exists:
    AWID3 CSVs have Wireshark-style column names that differ between releases
    (`wlan.sa` vs `wlan.ta`, `attack` vs `class`). Rather than scatter that
    fragility everywhere, we isolate column-name guessing here and expose one
    clean function: `load_awid3_dir`.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

COLUMN_ALIASES = {
    "timestamp": ["frame.time_epoch", "timestamp", "time", "frame.time"],
    "src":       ["wlan.sa", "wlan.ta", "src", "source", "source_address"],
    "dst":       ["wlan.da", "wlan.ra", "dst", "dest", "destination_address"],
    "fc_type":   ["wlan.fc.type", "frame.type", "type"],
    "fc_subtype":["wlan.fc.subtype", "frame.subtype", "subtype"],
    "rssi":      ["radiotap.dbm_antsignal", "rssi", "signal", "dbm"],
    "length":    ["data.len", "frame.len", "length"],
    "label":     ["attack", "class", "label", "is_attack"],
}

# Some traces pack type and subtype into one column (type*16 + subtype).
TYPE_SUBTYPE_ALIASES = ["wlan.fc.type_subtype", "frame.type_subtype"]

def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None

def _normalise_label(series: pd.Series) -> pd.Series:
    """Map AWID3's label conventions (numeric / 'normal' / 'deauth' / ...) to 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    return (~s.isin(["normal", "0", "false", "benign"])).astype(int)

def load_awid3_dir(data_dir: str | Path) -> pd.DataFrame:
    """Load every *.csv inside `data_dir` and concatenate into one frame."""
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSVs found under {data_dir}. "
            "Point --data-dir at the folder you extracted from AWID3."
        )
    frames = []
    for csv in csvs:
        df = pd.read_csv(csv, low_memory=False)
        frames.append(_normalise_one(df, csv.name))

    out = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"[awid3_loader] loaded {len(out):,} frames from {len(csvs)} file(s)")
    print(f"[awid3_loader] attack frames: {int(out['label'].sum()):,} "
          f"({out['label'].mean()*100:.2f} %)")
    return out

def _normalise_one(df: pd.DataFrame, name: str) -> pd.DataFrame:
    ts_col = _find_col(df, TYPE_SUBTYPE_ALIASES)
    type_col = _find_col(df, COLUMN_ALIASES["fc_type"])
    sub_col = _find_col(df, COLUMN_ALIASES["fc_subtype"])

    out = pd.DataFrame()

    t_col = _find_col(df, COLUMN_ALIASES["timestamp"])
    if t_col is None:
        raise KeyError(f"{name}: no timestamp column among "
                       f"{COLUMN_ALIASES['timestamp']}. Columns: {list(df.columns)}")
    out["timestamp"] = pd.to_numeric(df[t_col], errors="coerce")

    for key in ("src", "dst"):
        c = _find_col(df, COLUMN_ALIASES[key])
        if c is None:
            raise KeyError(f"{name}: no {key} column among {COLUMN_ALIASES[key]}")
        out[key] = df[c].astype(str)

    if ts_col is not None:
        packed = pd.to_numeric(df[ts_col], errors="coerce").fillna(0).astype(int)
        out["fc_type"] = packed // 16
        out["fc_subtype"] = packed % 16
    else:
        if type_col is None or sub_col is None:
            raise KeyError(f"{name}: need wlan.fc.type_subtype or both wlan.fc.type/subtype")
        out["fc_type"] = pd.to_numeric(df[type_col], errors="coerce").fillna(0).astype(int)
        out["fc_subtype"] = pd.to_numeric(df[sub_col], errors="coerce").fillna(0).astype(int)

    r_col = _find_col(df, COLUMN_ALIASES["rssi"])
    if r_col is None:
        out["rssi"] = -65.0
    else:
        r = pd.to_numeric(df[r_col], errors="coerce")
        out["rssi"] = r.fillna(r.median() if r.notna().any() else -65.0)

    l_col = _find_col(df, COLUMN_ALIASES["length"])
    out["length"] = (pd.to_numeric(df[l_col], errors="coerce").fillna(0).astype(int)
                     if l_col else 0)

    lab_col = _find_col(df, COLUMN_ALIASES["label"])
    out["label"] = _normalise_label(df[lab_col]) if lab_col else 0

    return out.dropna(subset=["timestamp", "src"]).reset_index(drop=True)
