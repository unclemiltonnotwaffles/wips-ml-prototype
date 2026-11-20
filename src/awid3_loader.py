"""Normalise common AWID3/Wireshark CSV exports into the project schema."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

COLUMN_ALIASES = {
    "timestamp": ["frame.time_epoch", "timestamp", "time", "frame.time"],
    "src": ["wlan.sa", "wlan.ta", "wlan.sa_resolved", "src", "source", "source_address"],
    "dst": ["wlan.da", "wlan.ra", "wlan.da_resolved", "dst", "dest", "destination_address"],
    "fc_type": ["wlan.fc.type", "frame.type", "type"],
    "fc_subtype": ["wlan.fc.subtype", "frame.subtype", "subtype"],
    "rssi": ["radiotap.dbm_antsignal", "rssi", "signal", "dbm"],
    "length": ["data.len", "frame.len", "length"],
    "label": ["attack", "class", "label", "is_attack"],
}
TYPE_SUBTYPE_ALIASES = ["wlan.fc.type_subtype", "frame.type_subtype"]


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    return next((lower[a.lower()] for a in aliases if a.lower() in lower), None)


def _timestamp_seconds(series: pd.Series) -> pd.Series:
    """Parse numeric epoch values and human-readable timestamps row by row."""
    strings = series.astype("string").str.strip()
    numeric_mask = strings.str.match(
        r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$",
        na=False,
    )
    result = pd.Series(float("nan"), index=series.index, dtype="float64")
    if numeric_mask.any():
        result.loc[numeric_mask] = pd.to_numeric(strings.loc[numeric_mask], errors="coerce")
    date_mask = ~numeric_mask & strings.notna()
    if date_mask.any():
        parsed = pd.to_datetime(strings.loc[date_mask], errors="coerce", utc=True)
        epoch = parsed.astype("int64").where(parsed.notna(), float("nan")) / 1e9
        result.loc[date_mask] = epoch
    return result


def _integer_field(series: pd.Series, field: str) -> pd.Series:
    """Parse decimal and hexadecimal Wireshark fields without silent zero-fill."""
    strings = series.astype(str).str.strip()
    hex_mask = strings.str.match(r"(?i)^0x[0-9a-f]+$")
    result = pd.to_numeric(strings.where(~hex_mask), errors="coerce")
    if hex_mask.any():
        result.loc[hex_mask] = strings.loc[hex_mask].map(lambda x: int(x, 16))
    return result


def _normalise_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    benign = {"normal", "0", "false", "benign", "background", "nan", ""}
    return (~s.isin(benign)).astype(int)


def load_awid3_dir(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    frames = [_normalise_one(pd.read_csv(path, low_memory=False), path.name) for path in csvs]
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp", "src", "dst"])
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if out.empty:
        raise ValueError("AWID3 files contained no rows with valid timestamps and addresses")
    print(f"[awid3_loader] loaded {len(out):,} frames from {len(csvs)} file(s)")
    print(f"[awid3_loader] labelled attack frames: {int(out['label'].sum()):,} "
          f"({out['label'].mean()*100:.2f}%)")
    return out


def _normalise_one(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{name}: CSV is empty")
    ts_col = _find_col(df, COLUMN_ALIASES["timestamp"])
    if ts_col is None:
        raise KeyError(f"{name}: missing timestamp column")
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = _timestamp_seconds(df[ts_col])

    for key in ("src", "dst"):
        col = _find_col(df, COLUMN_ALIASES[key])
        if col is None:
            raise KeyError(f"{name}: missing {key} address column")
        out[key] = df[col].astype("string").str.strip().str.lower()
        out.loc[out[key].isin(["", "nan", "none"]), key] = pd.NA

    packed_col = _find_col(df, TYPE_SUBTYPE_ALIASES)
    type_col = _find_col(df, COLUMN_ALIASES["fc_type"])
    subtype_col = _find_col(df, COLUMN_ALIASES["fc_subtype"])
    if packed_col:
        packed = _integer_field(df[packed_col], "frame control")
        # 802.11 frame-control byte uses high nibble for type and low nibble
        # for subtype in the commonly exported packed 0x00TS representation.
        out["fc_type"] = (packed // 16).where(packed.notna())
        out["fc_subtype"] = (packed % 16).where(packed.notna())
    elif type_col and subtype_col:
        out["fc_type"] = _integer_field(df[type_col], "frame type")
        out["fc_subtype"] = _integer_field(df[subtype_col], "frame subtype")
    else:
        raise KeyError(f"{name}: need packed frame control or both type/subtype columns")

    for field in ("fc_type", "fc_subtype"):
        if out[field].isna().any():
            raise ValueError(f"{name}: unparseable {field} values; refusing silent coercion")
        out[field] = out[field].astype("int64")

    rssi_col = _find_col(df, COLUMN_ALIASES["rssi"])
    if rssi_col:
        rssi = pd.to_numeric(df[rssi_col], errors="coerce")
        out["rssi"] = rssi.fillna(rssi.median() if rssi.notna().any() else -65.0)
    else:
        out["rssi"] = -65.0

    length_col = _find_col(df, COLUMN_ALIASES["length"])
    out["length"] = (pd.to_numeric(df[length_col], errors="coerce").fillna(0)
                     if length_col else 0)
    label_col = _find_col(df, COLUMN_ALIASES["label"])
    out["label"] = _normalise_label(df[label_col]) if label_col else 0
    return out
