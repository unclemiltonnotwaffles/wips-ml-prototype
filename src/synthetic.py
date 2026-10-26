"""
synthetic.py — generates fake-but-realistic 802.11 packet records.

WHY this file?
    - AWID3 requires an email sign-up; a live demo cannot depend on that.
    - Replaying fresh synthetic traffic proves the detector generalises
      (unseen attacker MAC, unseen timing) rather than memorising the training set.
    - Output schema is IDENTICAL to `awid3_loader.py`, so every downstream
      module is dataset-agnostic. Swapping synthetic → AWID3 is one command.

Schema (one row per 802.11 frame):
    timestamp | src | dst | fc_type | fc_subtype | rssi | length | label
"""

from __future__ import annotations

import random
import pandas as pd

# 802.11 subtype numbers (IEEE 802.11-2020, Table 9-1)
SUBTYPE_PROBE_REQ = 4
SUBTYPE_PROBE_RESP = 5
SUBTYPE_BEACON = 8
SUBTYPE_DISASSOC = 10
SUBTYPE_AUTH = 11
SUBTYPE_DEAUTH = 12

def _mac(rng: random.Random) -> str:
    return ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6))

def generate_traffic(
    duration_seconds: int = 600,
    n_devices: int = 15,
    attack_episodes: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a packet-level dataframe:
      • n_devices legitimate stations, low-rate mix of mgmt/control/data.
      • ~1 % of their frames are legitimate single deauths (so the model can't
        just threshold at 1 deauth).
      • attack_episodes deauth floods (150–400 fps, 3–8 s) from one attacker MAC.
    """
    rng = random.Random(seed)

    devices = [_mac(rng) for _ in range(n_devices)]
    broadcast = "ff:ff:ff:ff:ff:ff"
    ap_mac = _mac(rng)

    rows: list[dict] = []

    # ── 1. legitimate background traffic ─────────────────────────────────────
    for dev in devices:
        t = rng.uniform(0, 1.0)
        while t < duration_seconds:
            kind = rng.choices(
                ["probe_req", "data", "ack", "legit_deauth"],
                weights=[30, 60, 9, 1],
            )[0]
            if kind == "probe_req":
                rows.append(_pkt(t, dev, broadcast, 0, SUBTYPE_PROBE_REQ,
                                 rng.gauss(-65, 4), 40, 0))
            elif kind == "data":
                rows.append(_pkt(t, dev, ap_mac, 2, 0,
                                 rng.gauss(-65, 4), rng.randint(200, 1400), 0))
            elif kind == "ack":
                rows.append(_pkt(t, dev, ap_mac, 1, 13,
                                 rng.gauss(-65, 4), 14, 0))
            else:
                rows.append(_pkt(t, dev, ap_mac, 0, SUBTYPE_DEAUTH,
                                 rng.gauss(-65, 4), 26, 0))
            t += rng.uniform(0.3, 1.0)

    # AP beacons every ~100 ms
    t = 0.0
    while t < duration_seconds:
        rows.append(_pkt(t, ap_mac, broadcast, 0, SUBTYPE_BEACON,
                         rng.gauss(-45, 2), 250, 0))
        t += 0.1024

    # ── 2. attack episodes ───────────────────────────────────────────────────
    for _ in range(attack_episodes):
        attacker = _mac(rng)
        start = rng.uniform(5, duration_seconds - 10)
        length = rng.uniform(3, 8)
        rate = rng.uniform(150, 400)
        t = start
        while t < start + length:
            rows.append(_pkt(t, attacker, broadcast, 0, SUBTYPE_DEAUTH,
                             rng.gauss(-45, 3), 26, 1))
            t += 1.0 / rate

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

def _pkt(t, src, dst, fc_type, fc_subtype, rssi, length, label):
    return {
        "timestamp": round(float(t), 6),
        "src": src,
        "dst": dst,
        "fc_type": int(fc_type),
        "fc_subtype": int(fc_subtype),
        "rssi": float(rssi),
        "length": int(length),
        "label": int(label),
    }

if __name__ == "__main__":
    d = generate_traffic()
    print(d.head())
    print(f"{len(d)} packets, {int(d['label'].sum())} attack packets")
