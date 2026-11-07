"""Shared helpers for analysis/ scripts: load market_rows, format residual cells."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
DYN = ["fp_dynasty", "dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]
DD = ["dd_ktc", "dd_fantasycalc", "dd_dynastyprocess"]


def out_dir(sf=False):
    return HERE / "outputs" / ("sf" if sf else "")


def rows(markets, h, sf=False):
    """market_rows for the given markets at horizon h; r = residual as % of that market's mean outcome."""
    out = []
    for m in markets:
        f = out_dir(sf) / "market_rows" / f"{m}_h{h}.parquet"
        if f.exists():
            x = pd.read_parquet(f)
            x["r"] = x.resid / x[f"vorp{h}"].mean() * 100
            x["mk"], x["h"] = m, h
            out.append(x)
    return pd.concat(out, ignore_index=True) if out else None


def cell(g, col="r"):
    """mean/median, % beat, cohorts with positive mean / cohorts, distinct players."""
    if len(g) == 0:
        return ""
    coh = g.groupby("t")[col].mean()
    return (f"{g[col].mean():+.0f}/{g[col].median():+.0f} {(g[col] > 0).mean() * 100:.0f}%b "
            f"c{(coh > 0).sum()}/{len(coh)} n{g.gsis_id.nunique()}")


def table(frames, masks, title, col="r"):
    """frames: {column label: df}; masks: {row label: function df -> bool Series}."""
    out = {lab: {k: cell(d[f(d)], col) for k, f in masks.items()} for lab, d in frames.items()}
    print(f"\n== {title}  [mean/median %, % beat, cohorts positive/total, players]")
    print(pd.DataFrame(out).to_string())
