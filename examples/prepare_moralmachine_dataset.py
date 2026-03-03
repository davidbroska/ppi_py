#!/usr/bin/env python3
"""Prepare a reduced Moral Machine dataset for the example notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def CalcTheoreticalInt(r):
    # this function is applied to each row (r)
    if r["Intervention"]==0:
        if r["Barrier"]==0:
            if r["PedPed"]==1: p = 0.48
            else: p = 0.32
            
            if r["CrossingSignal"]==0:   p = p * 0.48
            elif r["CrossingSignal"]==1: p = p * 0.2
            else: p = p * 0.32
        else: p = 0.2

    else: 
        if r["Barrier"]==0:
            if r["PedPed"]==1: 
                p = 0.48
                if r["CrossingSignal"]==0: p = p * 0.48
                elif r["CrossingSignal"]==1: p = p * 0.32
                else: p = p * 0.2
            else: 
                p = 0.2
                if r["CrossingSignal"]==0: p = p * 0.48
                elif r["CrossingSignal"]==1: p = p * 0.2
                else: p = p * 0.32
        else: p = 0.32  
    
    return(p)  
        
def calcWeightsTheoretical(profiles):
    
    p = profiles.apply(CalcTheoreticalInt, axis=1)

    weight = 1/p 

    return(weight)  


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = (
        repo_root
        / "examples"
        / "osfstorage-archive (3)"
        / "Data"
        / "3_gpt4turbo_wp_20241118.csv.gz"
    )
    default_output = repo_root / "examples" / "moralmachine_reduced.npz"

    parser = argparse.ArgumentParser(
        description="Prepare a reduced Moral Machine dataset for AMCE examples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to full Moral Machine data CSV/CSV.GZ.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path for reduced output NPZ file.",
    )
    parser.add_argument(
        "--llm-col",
        default="gpt4turbo_wp_Saved",
        help="Column name for LLM-predicted outcome.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Keep these columns in memory to compute theoretical weights.
    required_cols = [
        "UserID",
        "Intervention",
        "Barrier",
        "CrossingSignal",
        "PedPed",
        "Saved",
        args.llm_col,
    ]

    try:
        df = pd.read_csv(args.input, usecols=required_cols)
    except ValueError as err:
        raise ValueError(
            f"Could not load required columns from {args.input}. "
            f"Requested columns: {required_cols}"
        ) from err

    df["weights"] = calcWeightsTheoretical(df)

    out_cols = [
        "UserID",
        "weights",
        "Intervention",
        "Saved",
        args.llm_col,
    ]
    out = df[out_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        UserID=np.asarray(out["UserID"].astype(str), dtype=np.str_),
        weights=out["weights"].to_numpy(),
        Intervention=out["Intervention"].to_numpy(),
        Saved=out["Saved"].to_numpy(),
        Yhat=out[args.llm_col].to_numpy(),
        Yhat_name=np.array(args.llm_col),
    )

    print(f"Rows: {len(out):,}")
    print(f"Columns: {out_cols}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
