import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)

from constants import POI_MAP, basepath, flips_model, mapp_models_filp, out_userstudy, path_annotator_poi, path_mrk, raters_all

out_folder = Path(basepath, "results", "long")
out_folder.mkdir(exist_ok=True)
# =========================
# Configuration
# =========================
ANGLE_COLS = [
    "tibia_torsion_2D",
    "femoral_torsion_2D",
    "mLDFA",
    "MPTA",
    "HKA_2D",
    "PDFA_sagittal_2D",
    "PDFA_medial_3D",
    "PDFA_lateral_3D",
    "tibial_slope_medial",
    "tibial_slope_lateral",
]

CLINICAL_THRESHOLDS = {
    "tibia_torsion_2D": 2.0,
    "femoral_torsion_2D": 2.0,
    "mLDFA": 2.0,
    "MPTA": 2.0,
    "HKA_2D": 2.0,
    "PDFA_sagittal_2D": 2.0,
    "PDFA_medial_3D": 2.0,
    "PDFA_lateral_3D": 2.0,
    "tibial_slope_medial": 2.0,
    "tibial_slope_lateral": 2.0,
}

RETEST_VARIABILITY = 0.3  # degrees
df = pd.read_excel(Path(__file__).parent.parent.parent / "results/cohort/angles_dataset_with_duplicate.xlsx")
# =========================
# Preprocessing
# =========================
df = df.copy()

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
df["date_str"] = df["date"].dt.strftime("%Y%m%d")

# age: extract number, fill within subject if missing
df["age_num"] = df["age"].astype(str).str.replace("Y", "", regex=False).replace("None", np.nan).astype(float)

df = df.sort_values("date")
df["age_filled"] = df.groupby("sub")["age_num"].transform(lambda x: x.ffill().bfill())

# Define knee unit
df["knee_id"] = df["sub"] + "_" + df["leg"]

# =========================
# Longitudinal deltas
# =========================
records = []


for knee_id, g in df.groupby("knee_id"):
    if len(g) < 2:
        continue

    g = g.sort_values("date")

    t_years = (g["date"].iloc[-1] - g["date"].iloc[0]).days / 365.25
    if t_years <= 0.5:
        continue
    for angle in ANGLE_COLS:
        delta = g[angle].iloc[-1] - g[angle].iloc[0]

        records.append(
            {
                "knee_id": knee_id,
                "angle": angle,
                "delta_total": delta,
                "delta_per_year": delta / t_years,
                "date_start": g["date_str"].iloc[0],
                "date_end": g["date_str"].iloc[-1],
                "clinically_relevant": abs(delta) >= CLINICAL_THRESHOLDS[angle],
            }
        )

long_df = pd.DataFrame(records)

# =========================
# Table: Δ / year
# =========================
summary_table = (
    long_df.groupby("angle")
    .agg(
        mean_delta_per_year=("delta_per_year", "mean"),
        sd_delta_per_year=("delta_per_year", "std"),
        clinically_relevant_pct=("clinically_relevant", "mean"),
        n_knees=("knee_id", "nunique"),
    )
    .reset_index()
)

summary_table["clinically_relevant_pct"] *= 100

print(f"\nΔ/Jahr Summary Table; averge images {len(long_df) / len(ANGLE_COLS)} knees")
print(summary_table.round(3))
long_df.to_excel(out_folder / "change_per_year_individual.xlsx")
summary_table.to_excel(out_folder / "change_per_year.xlsx")
# =========================
# Plot 3: Noise vs Signal
# =========================
noise_signal = []

for angle in ANGLE_COLS:
    # longitudinal changes
    for v in long_df.loc[long_df["angle"] == angle, "delta_per_year"]:
        noise_signal.append({"angle": angle, "value": (v), "type": "Longitudinale Änderung |Δ|/Jahr"})

    # re-test variability
    noise_signal.append(
        {
            "angle": angle,
            # "value": RETEST_VARIABILITY,
            # "type": "Re-Test-Variabilität",
        }
    )

ns_df = pd.DataFrame(noise_signal)

fig_noise = px.box(
    ns_df,
    x="angle",
    y="value",
    color="type",
    # box=True,
    # points="all",
    title="longitudinale Veränderung",
)

fig_noise.update_layout(yaxis_title="Grad pro Jahr (|Δ|)", xaxis_title="Winkelparameter")

fig_noise.write_image(out_folder / "fig_noise.svg", width=1500, height=800)

# =========================
# Population trend over time
# =========================
# Time since baseline per knee
df["baseline_date"] = df.groupby("knee_id")["date"].transform("min")
df["time_years"] = (df["date"] - df["baseline_date"]).dt.days / 365.25
df = df[df["time_years"] > 0.5]
for angle in ANGLE_COLS:
    fig = px.scatter(
        df,
        x="time_years",
        y=angle,
        opacity=0.3,
        trendline="lowess",
        title=f"Populationstrend über Zeit – {angle}",
    )

    fig.update_layout(xaxis_title="Zeit seit Baseline (Jahre)", yaxis_title="Winkel (°)")

    fig.write_image(out_folder / f"angle_{angle}.svg", width=1500, height=800)
