#%% Importing packages

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr 
import pandas as pd 
from pathlib import Path

# %%
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# %%
station = "AWS5"

def read_aws(station, year):
    file_path = DATA_DIR / station / f"K-transect_{station}_{year}.tab"
    if not file_path.exists():
        return None
    with open(file_path, encoding="utf-8") as f:
        skip = next(i + 1 for i, line in enumerate(f) if line.strip() == "*/")
    return pd.read_csv(file_path, sep="\t", skiprows=skip,
                       parse_dates=["Date/Time"], index_col="Date/Time")

dfs = [read_aws(station, y) for y in range(2003, 2022)]
aws = pd.concat([d for d in dfs if d is not None]).sort_index()

print(aws.columns.tolist())
print(aws["ID"].value_counts())

# %%
aws["seg"] = (aws["Vlog [V]"] == 100).cumsum()
print(aws.index[aws["Vlog [V]"] == 100])  

rows = []
for s, g in aws[aws["Vlog [V]"] != 100].groupby("seg"):
    sensor = g["ID"].iloc[0]
    if sensor == 0:
        h = g["Height rel [m]"].dropna()
        melt = h.iloc[-48:].median() - h.iloc[:48].median()
    else:
        m = g["Ice melt [m]"].dropna()
        melt = m.iloc[:48].median() - m.iloc[-48:].median()
    rows.append({"start": g.index[0], "end": g.index[-1],
                 "ID": sensor, "melt_m": melt})

melt = pd.DataFrame(rows)
print(melt)

# %%
