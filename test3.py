import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Directory where the data is stored
DATA_DIR = r'mmash\mmash\MMASH\DataPaper'

for user_id in range(1, 23):

    # Select a user (or loop over all, for demonstration we're using one user)
    #user_id = 1
    user_folder = os.path.join(DATA_DIR, f"user_{user_id}")

    # Read Actigraph.csv for HR data
    actigraph_file = os.path.join(user_folder, "Actigraph.csv")
    df_actigraph = pd.read_csv(actigraph_file)

    # 1) Parse timestamps and extract day/hour
    #    Suppose we have columns: 'day', 'time', 'HR'
    #    If 'time' is in "HH:MM:SS" format, parse it:
    df_actigraph['datetime'] = df_actigraph['time'].apply(lambda t: datetime.strptime(t, "%H:%M:%S"))

    # Extract hour from parsed time
    df_actigraph['hour'] = df_actigraph['datetime'].dt.hour

    # 2) Group by (day, hour) and compute average HR
    grouped = df_actigraph.groupby(['day', 'hour'])['HR'].mean().reset_index()

    # 3) Pivot for heatmap: rows=day, columns=hour, values=HR
    pivot_df = grouped.pivot(index='day', columns='hour', values='HR')

    # 4) Create a heatmap
    plt.figure(figsize=(12, 4))
    # Choose a colormap that transitions from green (low HR) to red (high HR)
    # e.g. 'RdYlGn_r' is reversed so low is green, high is red
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)

    ax = sns.heatmap(
        pivot_df,
        cmap=cmap,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Heart Rate (BPM)'}
    )
    ax.set_title("Day-by-Hour Heart Rate (Proxy for Stress)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day Number")

    plt.show()