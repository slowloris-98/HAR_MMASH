import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates



activity_colors = {
    1: 'blue',
    2: 'red',
    3: 'green',
    4: 'yellow',
    5: 'orange',
    6: 'purple',
    7: 'brown',
    8: 'pink',
    9: 'gray',
    10: 'cyan',
    11: 'magenta',
    12: 'lime'
}

# Directory where the data is stored
DATA_DIR = r'mmash\mmash\MMASH\DataPaper'

# Select a user (or loop over all, for demonstration we're using one user)
user_id = 14
user_folder = os.path.join(DATA_DIR, f"user_{user_id}")


activity_file = os.path.join(user_folder, "Activity.csv")
df_actv = pd.read_csv(activity_file)


# Convert "Start" / "End" to datetime, then to minutes
df_actv['Start'] = df_actv['Start'].str.replace('24:00','00:00')
df_actv['End']   = df_actv['End'].str.replace('24:00','00:00')
df_actv['Start'] = pd.to_datetime(df_actv['Start'], format='%H:%M')
df_actv['End'] = pd.to_datetime(df_actv['End'], format='%H:%M')
df_actv = df_actv[df_actv['Activity'] != 0]
df_actv.loc[df_actv['Start'] > df_actv['End'], 'End'] += pd.Timedelta(days=1)

# Calculate time of day in hours and duration (in minutes)
df_actv['TimeOfDay'] = df_actv['Start'].dt.hour + df_actv['Start'].dt.minute / 60.0
df_actv['Duration'] = (df_actv['End'] - df_actv['Start']).dt.total_seconds() / 60.0
df_actv['Act_color'] = df_actv['Activity'].map(activity_colors)


print(df_actv.shape)
print(df_actv.head())


# Get sorted unique dates
unique_days = sorted(df_actv['Day'].unique())
day2num = {day: i for i, day in enumerate(unique_days)}

# Create a new column mapping each date to a numeric index
df_actv['DayIndex'] = df_actv['Day'].map(day2num)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot all rows in a single scatter call
scatter = ax.scatter(
    df_actv['DayIndex'],       # x: numeric day index
    df_actv['TimeOfDay'],      # y: time in hours
    s=df_actv['Duration'],     # marker size
    c=df_actv['Act_color'],           # color scale = HR
    #cmap='viridis',
    alpha=0.8,
    edgecolors='black'
)

# Set x-ticks to these day indices
ax.set_xticks(list(day2num.values()))

# Label them with the actual date string
#ax.set_xticklabels([day.strftime('%m/%d') for day in unique_days])

# Add a colorbar for HR
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Heart Rate (BPM)')

ax.set_xlabel('Day')
ax.set_ylabel('Time of Day (hour)')
ax.set_title('Activity Profile with HR as Color')
plt.tight_layout()
plt.show()





'''
fig, ax = plt.subplots(figsize=(10, 6))

# Plot all rows in a single scatter call
scatter = ax.scatter(
    df_actv['Start'],            # x: actual datetime
    df_actv['TimeOfDay'],        # y: time in hours
    s=df_actv['Duration'],       # marker size = duration
    #c=activity_colors.get(df_actv['Activity'], 'gray'),             # color = HR
    cmap='viridis',              # or another colormap: 'plasma', 'coolwarm', etc.
    alpha=0.8,
    edgecolors='black'
)

# Format the x-axis to show just month/day (or any format you want)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

# Add a colorbar for HR
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Heart Rate (BPM)')

ax.set_xlabel('Day')
ax.set_ylabel('Time of Day (hour)')
ax.set_title('Activity Profile with HR as Color')
plt.tight_layout()
plt.show()
'''