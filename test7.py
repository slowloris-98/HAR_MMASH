import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates


# Define activity labels for readability
activity_labels = {
1: "Sleeping",
2: "Lying down",
3: "Sitting",
4: "Light movement",
5: "Medium movement",
6: "Heavy movement",
7: "Eating",
8: "Small screen usage",
9: "Large screen usage",
10: "Caffeinated drinks",
11: "Smoking",
12: "Alcohol consumption"
}

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
user_id = 1
user_folder = os.path.join(DATA_DIR, f"user_{user_id}")


activity_file = os.path.join(user_folder, "Activity.csv")
df_actv = pd.read_csv(activity_file)

# Read Actigraph.csv for HR data
#actigraph_file = os.path.join(user_folder, "Actigraph.csv")
#df_act = pd.read_csv(actigraph_file)

# Clean up any Unnamed columns
#df_act = df_act.loc[:, ~df_act.columns.str.contains('^Unnamed')]



# Parse time column (HH:MM:SS) -> datetime.time
#df_act['time_obj'] = df_act['time'].apply(parse_time_hms)
#df_act['time_obj'] = pd.to_datetime(df_act['time'], format='%H:%M:%S')

# Convert "Start" / "End" to datetime, then to minutes
df_actv['Start'] = df_actv['Start'].str.replace('24:00','00:00')
df_actv['End']   = df_actv['End'].str.replace('24:00','00:00')
df_actv['Start'] = pd.to_datetime(df_actv['Start'], format='%H:%M')
df_actv['End'] = pd.to_datetime(df_actv['End'], format='%H:%M')
df_actv.loc[df_actv['Start'] > df_actv['End'], 'End'] += pd.Timedelta(days=1)

# For each row in df_actv, find the mean HR between Start and End
#hr_values = []
#for i, row in df_actv.iterrows():
#    mask = (df_act['time_obj'] >= row['Start']) & (df_act['time_obj'] <= row['End'])
#    mean_hr = df_act.loc[mask, 'HR'].mean()
#    hr_values.append(mean_hr if not np.isnan(mean_hr) else 0)  # default to 0 or NaN if no data

#df_actv['HR'] = hr_values

# Extract date portion for x-axis
#df_actv['Day'] = df_actv['Start'].dt.date

# Calculate time of day in hours and duration (in minutes)
df_actv['TimeOfDay'] = df_actv['Start'].dt.hour + df_actv['Start'].dt.minute / 60.0
df_actv['Duration'] = (df_actv['End'] - df_actv['Start']).dt.total_seconds() / 60.0

'''
fig, ax = plt.subplots(figsize=(10, 6))

# Plot all rows in a single scatter call
scatter = ax.scatter(
    df_actv['Start'],            # x: actual datetime
    df_actv['TimeOfDay'],        # y: time in hours
    s=df_actv['Duration'],       # marker size = duration
    c=df_actv['HR'],             # color = HR
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

'''
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
    c=df_actv['HR'],           # color scale = HR
    cmap='coolwarm',
    alpha=0.8,
    edgecolors='black'
)

# Set x-ticks to these day indices
ax.set_xticks(list(day2num.values()))

# Label them with the actual date string
ax.set_xticklabels([day.strftime('%m/%d') for day in unique_days])

# Add a colorbar for HR
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Heart Rate (BPM)')

ax.set_xlabel('Day')
ax.set_ylabel('Time of Day (hour)')
ax.set_title('Activity Profile with HR as Color')
plt.tight_layout()
plt.show()
'''

'''
# Suppose df_actv is already defined and has:
# df_actv['Start'] (datetime)
# df_actv['TimeOfDay'] (numeric hours)
# df_actv['Duration'] (numeric, used for marker size)
# df_actv['HR'] (heart rate, used for color)

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    df_actv['Start'],        # x: datetime
    df_actv['TimeOfDay'],    # y: hours since midnight
    s=df_actv['Duration'],   # marker size
    c=df_actv['HR'],         # color
    cmap='viridis',
    alpha=0.8,
    edgecolors='black'
)

# 1) Collect the unique days from the 'Start' column (date part only).
unique_days = pd.to_datetime(df_actv['Start']).dt.normalize().unique()
# e.g., array of Timestamps like [Timestamp('2025-03-26 00:00:00'), Timestamp('2025-03-27 00:00:00')]

# 2) Sort them (just to be safe).
unique_days = sorted(unique_days)

# 3) Set the x-axis ticks to exactly these day timestamps (at midnight).
ax.set_xticks(unique_days)

# 4) Set the labels to, for example, "03/26" and "03/27".
ax.set_xticklabels([day.strftime('%m/%d') for day in unique_days])

# Optionally, if you want to remove any extra minor ticks, you can do:
#ax.xaxis.set_major_locator(mdates.FixedLocator(unique_days))
# or you can remove automatic locators by not calling any other locator methods

# Add a colorbar for HR
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Heart Rate (BPM)')

ax.set_xlabel('Day')
ax.set_ylabel('Time of Day (hour)')
ax.set_title('Activity Profile with HR as Color')
plt.tight_layout()
plt.show()
'''
'''
# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df_actv['Start'],       # x-axis: actual datetime values
    df_actv['TimeOfDay'],   # y-axis: time-of-day in hours
    s=df_actv['Duration'],  # marker size based on activity duration
    c=df_actv['HR'],        # color based on heart rate
    cmap='viridis',
    alpha=0.8,
    edgecolors='black'
)

# Customize x-axis ticks using the 'day' column.
# For each unique day, pick a representative datetime (e.g., the earliest start time).
unique_days = sorted(df_actv['Day'].unique())
day_ticks = []
day_labels = []
for day in unique_days:
    day_data = df_actv[df_actv['Day'] == day]
    tick_value = day_data['Start'].min()  # or you might choose the midpoint if preferred
    day_ticks.append(tick_value)
    day_labels.append(f'Day {day}')

# Set the ticks and labels on the x-axis
ax.set_xticks(day_ticks)
ax.set_xticklabels(day_labels)

# Optionally, if you want to remove extra auto-generated ticks,
# you can override the locator by using FixedLocator:
#ax.xaxis.set_major_locator(mdates.FixedLocator(day_ticks))

# Add a colorbar for HR
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Heart Rate (BPM)')

ax.set_xlabel('Day')
ax.set_ylabel('Time of Day (Hour)')
ax.set_title('Event-based Activity Profile with HR as Color')
plt.tight_layout()
plt.show()
'''
# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each activity as a scatter point
for idx, row in df_actv.iterrows():
    color = activity_colors.get(row['Activity'], 'gray')  # default to gray if not found
    print(row['Start'])
    ax.scatter(
        row['Start'],            # x: datetime from 'Start'
        row['TimeOfDay'],        # y: time-of-day (hours)
        s=row['Duration'] * 2,    # marker size (scaled as needed)
        c=color,                 
        alpha=0.8,
        edgecolors='black'
    )

# Customize the x-axis ticks based on the 'day' column.
# For each unique day, pick a representative datetime (here, the earliest start time for that day).
unique_days = sorted(df_actv['Day'].unique())
day_ticks = []
for d in unique_days:
    tick_value = df_actv[df_actv['Day'] == d]['Start'].min()
    day_ticks.append(tick_value)

print(day_ticks)

# Set the ticks and custom labels (e.g., "Day 1", "Day 2")
ax.set_xticks(day_ticks)
ax.set_xticklabels([f'Day {d}' for d in unique_days])
#ax.xaxis.set_major_locator(mdates.DateLocator(day_ticks))

# Label the axes and title the plot
ax.set_xlabel('Day')
ax.set_ylabel('Time of Day (Hour)')
ax.set_title('Event-based Activity Profile (Colored by Activity)')
plt.tight_layout()
plt.show()