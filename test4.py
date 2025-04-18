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

# --- Utility Functions ---
def parse_time_hms(t_str):
    """Parse 'HH:MM:SS' to a datetime.time object."""
    return datetime.strptime(t_str, "%H:%M:%S").time()

def time_to_minutes(t):
    """Convert a datetime.time object to minutes from midnight."""
    return t.hour * 60 + t.minute + t.second/60

def main():
    # 1) Read Actigraph data for HR
    #actigraph_file = os.path.join(user_folder, "Actigraph.csv")

    # Directory where the data is stored
    DATA_DIR = r'mmash\mmash\MMASH\DataPaper'
    
    # Select a user (or loop over all, for demonstration we're using one user)
    user_id = 6
    user_folder = os.path.join(DATA_DIR, f"user_{user_id}")

    #df_act = pd.read_csv(actigraph_file)
    # Read Actigraph.csv for HR data
    actigraph_file = os.path.join(user_folder, "Actigraph.csv")
    df_act = pd.read_csv(actigraph_file)

    # Clean up any Unnamed columns
    df_act = df_act.loc[:, ~df_act.columns.str.contains('^Unnamed')]

    # Parse time column (HH:MM:SS) -> datetime.time
    #df_act['time_obj'] = df_act['time'].apply(parse_time_hms)
    df_act['time_obj'] = pd.to_datetime(df_act['time'], format='%H:%M:%S')
    # Convert to minutes from midnight
    #df_act['time_minutes'] = df_act['time_obj'].apply(time_to_minutes)
    #df_act.loc[df_act['day'] == 2, 'time_obj'] += pd.Timedelta(days=1)

    print("df_act['time_obj']")
    print(df_act['time_obj'])

    # We'll combine day + time_minutes into one timeline
    # e.g., if day=1, we keep as is; if day=2, we add 24*60 = 1440 minutes offset
    #df_act['day_offset'] = (df_act['day'] - 1) * 1440
    #df_act['timeline'] = df_act['time_minutes'] + df_act['day_offset']

    # 2) Read Activity data for context
    activity_file = os.path.join(user_folder, "Activity.csv")
    df_actv = pd.read_csv(activity_file)
    
    # Convert "Start" / "End" to datetime, then to minutes
    df_actv['Start'] = df_actv['Start'].str.replace('24:00','00:00')
    df_actv['End']   = df_actv['End'].str.replace('24:00','00:00')
    df_actv['Start'] = pd.to_datetime(df_actv['Start'], format='%H:%M')
    df_actv['End'] = pd.to_datetime(df_actv['End'], format='%H:%M')
    


    #df_actv['Start'] = pd.to_datetime(df_actv['Start'], errors='coerce')
    #df_actv['End']   = pd.to_datetime(df_actv['End'], errors='coerce')
    # Fix "start > end" scenario
    df_actv.loc[df_actv['Start'] > df_actv['End'], 'End'] += pd.Timedelta(days=1)
    
    print("df_actv['Start']")
    print(df_actv['Start'])
    
    # Convert start/end to minutes from midnight + offset
    #df_actv['start_minutes'] = df_actv['Start'].dt.hour * 60 + df_actv['Start'].dt.minute + (df_actv['Day'] - 1)*1440
    #df_actv['end_minutes']   = df_actv['End'].dt.hour * 60 + df_actv['End'].dt.minute   + (df_actv['Day'] - 1)*1440

    # Filter out activity=0 if needed
    df_actv = df_actv[df_actv['Activity'] != 0]
    # Calculate duration in minutes
    df_actv['duration'] = (df_actv['End'] - df_actv['Start']).dt.total_seconds() / 60

    # Keep only activities longer than 60 minutes
    df_actv = df_actv[df_actv['duration'] > 20]

    # 3) Optionally read questionnaire for daily stress
    questionnaire_file = os.path.join(user_folder, "questionnaire.csv")
    if os.path.exists(questionnaire_file):
        df_q = pd.read_csv(questionnaire_file)
        daily_stress = df_q['Daily_stress'].iloc[0] if 'Daily_stress' in df_q.columns else np.nan
    else:
        daily_stress = np.nan

    # 4) Determine baseline stress (e.g., median HR) from Actigraph
    # We'll treat HR as "stress" for demonstration
    baseline_hr = df_act['HR'].median()

    # 5) Plot
    plt.figure(figsize=(10,6))
    
    # Plot HR (stress) over timeline
    # We can do a scatter or line. Let's do a scatter for more raw data look
    #plt.scatter(df_act['timeline'], df_act['HR'], c='red', s=10, alpha=0.5, label='Heart Rate')

    # Color mapping: Red for above baseline, Green for below baseline
    colors = ['red' if hr > baseline_hr else 'green' for hr in df_act['HR']]

    # Scatter plot with conditional colors
    plt.scatter(df_act['time_obj'], df_act['HR'], c=colors, s=1, alpha=0.5, label="Heart Rate")
    
    # Plot the baseline as a dashed line
    plt.axhline(baseline_hr, color='black', linestyle='--', label=f"Baseline HR = {baseline_hr:.1f}")
    
    # Add vertical lines for activity transitions
    # We'll just plot lines where a new activity starts
    for idx, row in df_actv.iterrows():
        # x is row['start_minutes'], y from 0 to e.g. max HR
        plt.axvline(x=row['Start'], color='b', linewidth=1, alpha=1)
        plt.axvline(x=row['End'], color='k', linewidth=1, alpha=1)

        
    # Combine start and end times, and get unique sorted values
    ticks = pd.concat([df_actv['Start'], df_actv['End']]).unique()
    #ticks.sort()  # sort the tick values

    # Set the ticks and labels
    plt.xticks(ticks, ticks, rotation=45)

    # (Optional) Label the activities near the top of the plot
    max_hr = df_act['HR'].max()
    for idx, row in df_actv.iterrows():
        activity = activity_labels.get(row['Activity'], 'Unknown')
        # Place label a bit above the max HR
        plt.text(row['Start'], max_hr+5, str(activity), rotation=45, fontsize=8)
    
    # If you want to annotate daily_stress:
    #if not np.isnan(daily_stress):
    #    plt.text(df_act['timeline'].min(), baseline_hr - 10, f"Daily Stress: {daily_stress}", fontsize=9, color='blue')
    # Set x-axis format to "date - HH:MM AM/PM"
    #plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Tick every hour
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))  # Format as HH:MM AM/PM
    # Format axis
    plt.xlabel("Time")
    plt.ylabel("Heart Rate (BPM)")
    plt.title(f"Contextual Stress Profile (User {user_id})")
    plt.legend()
    plt.ylim(0, max_hr+20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
