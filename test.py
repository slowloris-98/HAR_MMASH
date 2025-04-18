import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

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

# --- Helper Functions ---

def parse_time(time_str):
    """
    Parse a time string in HH:MM:SS (or HH:MM) format and return a datetime.time object.
    """
    try:
        return datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        return datetime.strptime(time_str, "%H:%M").time()

def time_to_minutes(t):
    """Convert a datetime.time object to minutes since midnight."""
    return t.hour * 60 + t.minute + t.second/60

def hr_from_ibi(ibi):
    """Compute heart rate (beats per minute) from inter-beat interval (seconds)."""
    return 60 / ibi if ibi > 0 else np.nan

def map_hr_to_color(hr, hr_min, hr_max):
    """
    Map a heart rate value to a color on a blue (low) to red (high) gradient.
    We'll use matplotlib's colormap.
    """
    import matplotlib.cm as cm
    # Normalize hr between 0 and 1
    norm_val = (hr - hr_min) / (hr_max - hr_min) if hr_max > hr_min else 0.5
    # Use a colormap (e.g., 'coolwarm' goes from blue to red)
    cmap = cm.get_cmap('coolwarm')
    return cmap(norm_val)

# --- Main Code ---

def build_graph(activity_df, rr_df):
    """
    Build the activity graph using:
    - Activity data to determine nodes and transitions.
    - RR data to compute average heart rate during each activity segment.
    
    Returns a NetworkX DiGraph with node attributes:
      - 'total_time': total minutes spent in that activity.
      - 'avg_hr': weighted average heart rate (across segments in that activity).
      - 'color': color based on avg_hr.
    And edges with attribute:
      - 'transitions': count of transitions between activities.
    """
    # Ensure times are parsed as datetime.time objects and convert them to minutes from midnight.
    # For the activity data, we assume "Start" and "End" are strings.
    activity_df['Start_minutes'] = activity_df['Start'].apply(lambda x: time_to_minutes(x))
    activity_df['End_minutes']   = activity_df['End'].apply(lambda x: time_to_minutes(x))
    
    
    activity_df['Duration'] = activity_df['End_minutes'] - activity_df['Start_minutes']
    
    # Create a new column to store the average HR for the segment.
    # For each row (activity segment), we will extract the RR data points that occur on the same day
    # and within the [Start_minutes, End_minutes] interval.
    avg_hr_segments = []
    for idx, row in activity_df.iterrows():
        day = row['Day']
        start = row['Start_minutes']
        end = row['End_minutes']
        # Filter RR data for the same day
        rr_day = rr_df[rr_df['day'] == day].copy()
        # Parse the 'time' column in RR data and convert to minutes
        rr_day['time_minutes'] = rr_day['time'].apply(lambda x: time_to_minutes(parse_time(x)))
        # Select RR data points that fall within the activity segment interval
        rr_segment = rr_day[(rr_day['time_minutes'] >= start) & (rr_day['time_minutes'] <= end)]
        if not rr_segment.empty:
            # Calculate HR from ibi_s values
            rr_segment['HR'] = rr_segment['ibi_s'].apply(hr_from_ibi)
            # Compute the average HR for this segment (weighted by the number of beats or simply the mean)
            avg_hr = rr_segment['HR'].mean()
        else:
            avg_hr = np.nan  # No RR data available in this period
        avg_hr_segments.append(avg_hr)
    activity_df['avg_hr'] = avg_hr_segments
    
    # Aggregate data per activity (node)
    node_data = {}
    # We'll also aggregate a weighted average HR (weight by duration)
    for idx, row in activity_df.iterrows():
        cat = row['Activity']
        duration = row['Duration']
        hr = row['avg_hr']
        if cat not in node_data:
            node_data[cat] = {'total_time': 0, 'hr_sum': 0, 'duration_sum': 0}
        node_data[cat]['total_time'] += duration
        if not np.isnan(hr):
            node_data[cat]['hr_sum'] += hr * duration
            node_data[cat]['duration_sum'] += duration

    # Determine global min and max HR (across nodes) for coloring.
    avg_hr_per_node = {}
    for cat, data in node_data.items():
        if data['duration_sum'] > 0:
            avg_hr = data['hr_sum'] / data['duration_sum']
        else:
            avg_hr = np.nan
        avg_hr_per_node[cat] = avg_hr

    # Get min and max HR values (ignoring nan)
    valid_hr = [v for v in avg_hr_per_node.values() if not np.isnan(v)]
    hr_min = min(valid_hr) if valid_hr else 50
    hr_max = max(valid_hr) if valid_hr else 100

    # Now, build the graph
    G = nx.DiGraph()

    # Add nodes: each activity category becomes a node
    for cat, data in node_data.items():
        avg_hr = avg_hr_per_node.get(cat, np.nan)
        # Map average HR to a color
        node_color = map_hr_to_color(avg_hr, hr_min, hr_max) if not np.isnan(avg_hr) else (0.5, 0.5, 0.5, 1.0)
        G.add_node(cat,
                   label=activity_labels.get(cat, f"Activity {cat}"),
                   total_time=data['total_time'],
                   avg_hr=avg_hr,
                   color=node_color)

    # Calculate transitions (edges) from sequential activities (within same participant)
    transitions = {}
    # Sort activity_df by day and start time
    activity_df = activity_df.sort_values(by=['Day', 'Start_minutes'])
    prev_cat = None
    prev_day = None
    for idx, row in activity_df.iterrows():
        curr_cat = row['Activity']
        curr_day = row['Day']
        if prev_cat is not None and curr_day == prev_day:
            pair = (prev_cat, curr_cat)
            transitions[pair] = transitions.get(pair, 0) + 1
        prev_cat = curr_cat
        prev_day = curr_day

    # Add edges to the graph
    for (src, dst), count in transitions.items():
        G.add_edge(src, dst, transitions=count)

    return G

def draw_graph(G, node_scale=5, edge_scale=1):
    """
    Draw the graph with:
      - Node size proportional to total_time.
      - Node color from 'color' attribute.
      - Edge width proportional to transitions.
    """
    pos = nx.spring_layout(G, seed=42)


    # Prepare node sizes and colors
    node_sizes = [G.nodes[n]['total_time'] * node_scale for n in G.nodes()]
    
    # scaling up the node size
    node_scale_factor = 2     
    node_sizes = [size * node_scale_factor for size in node_sizes]

    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    labels = {n: G.nodes[n]['label'] for n in G.nodes()}

    # Prepare edge widths
    edge_widths = [G[u][v]['transitions'] * edge_scale for u, v in G.edges()]

    # scaling up the edge width
    edge_width_factor = 3
    edge_widths = [width * edge_width_factor for width in edge_widths]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='->', arrowsize=12, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')
    plt.title("Participant Activity Profile (Node color: Avg HR)")
    plt.axis('off')
    plt.show()

def main():
    # Load the data files (adjust file paths as necessary)
    # For this example, we assume CSV files are in the current directory.
    #activity_csv = "Activity.csv"  # must contain: Participant, day, Category, Start, End
    #rr_csv = "RR.csv"              # must contain: ibi_s, day, time
    
    DATA_DIR = r'mmash\mmash\MMASH\DataPaper' 
    user_id = 10
    user_folder = os.path.join(DATA_DIR, f"user_{user_id}")
    activity_file = os.path.join(user_folder, "Activity.csv")
    df_activity = pd.read_csv(activity_file)

    rr_file = os.path.join(user_folder, "RR.csv")
    df_rr = pd.read_csv(rr_file)

    acti_file = os.path.join(user_folder, "Actigraph.csv")
    df_acti = pd.read_csv(acti_file)

    df_activity['Start'] = df_activity['Start'].str.replace('24:00', '00:00')
    df_activity['End'] = df_activity['End'].str.replace('24:00', '00:00')
    df_activity['Start'] = pd.to_datetime(df_activity['Start'], errors='coerce')
    df_activity['End'] = pd.to_datetime(df_activity['End'], errors='coerce')

    #df_activity['Start_minutes'] = df_activity['Start'].dt.hour * 60 + df_activity['Start'].dt.minute

    # Fix cases where 'End' time was '24:00' by adding a day to the date
    df_activity.loc[df_activity['Start'] > df_activity['End'], 'End'] += pd.Timedelta(days=1)

    
    df_activity = df_activity[df_activity['Activity'] != 0]

    # If there is a column 'Unnamed: 0', drop it:
    df_rr = df_rr.loc[:, ~df_rr.columns.str.contains('^Unnamed')]


    '''
    # For demonstration, select one participant (if there is a Participant column)
    participant_id = 1  # change as needed
    if "Participant" in df_activity.columns:
        df_activity = df_activity[df_activity["Participant"] == participant_id]
    # Similarly, if RR.csv contains participant info, filter accordingly.
    if "Participant" in df_rr.columns:
        df_rr = df_rr[df_rr["Participant"] == participant_id]
    '''
    # Build the graph (using activity and RR data)
    G = build_graph(df_activity, df_rr)

    # Draw the graph
    draw_graph(G, node_scale=5, edge_scale=0.5)

if __name__ == "__main__":
    main()
