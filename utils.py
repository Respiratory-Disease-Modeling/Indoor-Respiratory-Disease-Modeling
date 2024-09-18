import pandas as pd
import math
import datetime
import os
import matplotlib.pyplot as plt

from tabulate import tabulate
from configs import *

# ------------------------------------------
# ----------------- PART 2 -----------------
# ------------------------------------------

def calculate_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def analyze_data(df):
    success_count = 0
    failure_count = 0
    moving_count = 0

    distances = {}  # Dictionary to store distances for each tag
    same_place_tags = {}  # Dictionary to store tags in the same place or close proximity at specific times

    initial_timestamp = df['timestamp'].iloc[0]  # Initial timestamp
    final_timestamp = df['timestamp'].iloc[-1]  # Final timestamp
 
    unique_tag_ids = df['tagId'].nunique() # Calculate the total number of unique tagIds

    for index, row in df.iterrows():
        version = row['version']
        tag_id = row['tagId']
        success = row['success']
        x = row['x']
        y = row['y']
        z = row['z']
        latency = row['latency']
        timestamp = row['timestamp']

        if tag_id not in distances:
            distances[tag_id] = 0

        if success:
            success_count += 1
            moving_count += 1

            if distances[tag_id] == 0:
                distances[tag_id] = calculate_distance(0, 0, 0, x, y, z)
            else:
                previous_row = df.iloc[index - 1]
                previous_x = previous_row['x']
                previous_y = previous_row['y']
                previous_z = previous_row['z']
                distances[tag_id] += calculate_distance(previous_x, previous_y, previous_z, x, y, z)

            # Check if any other tags are in the same place or close proximity
            for other_tag_id, other_distance in distances.items():
                if other_tag_id != tag_id:
                    current_distance = calculate_distance(x, y, z, 0, 0, 0)
                    if current_distance - other_distance <= 10:  # Adjust the threshold as needed
                        if tag_id in same_place_tags:
                            same_place_tags[tag_id].append((other_tag_id, timestamp))
                        else:
                            same_place_tags[tag_id] = [(other_tag_id, timestamp)]

        else:
            failure_count += 1

    total_entries = len(df)

    total_time_seconds = final_timestamp - initial_timestamp  # Total time performed in seconds
    total_time_minutes = total_time_seconds / 60  # Convert to minutes

    print("Data Analysis Results:")
    print("----------------------")
    print(f"Total unique tagIds: {unique_tag_ids}")
    print(f"Total entries: {total_entries}")
    print(f"Success count: {success_count} ({(success_count / total_entries) * 100:.2f}%)")
    print(f"Failure count: {failure_count} ({(failure_count / total_entries) * 100:.2f}%)")
    print(f"Total time performed: {total_time_minutes:.2f} minutes")
    print(f"Initial time: {datetime.datetime.fromtimestamp(initial_timestamp)}")
    print(f"Final time: {datetime.datetime.fromtimestamp(final_timestamp)}")

    anchor_coordinates = {
         '5018': {'X': -12728, 'Y': 14229, 'Z': 0},
        '5043': {'X': -7391, 'Y': 12253, 'Z': 0},
        '5077': {'X': 6151, 'Y': 7401, 'Z': 0},
        '5084': {'X': -1714, 'Y': 12224, 'Z': 0},
        '5094': {'X': 2460, 'Y': 9355, 'Z': 0}
    }

    anchor_ids = list(anchor_coordinates.keys())

    anchor_distances = {}  # Dictionary to store distances between anchors

    for anchor_id in anchor_ids:
        anchor_x = anchor_coordinates[anchor_id]['X']
        anchor_y = anchor_coordinates[anchor_id]['Y']
        anchor_z = anchor_coordinates[anchor_id]['Z']

        anchor_distances[anchor_id] = {}

        for other_anchor_id in anchor_ids:
            if other_anchor_id != anchor_id:
                other_anchor_x = anchor_coordinates[other_anchor_id]['X']
                other_anchor_y = anchor_coordinates[other_anchor_id]['Y']
                other_anchor_z = anchor_coordinates[other_anchor_id]['Z']

                distance = calculate_distance(anchor_x, anchor_y, anchor_z, other_anchor_x, other_anchor_y, other_anchor_z)

                anchor_distances[anchor_id][other_anchor_id] = distance

    if anchor_distances:
        print("\nDistances Between Anchors:")
        headers = ['Anchor ID(mm)'] + anchor_ids
        anchor_table = []

        for anchor_id in anchor_ids:
            row = [anchor_id] + [anchor_distances[anchor_id].get(other_anchor_id, 0) for other_anchor_id in anchor_ids]
            anchor_table.append(row)

        print(tabulate(anchor_table, headers=headers, tablefmt="fancy_grid", floatfmt=".2f"))


# ------------------------------------------
# ----------------- PART 3 -----------------
# ------------------------------------------

def plot_anchor_trajectories(anchors):
    plt.figure()

    # Function to connect two anchors
    def connect_anchors(x1, y1, x2, y2):
        plt.plot([x1, x2], [y1, y2], linestyle='dotted', color='gray')

    # Function to connect all anchors to each other
    def connect_all_anchors():
        for anchor_id, anchor_coord in anchors.items():
            anchor_x = anchor_coord['X']
            anchor_y = anchor_coord['Y']

            for other_id, other_coord in anchors.items():
                if anchor_id != other_id:
                    connect_anchors(anchor_x, anchor_y, other_coord['X'], other_coord['Y'])

    # Connect all anchors to each other
    connect_all_anchors()

    # Plot all the anchor positions
    for anchor_id, anchor_coord in anchors.items():
        anchor_x = anchor_coord['X']
        anchor_y = anchor_coord['Y']
        plt.scatter(anchor_x, anchor_y, label=f'Anchor {anchor_id}')

    # Set plot labels and title
    plt.xlabel('X coordinate (mm)')
    plt.ylabel('Y coordinate (mm)')
    plt.title('Trajectory of Anchor Coordinates')

    # Move the legend to a better position (you can adjust the coordinates as needed)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Display the plot
    plt.show()


# ------------------------------------------
# ----------------- PART 4 -----------------
# ------------------------------------------

def plot_individual_tag_trajectories(df):
    # Function to check if a coordinate is missing (NaN)
    def is_coordinate_missing(x, y):
        return pd.isnull(x) or pd.isnull(y)

    # Function to find the last valid coordinates for a tag
    def find_last_valid_coordinates(tag_data):
        last_valid_x = None
        last_valid_y = None

        for i in range(len(tag_data) - 1, -1, -1):
            current_x = tag_data.iloc[i]['x']
            current_y = tag_data.iloc[i]['y']

            if not is_coordinate_missing(current_x, current_y):
                last_valid_x = current_x
                last_valid_y = current_y
                break

        return last_valid_x, last_valid_y

    # Get unique tag IDs
    unique_tag_ids = df['tagId'].unique()

    # Iterate over the unique tag IDs
    for tag_id in unique_tag_ids:
        # Filter data for the current tag
        tag_data = df[df['tagId'] == tag_id]

        # Get X, Y coordinates, timestamps, and success values
        x_coords = tag_data['x'].values
        y_coords = tag_data['y'].values
        timestamps = tag_data['timestamp'].values
        success_values = tag_data['success'].values

        # Find the first valid coordinate for initial point
        initial_x = None
        initial_y = None
        for i in range(len(x_coords)):
            if not is_coordinate_missing(x_coords[i], y_coords[i]):
                initial_x = x_coords[i]
                initial_y = y_coords[i]
                break

        # Find the last valid coordinate for final point
        last_x, last_y = find_last_valid_coordinates(tag_data)

        # Create a new plot for the current tag
        fig, ax = plt.subplots(figsize=(16, 12))  # Increase the figure size

        # Set grid off
        ax.grid(False)

        # Initialize previous coordinates and success value
        prev_valid_x = initial_x
        prev_valid_y = initial_y

        # Iterate over the points
        for i in range(len(x_coords)):
            current_x = x_coords[i]
            current_y = y_coords[i]
            current_success = success_values[i]

            if not is_coordinate_missing(current_x, current_y):
                if not is_coordinate_missing(prev_valid_x, prev_valid_y):
                    if success_values[i] and success_values[i - 1]:
                        ax.plot([prev_valid_x, current_x], [prev_valid_y, current_y], 'b-', linewidth=0.5)
                        # Add an arrow pointing from prev_valid to current point
                        arrow_props = dict(arrowstyle='->', color='blue', lw=0.5)
                        ax.annotate('', xy=(current_x, current_y), xytext=(prev_valid_x, prev_valid_y), arrowprops=arrow_props)
                    else:
                        ax.plot([prev_valid_x, current_x], [prev_valid_y, current_y], 'g--', linewidth=0.5)
                        # Add an arrow pointing from prev_valid to current point for dotted lines
                        arrow_props = dict(arrowstyle='->', color='green', linestyle='dotted', lw=0.5)
                        ax.annotate('', xy=(current_x, current_y), xytext=(prev_valid_x, prev_valid_y), arrowprops=arrow_props)

                prev_valid_x = current_x
                prev_valid_y = current_y

            # Plot the valid points with '*' markers
            if not is_coordinate_missing(current_x, current_y):
                ax.scatter(current_x, current_y, marker='*', color='black', s=50)

        # Highlight the initial and final coordinates if available
        if not is_coordinate_missing(initial_x, initial_y):
            ax.scatter(initial_x, initial_y, color='green', label='Initial', s=200)

        if not is_coordinate_missing(last_x, last_y):
            ax.scatter(last_x, last_y, color='red', label='Final', s=200)

        # Set plot labels and title with larger font size
        ax.set_xlabel('X coordinate', fontsize=14)
        ax.set_ylabel('Y coordinate', fontsize=14)
        ax.set_title(f'Trajectory of Tag {tag_id}', fontsize=16)

        # Add legend with larger font size
        ax.legend(fontsize=12)

        # Display the plot
        plt.show()


# ------------------------------------------
# ----------------- PART 5 -----------------
# ------------------------------------------

def plot_individual_tag_trajectories_wrt_anchors(df, anchors):
    # Function to check if a coordinate is missing (NaN)
    def is_coordinate_missing(x, y):
        return pd.isnull(x) or pd.isnull(y)

    # Get unique tag IDs
    unique_tag_ids = df['tagId'].unique()

    # Function to find the last valid coordinates for a tag
    def find_last_valid_coordinates(tag_data):
        last_valid_x = None
        last_valid_y = None

        for i in range(len(tag_data) - 1, -1, -1):
            current_x = tag_data.iloc[i]['x']
            current_y = tag_data.iloc[i]['y']

            if not is_coordinate_missing(current_x, current_y):
                last_valid_x = current_x
                last_valid_y = current_y
                break

        return last_valid_x, last_valid_y

    # Color dictionary for the anchors
    anchor_colors = {'5018': 'fuchsia', '5043': 'coral', '5077': 'orange', '5084': 'purple', '5094': 'green'}

    # Iterate over the unique tag IDs
    for tag_id in unique_tag_ids:
        # Filter data for the current tag
        tag_data = df[df['tagId'] == tag_id]

        # Get X, Y coordinates, timestamps, and success values
        x_coords = tag_data['x'].values
        y_coords = tag_data['y'].values
        timestamps = tag_data['timestamp'].values
        success_values = tag_data['success'].values

        # Find the first valid coordinate for the initial point
        initial_x = None
        initial_y = None
        for i in range(len(x_coords)):
            if not is_coordinate_missing(x_coords[i], y_coords[i]):
                initial_x = x_coords[i]
                initial_y = y_coords[i]
                break

        # Find the last valid coordinate for the final point
        last_x, last_y = find_last_valid_coordinates(tag_data)

        # Create a new plot for the current tag
        fig, ax = plt.subplots(figsize=(16, 12))  # Increase the figure size

        # Set grid off
        ax.grid(False)

        # Initialize previous coordinates and success value
        prev_valid_x = initial_x
        prev_valid_y = initial_y

        # Iterate over the points
        for i in range(len(x_coords)):
            current_x = x_coords[i]
            current_y = y_coords[i]
            current_success = success_values[i]

            if not is_coordinate_missing(current_x, current_y):
                if not is_coordinate_missing(prev_valid_x, prev_valid_y):
                    if success_values[i] and success_values[i - 1]:
                        ax.plot([prev_valid_x, current_x], [prev_valid_y, current_y], 'b-', linewidth=0.5)
                        # Add an arrow pointing from prev_valid to the current point
                        arrow_props = dict(arrowstyle='->', color='blue', lw=0.5)
                        ax.annotate('', xy=(current_x, current_y), xytext=(prev_valid_x, prev_valid_y), arrowprops=arrow_props)
                    else:
                        ax.plot([prev_valid_x, current_x], [prev_valid_y, current_y], 'g--', linewidth=0.5)
                        # Add an arrow pointing from prev_valid to the current point for dotted lines
                        arrow_props = dict(arrowstyle='->', color='green', linestyle='dotted', lw=0.5)
                        ax.annotate('', xy=(current_x, current_y), xytext=(prev_valid_x, prev_valid_y), arrowprops=arrow_props)

                prev_valid_x = current_x
                prev_valid_y = current_y

            # Plot the valid points with '*' markers
            if not is_coordinate_missing(current_x, current_y):
                ax.scatter(current_x, current_y, marker='*', color='black', s=50)

        # Get the respective anchor for the current tag (convert tag_id to string)
        anchor_data = anchors.get(str(tag_id), {'X': 0, 'Y': 0})  # Provide default coordinates (0, 0) if tag_id is not found
        anchor_x = anchor_data['X']
        anchor_y = anchor_data['Y']

        # Plot all four anchors on the same graph for each tag
        for anchor_id, anchor_coords in anchors.items():
            anchor_color = anchor_colors.get(anchor_id, 'black')  # Use black color if anchor_id is not found in the color dictionary
            ax.scatter(anchor_coords['X'], anchor_coords['Y'], marker='D', color=anchor_color, label=f'Anchor {anchor_id}', s=100)
            ax.text(anchor_coords['X'], anchor_coords['Y'], 'A', color=anchor_color, fontsize=12, ha='center', va='center')

        # Highlight the initial and final coordinates if available
        if not is_coordinate_missing(initial_x, initial_y):
            ax.scatter(initial_x, initial_y, color='green', label='Initial', s=200)

        if not is_coordinate_missing(last_x, last_y):
            ax.scatter(last_x, last_y, color='red', label='Final', s=200)

        # Set plot labels and title with larger font size, including the respective anchor
        ax.set_xlabel('X coordinate', fontsize=14)
        ax.set_ylabel('Y coordinate', fontsize=14)
        ax.set_title(f'Trajectory of Tag {tag_id} with Respect to Anchor', fontsize=16)

        # Add legend with larger font size
        ax.legend(fontsize=12)

        # Display the plot
        plt.show()


# ------------------------------------------
# ----------------- PART 6 -----------------
# ------------------------------------------
def plot_tag_pair_trajectories(df):

    # Function to check if a coordinate is missing (NaN)
    def is_coordinate_missing(x, y):
        return pd.isnull(x) or pd.isnull(y)

    # Get unique tag IDs
    unique_tag_ids = df['tagId'].unique()

    # Function to find the last valid coordinates for a tag
    def find_last_valid_coordinates(tag_data):
        last_valid_x = None
        last_valid_y = None

        for i in range(len(tag_data) - 1, -1, -1):
            current_x = tag_data.iloc[i]['x']
            current_y = tag_data.iloc[i]['y']

            if not is_coordinate_missing(current_x, current_y):
                last_valid_x = current_x
                last_valid_y = current_y
                break

        return last_valid_x, last_valid_y

    # Group unique tag IDs in pairs
    tag_pairs = [(tag_id1, tag_id2) for i, tag_id1 in enumerate(unique_tag_ids) for tag_id2 in unique_tag_ids[i+1:]]

    # Keep track of plotted tag pairs
    plotted_pairs = set()

    # Iterate over the tag pairs
    for tag_id1, tag_id2 in tag_pairs:
        # Check if the pair has been plotted already in reverse order
        if (tag_id2, tag_id1) in plotted_pairs:
            continue

        # Filter data for the first tag
        tag_data1 = df[df['tagId'] == tag_id1]

        # Filter data for the second tag
        tag_data2 = df[df['tagId'] == tag_id2]

        # Get X, Y coordinates, timestamps, and success values for both tags
        x_coords1 = tag_data1['x'].values
        y_coords1 = tag_data1['y'].values
        timestamps1 = tag_data1['timestamp'].values
        success_values1 = tag_data1['success'].values

        x_coords2 = tag_data2['x'].values
        y_coords2 = tag_data2['y'].values
        timestamps2 = tag_data2['timestamp'].values
        success_values2 = tag_data2['success'].values

        # Find the first valid coordinate for the initial point of both tags
        initial_x1 = None
        initial_y1 = None
        for i in range(len(x_coords1)):
            if not is_coordinate_missing(x_coords1[i], y_coords1[i]):
                initial_x1 = x_coords1[i]
                initial_y1 = y_coords1[i]
                break

        initial_x2 = None
        initial_y2 = None
        for i in range(len(x_coords2)):
            if not is_coordinate_missing(x_coords2[i], y_coords2[i]):
                initial_x2 = x_coords2[i]
                initial_y2 = y_coords2[i]
                break

        # Find the last valid coordinate for the final point of both tags
        last_x1, last_y1 = find_last_valid_coordinates(tag_data1)
        last_x2, last_y2 = find_last_valid_coordinates(tag_data2)

        # Create a new plot for the current tag pair
        fig, ax = plt.subplots(figsize=(16, 12))  # Increase the figure size

        # Set grid off
        ax.grid(False)

        # Iterate over the points of the first tag
        prev_valid_x1 = initial_x1
        prev_valid_y1 = initial_y1
        for i in range(len(x_coords1)):
            current_x1 = x_coords1[i]
            current_y1 = y_coords1[i]
            current_success1 = success_values1[i]

            if not is_coordinate_missing(current_x1, current_y1):
                if not is_coordinate_missing(prev_valid_x1, prev_valid_y1):
                    if success_values1[i] and success_values1[i - 1]:
                        ax.plot([prev_valid_x1, current_x1], [prev_valid_y1, current_y1], 'b-', linewidth=0.5)
                        # Add an arrow pointing from prev_valid to current point
                        arrow_props = dict(arrowstyle='->', color='blue', lw=0.5)
                        ax.annotate('', xy=(current_x1, current_y1), xytext=(prev_valid_x1, prev_valid_y1), arrowprops=arrow_props)
                    else:
                        ax.plot([prev_valid_x1, current_x1], [prev_valid_y1, current_y1], 'g--', linewidth=0.5)
                        # Add an arrow pointing from prev_valid to current point for dotted lines
                        arrow_props = dict(arrowstyle='->', color='green', linestyle='dotted', lw=0.5)
                        ax.annotate('', xy=(current_x1, current_y1), xytext=(prev_valid_x1, prev_valid_y1), arrowprops=arrow_props)

                prev_valid_x1 = current_x1
                prev_valid_y1 = current_y1

            # Plot the valid points with '*' markers
            if not is_coordinate_missing(current_x1, current_y1):
                ax.scatter(current_x1, current_y1, marker='*', color='black', s=100)

        # Iterate over the points of the second tag
        prev_valid_x2 = initial_x2
        prev_valid_y2 = initial_y2
        for i in range(len(x_coords2)):
            current_x2 = x_coords2[i]
            current_y2 = y_coords2[i]
            current_success2 = success_values2[i]

            if not is_coordinate_missing(current_x2, current_y2):
                if not is_coordinate_missing(prev_valid_x2, prev_valid_y2):
                    if success_values2[i] and success_values2[i - 1]:
                        ax.plot([prev_valid_x2, current_x2], [prev_valid_y2, current_y2], 'r-', linewidth=0.5)
                        # Add an arrow pointing from prev_valid to current point
                        arrow_props = dict(arrowstyle='->', color='red', lw=0.5)
                        ax.annotate('', xy=(current_x2, current_y2), xytext=(prev_valid_x2, prev_valid_y2), arrowprops=arrow_props)
                    else:
                        ax.plot([prev_valid_x2, current_x2], [prev_valid_y2, current_y2], 'm--', linewidth=0.5)
                        # Add an arrow pointing from prev_valid to current point for dotted lines
                        arrow_props = dict(arrowstyle='->', color='magenta', linestyle='dotted', lw=0.5)
                        ax.annotate('', xy=(current_x2, current_y2), xytext=(prev_valid_x2, prev_valid_y2), arrowprops=arrow_props)

                prev_valid_x2 = current_x2
                prev_valid_y2 = current_y2

            # Plot the valid points with '*' markers
            if not is_coordinate_missing(current_x2, current_y2):
                ax.scatter(current_x2, current_y2, marker='*', color='black', s=100)

        # Highlight the the initial and final coordinates if available for both tags
        if not is_coordinate_missing(initial_x1, initial_y1):
            ax.scatter(initial_x1, initial_y1, marker='o', edgecolors='green', facecolors='none', label=f'Initial Tag {tag_id1}', s=200)

        if not is_coordinate_missing(last_x1, last_y1):
            ax.scatter(last_x1, last_y1, marker='o', edgecolors='red', facecolors='none', label=f'Final Tag {tag_id1}', s=200)

        if not is_coordinate_missing(initial_x2, initial_y2):
            ax.scatter(initial_x2, initial_y2, marker='o', edgecolors='cyan', facecolors='none', label=f'Initial Tag {tag_id2}', s=200)

        if not is_coordinate_missing(last_x2, last_y2):
            ax.scatter(last_x2, last_y2, marker='o', edgecolors='blue', facecolors='none', label=f'Final Tag {tag_id2}', s=200)

        # Add legend with larger font size
        ax.legend(fontsize=12)

        # Set plot labels and title with larger font size
        ax.set_xlabel('X coordinate', fontsize=14)
        ax.set_ylabel('Y coordinate', fontsize=14)
        ax.set_title(f'Trajectories of Tags {tag_id1} and {tag_id2}', fontsize=16)

        # Add the pair to plotted_pairs to avoid duplication
        plotted_pairs.add((tag_id1, tag_id2))

        # Display the plot
        plt.show()


# ------------------------------------------
# ----------------- PART 7 -----------------
# ------------------------------------------

def plot_all_tags_trajectory(df):
    # Function to check if a coordinate is missing (NaN)
    def is_coordinate_missing(x, y):
        return pd.isnull(x) or pd.isnull(y)

    # Function to find the last valid coordinates for a tag
    def find_last_valid_coordinates(tag_data):
        last_valid_x = None
        last_valid_y = None

        for i in range(len(tag_data) - 1, -1, -1):
            current_x = tag_data.iloc[i]['x']
            current_y = tag_data.iloc[i]['y']

            if not is_coordinate_missing(current_x, current_y):
                last_valid_x = current_x
                last_valid_y = current_y
                break

        return last_valid_x, last_valid_y

    # Get unique tag IDs
    unique_tag_ids = df['tagId'].unique()

    # Create a new plot for all tags
    fig, ax = plt.subplots(figsize=(16, 12))  # Increase the figure size

    # Set grid off
    ax.grid(False)

    # Specify custom colors for each tag
    custom_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'black', 'olive', 'teal']

    # Iterate over the unique tag IDs and their corresponding custom colors
    for i, tag_id in enumerate(unique_tag_ids):
        # Filter data for the current tag
        tag_data = df[df['tagId'] == tag_id]

        # Get X, Y coordinates, timestamps, and success values
        x_coords = tag_data['x'].values
        y_coords = tag_data['y'].values
        timestamps = tag_data['timestamp'].values
        success_values = tag_data['success'].values

        # Find the first valid coordinate for initial point
        initial_x = None
        initial_y = None
        for i in range(len(x_coords)):
            if not is_coordinate_missing(x_coords[i], y_coords[i]):
                initial_x = x_coords[i]
                initial_y = y_coords[i]
                break

        # Find the last valid coordinate for final point
        last_x, last_y = find_last_valid_coordinates(tag_data)

        # Initialize previous coordinates and success value
        prev_valid_x = initial_x
        prev_valid_y = initial_y

        # Iterate over the points
        for i in range(len(x_coords)):
            current_x = x_coords[i]
            current_y = y_coords[i]
            current_success = success_values[i]

            if not is_coordinate_missing(current_x, current_y):
                if not is_coordinate_missing(prev_valid_x, prev_valid_y):
                    if success_values[i] and success_values[i - 1]:
                        ax.plot([prev_valid_x, current_x], [prev_valid_y, current_y], color=custom_colors[i % len(custom_colors)], linewidth=0.5)
                        # Add an arrow pointing from prev_valid to current point
                        arrow_props = dict(arrowstyle='->', color=custom_colors[i % len(custom_colors)], lw=0.5)
                        ax.annotate('', xy=(current_x, current_y), xytext=(prev_valid_x, prev_valid_y), arrowprops=arrow_props)
                    else:
                        ax.plot([prev_valid_x, current_x], [prev_valid_y, current_y], color=custom_colors[i % len(custom_colors)], linestyle='dotted', linewidth=0.5)
                        # Add an arrow pointing from prev_valid to current point for dotted lines
                        arrow_props = dict(arrowstyle='->', color=custom_colors[i % len(custom_colors)], linestyle='dotted', lw=0.5)
                        ax.annotate('', xy=(current_x, current_y), xytext=(prev_valid_x, prev_valid_y), arrowprops=arrow_props)

                prev_valid_x = current_x
                prev_valid_y = current_y

        # Plot the valid points with '*' markers and create a separate legend entry for each tag
        ax.scatter(x_coords, y_coords, marker='*', color=custom_colors[i % len(custom_colors)], s=100, label=f'Tag {tag_id}')

    # Set plot labels and title with larger font size
    ax.set_xlabel('X coordinate', fontsize=14)
    ax.set_ylabel('Y coordinate', fontsize=14)
    ax.set_title('Trajectories of All Tags', fontsize=16)

    # Add legend with larger font size, displaying a separate legend entry for each tag
    ax.legend(fontsize=12, scatterpoints=1)

    # Display the plot
    plt.show()

# ------------------------------------------
# ----------------- PART 8 -----------------
# ------------------------------------------

def plot_all_tags_trajectory_points(df):
    # Function to check if a coordinate is missing (NaN)
    def is_coordinate_missing(x, y):
        return pd.isnull(x) or pd.isnull(y)

    # Function to find the last valid coordinates for a tag
    def find_last_valid_coordinates(tag_data):
        last_valid_x = None
        last_valid_y = None

        for i in range(len(tag_data) - 1, -1, -1):
            current_x = tag_data.iloc[i]['x']
            current_y = tag_data.iloc[i]['y']

            if not is_coordinate_missing(current_x, current_y):
                last_valid_x = current_x
                last_valid_y = current_y
                break

        return last_valid_x, last_valid_y

    # Get unique tag IDs
    unique_tag_ids = df['tagId'].unique()

    # Create a new plot for all tags
    fig, ax = plt.subplots(figsize=(16, 12))  # Increase the figure size

    # Set grid off
    ax.grid(False)

    # Specify custom colors for each tag
    custom_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'lime']

    # Iterate over the unique tag IDs and their corresponding custom colors
    for i, tag_id in enumerate(unique_tag_ids):
        # Filter data for the current tag
        tag_data = df[df['tagId'] == tag_id]

        # Get X, Y coordinates, timestamps, and success values
        x_coords = tag_data['x'].values
        y_coords = tag_data['y'].values
        timestamps = tag_data['timestamp'].values
        success_values = tag_data['success'].values

        # Find the first valid coordinate for initial point
        initial_x = None
        initial_y = None
        for i in range(len(x_coords)):
            if not is_coordinate_missing(x_coords[i], y_coords[i]):
                initial_x = x_coords[i]
                initial_y = y_coords[i]
                break

        # Find the last valid coordinate for final point
        last_x, last_y = find_last_valid_coordinates(tag_data)

        # Plot the points without lines or asterisks with smaller size
        ax.scatter(x_coords, y_coords, color=custom_colors[i % len(custom_colors)], s=50, label=f'Tag {tag_id}')

    # Set plot labels and title with larger font size
    ax.set_xlabel('X coordinate', fontsize=14)
    ax.set_ylabel('Y coordinate', fontsize=14)
    ax.set_title('Trajectories of All Tags', fontsize=16)

    # Add legend with larger font size, displaying a separate legend entry for each tag
    ax.legend(fontsize=12, scatterpoints=1)

    # Display the plot
    plt.show()

# ------------------------------------------
# ----------------- PART 13 -----------------
# ------------------------------------------

def tag_pair_distances(df):
    # Create output directory
    output_folder = 'outputs'
    os.makedirs(output_folder, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    output_file = os.path.join(output_folder, f'tag_pair_distances_{time_stamp}.csv')

    # Function to calculate distance between two 2D points (x, y)
    def calculate_distance(x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # Filter the required columns and remove rows with missing coordinates
    df = df[['tagId', 'time', 'x', 'y']].copy()  # Make a copy to avoid SettingWithCopyWarning
    df.dropna(subset=['x', 'y'], inplace=True)

    # Drop milliseconds from the 'time' column
    df['time'] = df['time'].str.replace(r'\.\d+$', '', regex=True)

    # Initialize an empty list to store the results
    output_data = []

    # Group by 'time' and calculate the distances for each timestamp
    for timestamp, group in df.groupby('time'):
        distances = {}
        tag_ids = group['tagId'].unique()

        # Calculate distances for unique tag pairs only once
        for i, tag1_id in enumerate(tag_ids):
            for j, tag2_id in enumerate(tag_ids):
                if i < j:  # Avoid redundant pairs (tag1 - tag2 and tag2 - tag1)
                    tag1_data = group[group['tagId'] == tag1_id]
                    tag2_data = group[group['tagId'] == tag2_id]

                    if len(tag1_data) > 0 and len(tag2_data) > 0:
                        distance = calculate_distance(tag1_data['x'].iloc[0], tag1_data['y'].iloc[0], tag2_data['x'].iloc[0], tag2_data['y'].iloc[0])
                        # Include the tag IDs in the output as "tag1_id - tag2_id"
                        distances[f'{tag1_id}-{tag2_id}'] = distance

        # Append the distances for this timestamp to the output list
        output_data.append({'time': timestamp, **distances})

    # Convert the list of dictionaries to a DataFrame
    output_df = pd.DataFrame(output_data)

    # Save the DataFrame to a CSV file in the specified output folder
    output_df.to_csv(output_file, index=False)

    print(f"Distance values saved to '{output_file}'.")

    return output_df