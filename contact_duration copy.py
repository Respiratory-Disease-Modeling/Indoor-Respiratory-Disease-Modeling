import pandas as pd
import numpy as np
import os

# Configurations
feet = 6
time_threshold = 1
input_file = 'pizzaData.csv'

# Create output directory
output_directory = f'all_tags_{feet}_feet_{time_threshold}_time'
os.makedirs(output_directory, exist_ok=True)

# Create dataframe
df = pd.read_csv(input_file)
all_tags = df['tagId'].unique().tolist()
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')

# Drop duplicates based on time column
df = df.drop_duplicates(subset=['time', 'tagId'])

for tag in all_tags:
    selected_tag = str(tag)
    # Pivot the DataFrame
    flatten_df = df.pivot(index='time', columns='tagId')

    # Flatten the column names
    flatten_df.columns = [f'{col[1]}_{col[0]}' for col in flatten_df.columns]
    flatten_df = flatten_df.reset_index()
    flatten_df['time'] = flatten_df['time'].dt.strftime('%H:%M:%S')

    # columns with co-ordinates
    column_names = flatten_df.columns
    x_columns = [col for col in column_names if col.endswith('_x')]
    y_columns = [col for col in column_names if col.endswith('_y')]
    z_columns = [col for col in column_names if col.endswith('_z')]

    # Create a new dataframe to store distances
    distance_df = pd.DataFrame({'time': flatten_df['time']})

    # Selected tags x, y and z
    x_selected = flatten_df[selected_tag + '_x'].values
    y_selected = flatten_df[selected_tag + '_y'].values
    z_selected = flatten_df[selected_tag + '_z'].values

    # Calculate distances and create new columns in the DataFrame
    for col in x_columns:
        tag = col.split('_')[0]
        x = flatten_df[col].values
        y = flatten_df[col.replace('_x', '_y')].values
        z = flatten_df[col.replace('_x', '_z')].values
        distances = np.sqrt((x - x_selected)**2 + (y - y_selected)**2 + (z - z_selected)**2)
        # Convert distances to feet
        distances = np.round(distances / 304.8, 3)
        distance_df['distance_' + tag] = distances

    # Calculate duration

    # Convert the 'time' column to datetime objects
    distance_df['time'] = pd.to_datetime(distance_df['time'], format='%H:%M:%S', errors='coerce')

    # Initialize lists to store the new data
    result_data = []

    # Filter columns with names starting with 'distance_'
    distance_columns = distance_df.filter(like='distance_')

    for col_name in distance_columns.columns:

        if col_name == f'distance_{selected_tag}':
            continue

        # Add x, y and z co-ordinates in the distance_df
        current_tag = col_name[len("distance_"):]
        distance_df['x'] = flatten_df[current_tag + '_x'].copy()
        distance_df['y'] = flatten_df[current_tag + '_y'].copy()
        distance_df['z'] = flatten_df[current_tag + '_z'].copy()
        #print(distance_df.head(5))

        # Initialize variables to keep track of consecutive rows
        start_time = None
        end_time = None
        x_agg = 0
        y_agg = 0
        z_agg = 0
        durations = []
        start_times = []
        end_times = []
        x_avg = []
        y_avg = []
        z_avg = []

        for index, row in distance_df.iterrows():
            if not pd.isnull(row[col_name]):
                if start_time is None and row[col_name] <= feet:
                    start_time = row['time']
                    #if row['x'] is not None:
                    x_agg =  row['x']
                    y_agg =  row['y']
                    z_agg =  row['z']
                if row[col_name] <= feet:
                    end_time = row['time']
                    x_agg = x_agg + row['x']
                    y_agg = y_agg + row['y']
                    z_agg = z_agg + row['z']
            elif start_time is not None:
                duration_seconds = (end_time - start_time).total_seconds() + 1
                durations.append(duration_seconds)
                x_avg.append(x_agg/ duration_seconds)
                y_avg.append(y_agg/ duration_seconds)
                z_avg.append(z_agg/ duration_seconds)
                start_times.append(start_time.strftime('%H:%M:%S'))
                end_times.append(end_time.strftime('%H:%M:%S'))
                x_agg = 0
                y_agg = 0
                z_agg = 0
                start_time = None
                end_time = None

        # Create a new DataFrame with the calculated data for the current distance column.
        # len(durations), because all columns in df should be of same length.
        new_df = pd.DataFrame({'tag1': [selected_tag] * len(durations), 'tag2': [current_tag] * len(durations), 'duration_seconds': durations, 'start_time': start_times, 'end_time': end_times, 'x': x_avg, 'y': y_avg})
        x_avg = []
        y_avg = []
        z_avg = []
        # Round of the co-ordinates
        new_df['x'] = new_df['x'].round().astype(int)
        new_df['y'] = new_df['y'].round().astype(int)
        #print(new_df.head(5))
        # Store the new DataFrame along with the column name in result_data
        result_data.append((col_name, new_df))

    # # Create output directory
    # output_directory = f'{selected_tag}_{feet}_feet_{time_threshold}_time'
    # os.makedirs(output_directory, exist_ok=True)

    finals_dfs = []
    # Save the new DataFrames to separate CSV files
    for col_name, new_df in result_data:
        output_file_path = os.path.join(output_directory, f'{col_name}.csv')
        new_df = new_df[new_df['duration_seconds'] >= time_threshold]
        finals_dfs.append(new_df)
        # new_df.to_csv(output_file_path, index=False)

    merged_df = pd.concat(finals_dfs, axis=0, ignore_index=True)
    output_file_path = os.path.join(output_directory, f'distance_all_tags.csv')
    if os.path.exists(output_file_path):
        merged_df.to_csv(output_file_path, mode='a', header=False, index=False)
    else:
        merged_df.to_csv(output_file_path, index=False)
 