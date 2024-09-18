import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import matplotlib.cm as cm

from matplotlib.backends.backend_pdf import PdfPages

# ------------------------------------------
# ----------------- PART 9 -----------------
# ------------------------------------------

def process_data(feet, time_threshold, input_file):

    # Create output directory
    output_directory = 'outputs'
    os.makedirs(output_directory, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_path = os.path.join(output_directory, f'distance_all_tags_{time_stamp}.csv')    

    # Create dataframe
    df = pd.read_csv(input_file)
    all_tags = df['tagId'].unique().tolist()
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')

    # Drop duplicates based on time column
    df = df.drop_duplicates(subset=['time', 'tagId'])

    mega_df = pd.DataFrame()

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
        x = [col for col in column_names if col.endswith('x')]
        y = [col for col in column_names if col.endswith('_y')]
        z = [col for col in column_names if col.endswith('_z')]

        # Create a new dataframe to store distances
        distance_df = pd.DataFrame({'time': flatten_df['time']})

        # Selected tags x, y and z
        x_selected = flatten_df[selected_tag + '_x'].values
        y_selected = flatten_df[selected_tag + '_y'].values
        z_selected = flatten_df[selected_tag + '_z'].values

        # Calculate distances and create new columns in the DataFrame
        for col in x:
            tag = col.split('_')[0]
            x = flatten_df[col].values
            y = flatten_df[col.replace('_x', '_y')].values
            z = flatten_df[col.replace('_x', '_z')].values
            distances = np.sqrt((x - x_selected)**2 + (y - y_selected)**2 + (z - z_selected)**2)
            # Convert distances to feet
            #distances = np.round(distances / 304.8, 3)
            distances = np.round(distances / 12, 3)
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

            # Store the new DataFrame along with the column name in result_data
            result_data.append((col_name, new_df))

        finals_dfs = []
        # Save the new DataFrames to separate CSV files
        for col_name, new_df in result_data:
            new_df = new_df[new_df['duration_seconds'] >= time_threshold]
            finals_dfs.append(new_df)

        merged_df = pd.concat(finals_dfs, axis=0, ignore_index=True)
        mega_df = pd.concat([mega_df, merged_df], ignore_index=True)
        if os.path.exists(output_file_path):
            merged_df.to_csv(output_file_path, mode='a', header=False, index=False)
        else:
            merged_df.to_csv(output_file_path, index=False)

    print("Process completed successfully.")

    # Sorting by time before returning
    mega_df.sort_values(by='start_time', inplace=True)
    return mega_df


# ------------------------------------------
# ----------------- PART 10 ----------------
# ------------------------------------------

def contact_intensity(df):
    # Create output directory
    output_directory = 'outputs'
    os.makedirs(output_directory, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    output_file = os.path.join(output_directory, f'contact_intensity_{time_stamp}.txt')

    # Calculate the total number of contacts for each tag
    contact_counts = df.groupby('tag1').size().reset_index(name='total_contacts')

    # Calculate the total number of contacts for all tags divided by 2
    total_contacts = contact_counts['total_contacts'].sum() // 2

    # Calculate the contact rate as a percentage
    contact_counts['contact_rate (%)'] = (contact_counts['total_contacts'] / total_contacts) * 100

    # Sort contact counts in ascending order
    contact_counts_sorted = contact_counts.sort_values(by='total_contacts', ascending=True)

    # Redirect print output to a file
    original_stdout = sys.stdout
    
    with open(output_file, 'w') as f:
        sys.stdout = f
        # Print the sorted contact counts and contact rates for each tag, and total contacts
        print(contact_counts_sorted)
        print(f'Total Contacts for All Tags: {total_contacts}')

def plot_contact_density(all_distances):
    # Create output directory
    output_directory = 'outputs'
    os.makedirs(output_directory, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    output_pdf_file = os.path.join(output_directory, f'contact_density_{time_stamp}.pdf')

    # Calculate contact duration summary statistics
    mean_duration = all_distances['duration_seconds'].mean()
    std_duration = all_distances['duration_seconds'].std()

    # Filter data for durations greater than 5 seconds
    durations_greater_than_5 = all_distances[all_distances['duration_seconds'] > 5]

    # Plot density estimation
    sns.kdeplot(all_distances['duration_seconds'], label='All Durations')
    sns.kdeplot(all_distances[all_distances['duration_seconds'] > 0]['duration_seconds'], label='Valid Durations (> 0)')
    sns.kdeplot(all_distances[all_distances['duration_seconds'] > 1]['duration_seconds'], label='Valid Durations (> 1)')
    sns.kdeplot(all_distances[all_distances['duration_seconds'] > 2]['duration_seconds'], label='Valid Durations (> 2)')
    sns.kdeplot(all_distances[all_distances['duration_seconds'] > 5]['duration_seconds'], label='Valid Durations (> 5)')
    sns.kdeplot(all_distances[all_distances['duration_seconds'] > 10]['duration_seconds'], label='Valid Durations (> 10)')
    sns.kdeplot(all_distances[all_distances['duration_seconds'] > 60]['duration_seconds'], label='Valid Durations (> 60)')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Density')
    plt.title('Contact Duration Density Plot')
    plt.legend()
    # plt.show()

    # Save the plot to a PDF file
    plt.savefig(output_pdf_file, format='pdf', bbox_inches='tight')
    plt.close()

    # Display summary statistics and count of durations > 5 seconds
    print(f"Mean Duration: {mean_duration}")
    print(f"Standard Deviation: {std_duration}")
    print(f"Count of Durations > 5 seconds: {len(durations_greater_than_5)}")

# ------------------------------------------
# ----------------- PART 12 ----------------
# ------------------------------------------

def generate_distance_matrices(df):
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder= f'outputs/distance_matrices_{time_stamp}'

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)        

    # Function to calculate distance between two 3D points (x, y, z)
    def calculate_distance(x1, y1, z1, x2, y2, z2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5

    # Filter the required columns and remove rows with missing coordinates
    df = df[['tagId', 'converted_date', 'converted_time', 'x', 'y', 'z']].copy()
    df.dropna(subset=['x', 'y', 'z'], inplace=True)

    # Combine 'converted_date' and 'converted_time' columns into a single datetime column
    df.loc[:, 'timestamp'] = pd.to_datetime(df['converted_date'] + ' ' + df['converted_time'])

    # Map the DataFrame tag IDs to corresponding matrix indices
    tag_id_to_index = {tag_id: index for index, tag_id in enumerate(df['tagId'].unique())}

    # Group by timestamp and calculate the distance matrices for each 5-second timeframe
    time_interval = 5  # Set your desired time interval in seconds

    # Define the maximum value for the color scale
    vmax = 8000

    for timestamp, group in df.groupby(pd.Grouper(key='timestamp', freq=f'{time_interval}S')):
        # Create a 9x9 distance matrix for each 5-second timeframe
        num_tags = 9
        distance_matrix = np.zeros((num_tags, num_tags))

        for i in range(num_tags):
            tag1_id = df['tagId'].unique()[i]
            tag1_data = group[group['tagId'] == tag1_id]

            for j in range(i + 1, num_tags):
                tag2_id = df['tagId'].unique()[j]
                tag2_data = group[group['tagId'] == tag2_id]

                if len(tag1_data) > 0 and len(tag2_data) > 0:
                    distance = calculate_distance(tag1_data['x'].iloc[0], tag1_data['y'].iloc[0], tag1_data['z'].iloc[0],
                                                tag2_data['x'].iloc[0], tag2_data['y'].iloc[0], tag2_data['z'].iloc[0])
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance

        # Create a heatmap plot
        plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
        sns.heatmap(distance_matrix, cmap='coolwarm', annot=True, fmt='.2f', cbar=True)
        
        # Set X and Y axis labels to show the actual tag IDs
        tag_ids = [df['tagId'].unique()[i] for i in range(num_tags)]
        plt.xticks(np.arange(num_tags) + 0.5, tag_ids, rotation=90)
        plt.yticks(np.arange(num_tags) + 0.5, tag_ids, rotation=0)

        plt.xlabel('Tag ID')
        plt.ylabel('Tag ID')
        plt.title(f'Distance Matrix at Time: {timestamp}')
        
        # Save the heatmap plot as a PNG file
        output_file = os.path.join(output_folder, f'distance_matrix_{timestamp}.png')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

    print("Distance matrices saved as PNG files in the 'distance_matrices' folder.")


# ------------------------------------------
# ----------------- PART 14 ----------------
# ------------------------------------------

def create_contact_network_longer_duration(df):
    # Create output directory
    output_directory = 'outputs'
    os.makedirs(output_directory, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    output_pdf_file = os.path.join(output_directory, f'contact_network_longer_duration_{time_stamp}.pdf')

    # Convert start_time and end_time columns to datetime objects
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    # Create a directed graph for the contact network
    G = nx.DiGraph()

    # Create a mapping of tags to colors
    unique_tags = set(df['tag1']).union(set(df['tag2']))
    tag_color_map = {tag: plt.cm.tab10.colors[i % 10] for i, tag in enumerate(unique_tags)}

    # List to hold edges with longer durations
    longer_duration_edges = []

    # Iterate through rows in the data and add edges to the graph
    for _, row in df.iterrows():
        tagid1, tagid2, start_time, end_time = row['tag1'], row['tag2'], row['start_time'], row['end_time']
        duration = (end_time - start_time).total_seconds()  # Calculate duration in seconds
        edge_label = f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"
        if duration > 5: # Change value according to the longer duration contacts
            G.add_edge(tagid1, tagid2, label=edge_label)
            longer_duration_edges.append([tagid1, tagid2, start_time, end_time, duration])

    # Create a total contact network plot
    plt.figure(figsize=(12, 9))
    plt.title('Total Contact Trace Network')

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'label')

    # Draw nodes with consistent colors for tags
    node_colors = [tag_color_map[node] for node in G.nodes()]

    # Adjust label positions for better readability
    label_pos = {node: (x, y + 0.05) for node, (x, y) in pos.items()}

    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, font_size=10, font_color='black', font_weight='bold', edge_color='gray', alpha=0.7)
    nx.draw_networkx_edge_labels(G, label_pos, edge_labels=edge_labels, font_size=8)

    # Save the plot to a PDF file
    plt.savefig(output_pdf_file, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"Plot saved as PDF file: {output_pdf_file}")


# ------------------------------------------
# ----------------- PART 15 ----------------
# ------------------------------------------

def create_contact_network_5_sec_duration(df):
    # Create output directory
    output_directory = 'outputs'
    os.makedirs(output_directory, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    output_pdf_file = os.path.join(output_directory, f'contact_network_5_sec_{time_stamp}.pdf')

    # Convert start_time and end_time columns to datetime objects
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    # Create a directed graph for the contact network
    G = nx.DiGraph()

    # Extract the unique time intervals in your data
    time_intervals = sorted(set(df['start_time']))
    time_intervals = [time_intervals[i:i+6] for i in range(0, len(time_intervals), 6)]

    # Create a mapping of tags to colors
    unique_tags = set(df['tag1']).union(set(df['tag2']))
    tag_color_map = {tag: plt.cm.tab10.colors[i % 10] for i, tag in enumerate(unique_tags)}

    # Initialize a PdfPages object to save all plots in a single PDF
    with PdfPages(output_pdf_file) as pdf:
        # Visualize the contact networks for each 5-minute interval
        for i, interval in enumerate(time_intervals):
            plt.figure(figsize=(16, 12))
            plt.title(f'Contact Trace Network for Time Interval {i+1}')

            # Create a subgraph for the current time interval
            subgraph = df[(df['start_time'] >= interval[0]) & (df['start_time'] <= interval[-1])]

            # Iterate through rows in the subgraph
            for _, row in subgraph.iterrows():
                tagid1, tagid2, start_time, end_time = row['tag1'], row['tag2'], row['start_time'], row['end_time']
                edge_label = f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"
                G.add_edge(tagid1, tagid2, label=edge_label)

            pos = nx.spring_layout(G, seed=42)
            edge_labels = nx.get_edge_attributes(G, 'label')

            # Draw nodes with consistent colors for tags
            node_colors = [tag_color_map[node] for node in G.nodes()]

            # Adjust label positions for better readability
            label_pos = {node: (x, y + 0.05) for node, (x, y) in pos.items()}

            nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, font_size=10, font_color='black', font_weight='bold', edge_color='gray', alpha=0.7)
            nx.draw_networkx_edge_labels(G, label_pos, edge_labels=edge_labels, font_size=8)

            # Save the current plot to the PDF
            pdf.savefig()
            plt.close()

            # Clear the graph for the next interval
            G.clear()

    print(f"All plots saved in the PDF file: {output_pdf_file}")


# ------------------------------------------
# ----------------- PART 16 ----------------
# ------------------------------------------

def create_contact_network_for_each_individual_tag(df):
    # Create output directory
    output_directory = 'outputs'
    os.makedirs(output_directory, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    output_pdf_file = os.path.join(output_directory, f'contact_network_for_each_individual_tag_{time_stamp}.pdf')

    all_target_tags = pd.concat([df['tag1'], df['tag2']]).unique()

    # Open a PdfPages object to save all plots in a single PDF
    with PdfPages(output_pdf_file) as pdf:
        for target_tag in all_target_tags:
            # Convert start_time and end_time columns to datetime objects
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])

            # Ensure tag1 and tag2 are strings
            df['tag1'] = df['tag1'].astype(str)
            df['tag2'] = df['tag2'].astype(str)

            # Filter the dataframe for interactions involving the target tag
            filtered_df = df[(df['tag1'] == target_tag) | (df['tag2'] == target_tag)]

            # Create a directed graph for the contact network
            G = nx.DiGraph()

            # Extract unique tags and map them to colors
            unique_tags = pd.concat([filtered_df['tag1'], filtered_df['tag2']]).unique()
            colors = cm.rainbow(np.linspace(0, 1, len(unique_tags)))
            color_map = {tag[-2:]: color for tag, color in zip(unique_tags, colors)}

            # Add nodes and edges to the graph based on the contact hierarchy
            for _, row in filtered_df.iterrows():
                if row['tag1'] == target_tag:
                    contact_tag = row['tag2']
                else:
                    contact_tag = row['tag1']

                start_time = row['start_time']
                end_time = row['end_time']
                edge_label = f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"

                # Add edges with last two digits of the tags
                G.add_edge(target_tag[-2:], contact_tag[-2:], label=edge_label)

            # Use a spring layout for the graph
            pos = nx.spring_layout(G, seed=42, k=0.3, iterations=50)

            # Get the node colors based on the color_map
            node_colors = [color_map[node] for node in G.nodes()]

            # Plot the graph
            plt.figure(figsize=(14, 10))
            nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, 
                    font_size=10, font_color='black', font_weight='bold', edge_color='gray', alpha=0.7)

            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            
            # Add the plot to the PDF
            pdf.savefig()
            plt.close()

    print(f"All plots have been saved in {output_pdf_file}")

# ------------------------------------------
# ----------------- PART 17 ----------------
# ------------------------------------------

def calculate_disease_beta_values(distances):
    # Load the dataset
    #distances = pd.read_csv(dataset)
    total_time = len(distances)

    # Distance threshold (6 ft in meters)
    distance_threshold = 6 * 12 * 0.0254  # 6 ft in meters

    # Define probability functions for each disease
    # def covid(d):
    #     a = 0.865
    #     c = 1.246
    #     return a / (1 + c * d**2)

    def influenza(d):
        a = 0.06
        b = -1.45
        c = 0.01
        return a * d**b + c

    def measles(d):
        sigma = 2
        mu = 0
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((d - mu)**2) / (2 * sigma**2))

    def chickenpox(d):
        lambd = 1.5
        k = 1.8
        return 1 - np.exp(-(d/lambd)**k)

    def mers(d):
        c = 0.5
        return 1 / (d + c)**2

    def pertussis(d):
        P0 = 0.5
        alpha = 0.3
        return P0 * np.exp(-alpha * d)

    def legionnaires(d):
        beta = 0.02
        gamma = 0.15
        return beta * np.exp(-gamma * d)

    def common_cold(d):
        C1 = 0.1
        C2 = 0.01
        n = 2
        return C1 / d**n + C2

    # List of tags (these are extracted from the columns of the dataset)
    tags = set()
    for col in distances.columns[1:]:  # Skip the time column
        tag1, tag2 = col.split("-")
        tags.add(tag1.strip())
        tags.add(tag2.strip())
    tags = list(tags)

    # Create a dictionary to map each disease to its probability function
    disease_functions = {
        # "COVID-19": covid,
        "Influenza": influenza,
        "Measles": measles,
        "Chickenpox": chickenpox,
        "MERS": mers,
        "Pertussis": pertussis,
        "Legionnaires": legionnaires,
        "Common Cold": common_cold
    }

    # Extract the dataset name to create a subfolder for it
    # dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    dataset_folder = os.path.join("outputs/respiratory_disease_modeling")
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Loop through each disease and calculate the beta values
    for disease, P in disease_functions.items():
        # Initialize betas for all tags
        betas = {tag: 0 for tag in tags}

        # Compute betas
        for i in range(total_time):
            for tag in tags:
                P_list = []
                other_tags = [t for t in tags if t != tag]
                for other_tag in other_tags:
                    tag_pair = f"{tag}-{other_tag}"

                    if tag_pair in distances.columns:
                        distance = distances.at[i, tag_pair]

                        if pd.isna(distance):
                            continue

                        # Convert inches to meters
                        distance = distance * 0.0254

                        if distance <= distance_threshold:
                            P_list.append(P(distance))

                if P_list:
                    prob = 1 - np.prod([1 - p for p in P_list])
                    betas[tag] += prob

        # Create DataFrame for betas and divide by total_time
        beta_table = pd.DataFrame(list(betas.items()), columns=['tag', 'beta_value'])
        beta_table['beta_value'] = beta_table['beta_value'] / total_time

        # Save the beta table to CSV in the dataset-specific folder
        output_filename = os.path.join(dataset_folder, f"{disease.lower().replace(' ', '_')}_betavalues.csv")
        beta_table.to_csv(output_filename, index=False)
        print(f"Saved {disease} beta values to {output_filename}")


# ------------------------------------------
# ----------------- PART 18 ----------------
# ------------------------------------------

def calculate_covid19_beta_values(all_distances):
    dataset_folder = os.path.join("outputs/respiratory_disease_modeling")
    # Read distances
    distances = all_distances

    total_time = len(distances)
    ncol_distances = distances.shape[1]

    # Distance threshold 6 ft --- CHANGE TO OTHER DISTANCES
    distance_threshold = 6 * 12 * 0.0254  # 6 ft in meters where 1 inch = 0.0254 meters

    # General probability functions
    a = 0.865
    c = 1.246

    def P1(d):
        return a / (1 + c * d**2)

    def P2(d):
        return (-18.19 * np.log(d) + 43.276) / 100

    # Modified betas for all tags
    #tags =  distances['tagId'].unique() # ["62019", "62020", "62023", "62025", "62026", "62027", "62028", "62029", "62030", "62031", "62034", "62035", "62037", "62038", "62039", "62040", "62044", "62045", "62046"]
    # List of tags (these are extracted from the columns of the dataset)
    tags = set()
    for col in distances.columns[1:]:  # Skip the time column
        tag1, tag2 = col.split("-")
        tags.add(tag1.strip())
        tags.add(tag2.strip())
    tags = list(tags)

    betas = {tag: 0 for tag in tags}

    P = P1  # Using the P1 probability function
    for i in range(total_time):
        for tag in tags:
            P_list = []
            other_tags = [t for t in tags if t != tag]
            for other_tag in other_tags:
                tag_pair = f'{tag}-{other_tag}'

                if tag_pair not in distances.columns:
                    continue

                distance = distances.at[i, tag_pair]
                if pd.isna(distance):
                    continue

                # Convert inches to meters
                distance = distance * 0.0254
                if distance <= distance_threshold:
                    P_list.append(P(distance))

            if len(P_list) > 0:
                prob = 1 - np.prod([1 - p for p in P_list])
                betas[tag] += prob

    # Put beta values into a DataFrame
    beta_table = pd.DataFrame({
        'tag': list(betas.keys()),
        'beta_value': list(betas.values())
    })

    # Divide beta_values by total_time
    beta_table['beta_value'] = beta_table['beta_value'] / total_time

    # Save the beta table to CSV in the dataset-specific folder
    output_filename = os.path.join(dataset_folder, 'covid19_betavalues.csv')
    beta_table.to_csv(output_filename, index=False)
    print(f"Saved covid19 beta values to {output_filename}")


# ------------------------------------------
# ----------------- PART 19 ----------------
# ------------------------------------------

def simulate_infection_spread(distances):
    output_folder = os.path.join("outputs/R0")
    distance_threshold=6 * 12 * 0.0254, iterations=100

    total_time = len(distances)

    # General probability function
    a = 0.865
    c = 1.246

    def P1(d):
        return a / (1 + c * d**2)

    # Extract unique tag IDs from the column names
    tag_columns = distances.columns[1:]  # Assuming the first column is time
    tags = set()
    for col in tag_columns:
        tags.update(col.split('-'))
    tags = list(tags)

    # Set the random seed
    np.random.seed(234)

    # Function to simulate infection spread for a single seed tag
    def simulate_infection(seed_tag, tags, distances, distance_threshold, P, total_time, iterations=100):
        total_secondary_infections = 0

        for run in range(iterations):
            infected = {tag: 1 if tag == seed_tag else 0 for tag in tags}

            for i in range(total_time):
                for tag in [seed_tag]:
                    other_tags = [t for t in tags if t != tag]

                    for other_tag in other_tags:
                        tag_pair_1 = f"{tag}-{other_tag}"
                        tag_pair_2 = f"{other_tag}-{tag}"

                        if tag_pair_1 in distances.columns:
                            distance = distances.loc[i, tag_pair_1]
                        elif tag_pair_2 in distances.columns:
                            distance = distances.loc[i, tag_pair_2]
                        else:
                            distance = np.nan

                        if pd.isna(distance):
                            continue

                        distance = distance * 0.0254  # Convert inches to meters

                        if distance <= distance_threshold:
                            prob = P(distance)
                            prob /= 60  # Adjust for time

                            random_value = np.random.rand()
                            if random_value <= prob:
                                if infected[other_tag] == 0:  # Only count new infections
                                    total_secondary_infections += 1
                                infected[other_tag] = 1
                                print(f"{run} {i} {other_tag} is getting infected by {tag} distance {distance} prob {prob} rnd {random_value}")

        # Calculate R0 as the average number of secondary infections
        r0 = total_secondary_infections / iterations

        # Create a table of infections for the seed tag
        inf_table = pd.DataFrame({
            'seed_tag': [seed_tag],
            'r0': [r0]
        })

        return inf_table

    # Run the simulation for each tag as a single seed infection
    all_results = []
    for seed_tag in tags:
        results = simulate_infection(seed_tag, tags, distances, distance_threshold, P1, total_time, iterations)
        all_results.append(results)

    # Combine all results into a single data frame
    final_results = pd.concat(all_results, ignore_index=True)

    output_file = os.path.join(output_folder, 'R0.csv')
    final_results.to_csv(output_file, index=False)
    print(f"Saved covid19 beta values to {output_file}")

    final_results.to_csv(output_file, index=False)
    print(f'All results saved to {output_file}')
