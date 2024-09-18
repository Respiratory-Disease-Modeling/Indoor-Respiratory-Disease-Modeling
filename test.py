import pandas as pd
import numpy as np
import os

def calculate_betas(all_distances):
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

