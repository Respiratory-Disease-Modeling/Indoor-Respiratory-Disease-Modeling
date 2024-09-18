from preprocess import *
from utils import analyze_data
from utils import plot_anchor_trajectories
from utils import plot_individual_tag_trajectories
from utils import plot_individual_tag_trajectories_wrt_anchors
from utils import plot_tag_pair_trajectories
from utils import plot_all_tags_trajectory
from utils import plot_all_tags_trajectory_points
from utils import tag_pair_distances
from contact_utils import process_data, contact_intensity, plot_contact_density
from contact_utils import generate_distance_matrices, create_contact_network_longer_duration
from contact_utils import create_contact_network_5_sec_duration, create_contact_network_for_each_individual_tag
from contact_utils import calculate_disease_beta_values, calculate_covid19_beta_values, simulate_infection_spread
from configs import *

import pandas as pd

# file_path = './raw_data/gathering1.json'
input_file_path = './data/gathering4.csv'
input_df = pd.read_csv(input_file_path)

# Part 1
# preprocessed xslx will be generated in the raw_data folder
# preprocessed_df = preprocess2(file_path)

# Part 2
# analyzed_data = analyze_data(preprocessed_df)

# Part 3: Plotting anchors
#plot_anchor_trajectories(ANCHOR_COORDINATES)

# Part 4: Plotting individual tags
# plot_individual_tag_trajectories(preprocessed_df)

# Plot 5
# plot_individual_tag_trajectories_wrt_anchors(preprocessed_df, ANCHOR_COORDINATES) 

# Plot 6
# plot_tag_pair_trajectories(preprocessed_df)

# Plot 7
# plot_all_tags_trajectory(preprocessed_df)

# Plot 8
# plot_all_tags_trajectory_points(preprocessed_df)

# Part 9  - Contact duration
# input_file = './data/micro_test_lab508.csv'
# input_file = './data/microtest_interpolation.csv'
# feet = 6
# time_threshold = 1
# flatten_df = process_data(feet, time_threshold, input_file)

# Part 10 - Contact intensity
# contact_intensity(flatten_df)

# Part 11 - Plot Contact density
# plot_contact_density(flatten_df)

# Part 12 - Distance Matrices
# generate_distance_matrices(preprocessed_df)

# Part 13 - Tag pair distances
# distance_df = tag_pair_distances(input_df)

# Part 14 - Contact tracing network for longer duration
# create_contact_network_longer_duration(flatten_df)

# Part 15 - Contact tracing network (5 sec duration)
# create_contact_network_5_sec_duration(flatten_df)

# Part 16 - Contact tracing network for each individual tag
# create_contact_network_for_each_individual_tag(flatten_df)

# Part 17 - Individual transmission rates for various diseases
# calculate_disease_beta_values(distance_df)

# Part 18 - Individual transmission rates for COVID-19
# calculate_covid19_beta_values(distance_df)

# Part 19 - Simulation of R0
# simulate_infection_spread(distance_df)


