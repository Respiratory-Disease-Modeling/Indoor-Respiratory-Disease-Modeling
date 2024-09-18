# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from utils import tag_pair_distances
# from contact_utils import process_data

# # Read distances
# input_file_path = './data/output_pizzatest.xlsx'
# input_df = pd.read_excel(input_file_path)
# distances = tag_pair_distances(input_df)

# # distances = pd.read_csv('Pizzatest_distances.csv')

# total_time = len(distances)

# # Distance threshold 6 ft --- CHANGE TO OTHER DISTANCES
# distance_threshold = 6 * 0.3048  # 6 ft

# # General probability function
# a = 0.865
# c = 1.246

# def P1(d):
#     return a / (1 + (c * (d**2)))

# def P2(d):
#     return (-18.19 * np.log(d) + 43.276) / 100

# # For plotting
# x = np.arange(0, 100, 0.1)
# y1 = P1(x)
# y2 = P2(x)

# plt.figure()
# plt.plot(x, y2, 'b-', label='P2')
# plt.plot(x, y1, 'r-', label='P1')
# plt.legend()
# plt.savefig('probability_functions.pdf')
# plt.close()

# # Modified betas for all tags
# betas = {tag: 0 for tag in ["200000638", "200000641", "200000661", "200000640", "200000663", "200000636", "200000639", "200000662", "200000654"]}

# # Probability function to use
# P = P1

# for i in range(total_time):
#     for tag in betas.keys():
#         P_list = []
#         other_tags = [t for t in betas.keys() if t != tag]

#         for other_tag in other_tags:
#             tag_pair = f'{tag}-{other_tag}'
#             distance = distances.loc[i, tag_pair]

#             if pd.isna(distance):  # Skip if distance is NA
#                 continue

#             distance = distance / 1000  # Convert mm to meters
#             if distance <= distance_threshold:
#                 P_list.append(P1(distance))

#         betas[tag] += sum(P_list)  # Update with the sum of P_list

# # Put beta values into a DataFrame
# beta_table = pd.DataFrame({'tag': list(betas.keys()), 'beta_value': [betas[tag] for tag in betas.keys()]})

# # Divide beta_values by total_time
# beta_table['beta_value'] = beta_table['beta_value'] / total_time
# beta_table.to_csv("pizzabetavalues.csv", index=False)

# # Contact counts by durations
# input_file = 'pizzaData.csv'
# feet = 6
# time_threshold = 1
# flatten_df = process_data(feet, time_threshold, input_file)
# contact_table = flatten_df

# count_table = contact_table.groupby('tag1').agg(
#     MC=('duration_seconds', lambda x: (x <= 30).sum()),
#     SC=('duration_seconds', lambda x: ((x > 30) & (x <= 60)).sum()),
#     LC=('duration_seconds', lambda x: (x > 60).sum()),
#     MEAN_DUR=('duration_seconds', 'mean')
# ).reset_index()

# count_table.rename(columns={'tag1': 'tag'}, inplace=True)

# # Merge count_table with beta_table
# beta_table = pd.merge(count_table, beta_table, on='tag')
# beta_table['event_name'] = 'event1'
# beta_table.to_csv("pizzabetavalues.csv", index=False)

# # Plot data
# pdata = beta_table.sort_values(by=['LC', 'SC', 'MC'])
# plt.figure()
# plt.plot(pdata['LC'], pdata['beta_value'], 'r-', linewidth=4, label='LC')
# pdata = beta_table.sort_values(by='SC')
# plt.plot(pdata['SC'], pdata['beta_value'], 'b-', linewidth=4, label='SC')
# pdata = beta_table.sort_values(by='MC')
# plt.plot(pdata['MC'], pdata['beta_value'], 'g-', linewidth=4, label='MC')
# plt.xscale('log')
# plt.xlim(1, 300)
# plt.ylim(0, 1)
# plt.legend(['LC', 'SC', 'MC'])
# plt.savefig('beta_values_plot.pdf')
# plt.close()

# # Scatter plot
# plt.figure()
# plt.scatter(beta_table['MC'], beta_table['beta_value'], color='green', label='MC')
# plt.xlabel('MC')
# plt.ylabel('Beta Value')
# plt.savefig('beta_value_scatter_plot.pdf')
# plt.close()
