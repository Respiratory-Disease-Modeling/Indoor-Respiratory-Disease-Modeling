import pandas as pd
import numpy as np
import datetime

def interpolate_missing_data_and_update_excel(file_path, output_file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Define the columns with missing data (x and y columns)
    missing_columns = ['x', 'y']

    # Set a threshold for the maximum number of consecutive missing rows
    max_consecutive_missing = 100

    # Set a threshold for the maximum time gap in seconds (3 seconds)
    max_time_gap = 2

    # Iterate through missing columns and interpolate missing 'x' and 'y' values
    for col in missing_columns:
        consecutive_missing = 0  # Initialize a counter for consecutive missing rows
        prev_non_missing_row = 0  # Initialize the first row with data
        for i in range(len(df)):
            if pd.isna(df.loc[i, col]):
                consecutive_missing += 1
            else:
                if consecutive_missing > 0 and consecutive_missing <= max_consecutive_missing:
                    # Check if 'converted_date' and 'converted_time' columns are valid for both current and previous rows
                    if pd.notna(df.loc[i, 'converted_date']) and pd.notna(df.loc[i, 'converted_time']) and \
                       pd.notna(df.loc[prev_non_missing_row, 'converted_date']) and pd.notna(df.loc[prev_non_missing_row, 'converted_time']):
                        # Combine 'converted_date' and 'converted_time' to create a timestamp
                        current_timestamp = pd.to_datetime(df.loc[i, 'converted_date'] + ' ' + df.loc[i, 'converted_time'])
                        prev_timestamp = pd.to_datetime(df.loc[prev_non_missing_row, 'converted_date'] + ' ' + df.loc[prev_non_missing_row, 'converted_time'])

                        # Calculate the time gap between the current row and the previous non-missing row
                        time_gap = (current_timestamp - prev_timestamp).total_seconds()

                        if time_gap <= max_time_gap:
                            # Handle NaN values in start_value and end_value
                            start_value = df.loc[prev_non_missing_row, col]
                            if pd.isna(start_value):
                                start_value = np.nan
                            end_value = df.loc[i, col]
                            if pd.isna(end_value):
                                end_value = np.nan

                            # Interpolate missing 'x' and 'y' values for up to max_consecutive_missing rows within the time threshold
                            step = (end_value - start_value) / (consecutive_missing + 1)
                            for j in range(1, consecutive_missing + 1):
                                df.loc[prev_non_missing_row + j, col] = start_value + j * step
                                # Update the "success" column to TRUE for interpolated rows
                                df.loc[prev_non_missing_row + j, 'success'] = 'TRUE'
                                # Update 'z' value for the interpolated row
                                df.loc[prev_non_missing_row + j, 'z'] = 1000
                consecutive_missing = 0  # Reset the counter
                prev_non_missing_row = i  # Update the previous non-missing row

    # Save the updated DataFrame back to the Excel file without decimal values in "x" and "y"
    df.to_excel(output_file_path, index=False)

    print("Missing data interpolation and Excel file update complete.")

# Example usage:
# interpolate_missing_data_and_update_excel('output_sutest_orientation.xlsx', 'interpolationfile_output_sutest_orientation.xlsx')
