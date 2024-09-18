import json
import datetime
import pandas as pd
import os

from pytz import timezone

# ------------------------------------------
# ---------------- PART 1.1 ----------------
# ------------------------------------------

# For POZYX 
def preprocess(file_path):    
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    output_folder = 'raw_data'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'preprocessed_{file_name}.csv')

    # Read the JSON file
    with open(file_path, 'r') as file:
        data = file.read()

    # Parse the JSON data
    json_list = json.loads(data)

    # Extract the required fields and create a list of dictionaries
    cleaned_data = []
    for entry in json_list:
        json_entry = json.loads(entry)  # Parse each item

        version = json_entry[0]['version']
        tagId = json_entry[0]['tagId']
        timestamp = json_entry[0]['timestamp']
        coordinates = json_entry[0]['data'].get('coordinates', {})
        success = json_entry[0]['success']
        blinkIndex = json_entry[0]['data']['tagData'].get('blinkIndex')
        anchor_data = json_entry[0]['data'].get('anchorData', [])
        latency = json_entry[0]['data']['metrics'].get('latency')

        anchorid = []
        rss = []

        # Iterate over anchor_data and extract anchorid and rss values
        for anchor in anchor_data:
            anchorid.append(anchor['anchorId'])
            rss.append(anchor['rss'])

        # Convert the timestamp to datetime
        converted_datetime = datetime.datetime.fromtimestamp(timestamp)
        converted_date = converted_datetime.strftime('%Y-%m-%d')
        time = converted_datetime.strftime('%H:%M:%S.%f')[:-3]  # Keep only milliseconds

        cleaned_data.append({
            'version': version,
            'tagId': tagId,
            'timestamp': timestamp,
            'x': coordinates.get('x'),
            'y': coordinates.get('y'),
            'z': coordinates.get('z'),
            'success': success,
            'blinkIndex': blinkIndex,
            'latency': latency,
            'anchorid1': anchorid[0] if len(anchorid) >= 1 else None,
            'anchorid2': anchorid[1] if len(anchorid) >= 2 else None,
            'anchorid3': anchorid[2] if len(anchorid) >= 3 else None,
            'anchorid4': anchorid[3] if len(anchorid) >= 4 else None,
            'anchorid5': anchorid[4] if len(anchorid) >= 5 else None,
            'rss1': rss[0] if len(rss) >= 1 else None,
            'rss2': rss[1] if len(rss) >= 2 else None,
            'rss3': rss[2] if len(rss) >= 3 else None,
            'rss4': rss[3] if len(rss) >= 4 else None,
            'rss5': rss[4] if len(rss) >= 5 else None,
            'converted_date': converted_date,
            'time': time,
        })

    # Create a DataFrame from the cleaned data
    df = pd.DataFrame(cleaned_data)

    # Define the output Excel file path
    # time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headers = ['version', 'tagId', 'timestamp', 'x', 'y', 'z', 'success', 'blinkIndex', 'latency',
            'anchorid1', 'anchorid2', 'anchorid3', 'anchorid4', 'anchorid5',
            'rss1', 'rss2', 'rss3', 'rss4', 'rss5', 'converted_date', 'time']

    # Reorder columns
    df = df[headers]

    # Save the DataFrame to Excel
    df.to_csv(output_file, index=False)
    print('Data saved to', output_file)

    return df

# ------------------------------------------
# ---------------- PART 1.2 ----------------
# ------------------------------------------

def preprocess2(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    output_folder = 'raw_data'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'preprocessed_{file_name}.csv')

    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)  # Directly load the JSON data as a list of dictionaries

    # Extract the required fields and create a list of dictionaries
    cleaned_data = []
    for json_entry in data:  # Now json_entry is already a dictionary
        version = json_entry[0]['version']
        tagId = json_entry[0]['tagId']
        timestamp = json_entry[0]['timestamp']
        coordinates = json_entry[0]['data'].get('coordinates', {})
        success = json_entry[0]['success']
        blinkIndex = json_entry[0]['data']['tagData'].get('blinkIndex')
        anchor_data = json_entry[0]['data'].get('anchorData', [])
        latency = json_entry[0]['data']['metrics'].get('latency')

        anchorid = []
        rss = []

        # Iterate over anchor_data and extract anchorid and rss values
        for anchor in anchor_data:
            anchorid.append(anchor['anchorId'])
            rss.append(anchor['rss'])

        # Convert the timestamp to datetime
        converted_datetime = datetime.datetime.fromtimestamp(timestamp)
        converted_date = converted_datetime.strftime('%Y-%m-%d')
        time = converted_datetime.strftime('%H:%M:%S.%f')[:-3]  # Keep only milliseconds

        cleaned_data.append({
            'version': version,
            'tagId': tagId,
            'timestamp': timestamp,
            'x': coordinates.get('x'),
            'y': coordinates.get('y'),
            'z': coordinates.get('z'),
            'success': success,
            'blinkIndex': blinkIndex,
            'latency': latency,
            'anchorid1': anchorid[0] if len(anchorid) >= 1 else None,
            'anchorid2': anchorid[1] if len(anchorid) >= 2 else None,
            'anchorid3': anchorid[2] if len(anchorid) >= 3 else None,
            'anchorid4': anchorid[3] if len(anchorid) >= 4 else None,
            'anchorid5': anchorid[4] if len(anchorid) >= 5 else None,
            'rss1': rss[0] if len(rss) >= 1 else None,
            'rss2': rss[1] if len(rss) >= 2 else None,
            'rss3': rss[2] if len(rss) >= 3 else None,
            'rss4': rss[3] if len(rss) >= 4 else None,
            'rss5': rss[4] if len(rss) >= 5 else None,
            'converted_date': converted_date,
            'time': time,
        })

    # Create a DataFrame from the cleaned data
    df = pd.DataFrame(cleaned_data)

    # Define the output Excel file path
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    excel_file = f'output_{time_stamp}.xlsx'

    headers = ['version', 'tagId', 'timestamp', 'x', 'y', 'z', 'success', 'blinkIndex', 'latency',
               'anchorid1', 'anchorid2', 'anchorid3', 'anchorid4', 'anchorid5',
               'rss1', 'rss2', 'rss3', 'rss4', 'rss5', 'converted_date', 'time']

    # Reorder columns
    df = df[headers]

    # Save the DataFrame to Excel
    df.to_csv(output_file, index=False)
    print('Data saved to', excel_file)

    return df

# ------------------------------------------
# ---------------- PART 1.3 ----------------
# ------------------------------------------

# For WISER
def preprocess3(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    output_folder = 'raw_data'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'preprocessed_{file_name}.csv')

    cleaned_data = []

    # Set the local timezone to 'America/Chicago' (Central Standard Time)
    local_tz = timezone('America/Chicago')

    # Read the JSON file
    with open(file_path, 'r') as file:
        # Read each line separately and process
        for line in file:
            try:
                item = json.loads(line)
                tagId = item['id']

                report = item['report']
                timestamp_ms = report['timestamp']

                # Convert timestamp to readable format with local timezone
                converted_timestamp = pd.to_datetime(int(timestamp_ms), unit='ms').tz_localize('UTC').tz_convert(local_tz)

                # Round off the timestamp to the nearest second
                rounded_timestamp = converted_timestamp.round('S')

                # Extract the rounded time in H:M:S format
                rounded_time = rounded_timestamp.strftime('%H:%M:%S')

                # Extract the date
                converted_date = rounded_timestamp.strftime('%Y-%m-%d')

                cleaned_data.append({
                    'tagId': tagId,
                    'timestamp': timestamp_ms,
                    'converted_timestamp': rounded_timestamp,
                    'time': rounded_time,
                    'converted_date': converted_date,
                    'x': report['location'].get('x'),
                    'y': report['location'].get('y'),
                    'z': report['location'].get('z'),
                    'numAnchors': report.get('numAnchors'),
                    'battery': report.get('battery'),
                    'error': report.get('error'),
                })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    # Create a DataFrame from the cleaned data
    df = pd.DataFrame(cleaned_data)

    # Drop duplicates only if both 'tagId' and 'timestamp' are the same
    df.drop_duplicates(subset=['tagId', 'timestamp'], inplace=True)

    # Sort DataFrame by 'converted_timestamp' column
    df.sort_values(by='converted_timestamp', inplace=True)

    # Drop rows where the converted_date is not the same as the first row's converted_date
    first_date = df['converted_date'].iloc[0]
    df = df[df['converted_date'] == first_date]

    # Reorder columns
    df = df[['tagId', 'timestamp', 'converted_timestamp', 'time', 'converted_date', 'x', 'y', 'z', 'numAnchors', 'battery', 'error']]

    # Save the DataFrame to a CSV file with specific columns
    df.to_csv(output_file, index=False)
    print('Data saved to', output_file)