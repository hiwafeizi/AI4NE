import os
import pandas as pd

# Define the data directory and output file path
input_dir = "data/viirs-snpp"
output_file = "data/processed/netherlands_data.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the output directory exists

# Define the updated header for the output file
HEADER = [
    "latitude", "longitude", "bright_ti4", "scan", "track", "mm_dd",
    "acq_time", "confidence", "bright_ti5", "frp", "daynight", "type"
]

# Columns to validate as numeric
NUMERIC_COLUMNS = ["latitude", "longitude", "bright_ti4", "bright_ti5", "frp"]

def process_file(file_path):
    """
    Process a single file: clean, filter, and prepare the data.
    """
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Remove rows with null values
    data.dropna(inplace=True)
    
    # Filter out rows with "l" confidence
    data = data[data['confidence'] != 'l']
    
    # Round latitude and longitude to 0.01
    data['latitude'] = data['latitude'].round(2)
    data['longitude'] = data['longitude'].round(2)
    
    # Convert numerical columns to numeric, coercing errors to NaN
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Remove rows with NaN in any of the numerical columns
    data.dropna(subset=NUMERIC_COLUMNS, inplace=True)
    
    # Transform 'acq_date' to 'mm_dd' (removing the year)
    data['mm_dd'] = pd.to_datetime(data['acq_date']).dt.strftime('%m_%d')
    data.drop(columns=['acq_date'], inplace=True)

    # Drop unnecessary columns
    columns_to_drop = ['satellite', 'instrument']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

    # Reorder columns to match the HEADER
    data = data[HEADER]

    return data

def process_and_append_netherlands_files(input_dir, output_file):
    """
    Process only Netherlands files and append the results to a single CSV file.
    """
    # Empty the output file and write the header
    with open(output_file, 'w') as f:
        f.write(",".join(HEADER) + "\n")  # Write the header line

    # Walk through the input directory to find files
    for year_dir in os.listdir(input_dir):
        year_path = os.path.join(input_dir, year_dir)
        if os.path.isdir(year_path):  # Ensure it's a directory
            for file_name in os.listdir(year_path):
                if file_name.endswith(".csv") and "Netherlands" in file_name:  # Only process Netherlands files
                    input_file = os.path.join(year_path, file_name)
                    
                    # Process the file
                    processed_data = process_file(input_file)
                    
                    # Append the processed data to the output file
                    processed_data.to_csv(output_file, mode='a', header=False, index=False)
                    print(f"Processed and appended: {input_file}")

# Run the script
process_and_append_netherlands_files(input_dir, output_file)
