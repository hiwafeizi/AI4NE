import os
import pandas as pd

# Define the data directory and output file path
input_dir = "data/viirs-snpp"
output_file = "data/processed/all_processed_data.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the output directory exists

def process_file(file_path, country_name):
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
    
    # Drop the 'satellite' column
    if 'satellite' in data.columns:
        data.drop(columns=['satellite'], inplace=True)
    
    # Add the country name column
    data['country'] = country_name
    
    return data

def process_and_append_all_files(input_dir, output_file):
    """
    Process all files and append the results to a single CSV file.
    """
    # Initialize the output file if it doesn't exist
    if not os.path.exists(output_file):
        pd.DataFrame().to_csv(output_file, index=False)  # Create an empty CSV file
    
    # Walk through the input directory to find files
    for year_dir in os.listdir(input_dir):
        year_path = os.path.join(input_dir, year_dir)
        if os.path.isdir(year_path):  # Ensure it's a directory
            for file_name in os.listdir(year_path):
                if file_name.endswith(".csv"):  # Only process CSV files
                    input_file = os.path.join(year_path, file_name)
                    
                    # Extract the country name from the file name
                    country_name = file_name.split('_')[-1].split('.')[0]  # Extract the part after the last underscore and before the file extension
                    
                    # Process the file
                    processed_data = process_file(input_file, country_name)
                    
                    # Append the processed data to the output file
                    processed_data.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                    print(f"Processed and appended: {input_file} with country: {country_name}")

# Run the script
process_and_append_all_files(input_dir, output_file)
