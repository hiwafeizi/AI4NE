import pandas as pd

# Define the processed data file path
data_file = "data/wildfiredb.csv"
output_file = "data/wildfiredb_first_10000.csv"

def load_data(file_path, nrows=10000):
    """
    Load the first `nrows` rows of the processed data into a Pandas DataFrame.
    """
    data = pd.read_csv(file_path, nrows=nrows)
    print(data.head())  # Display the first few rows
    return data

def save_data(data, output_file):
    """
    Save the DataFrame to a new CSV file.
    """
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

def get_frp_stats(data):
    """
    Display summary statistics for the FRP column.
    """
    if 'frp' in data.columns:
        print("\nFRP Statistics:")
        print(data['frp'].describe())  # Descriptive statistics for FRP
    else:
        print("\nThe column 'frp' is not found in the dataset.")

def main():
    # Load the data
    data = load_data(data_file)
    
    # Save the data to a new file
    save_data(data, output_file)
    
    # Get FRP stats
    get_frp_stats(data)

# Run the script
if __name__ == "__main__":
    main()
