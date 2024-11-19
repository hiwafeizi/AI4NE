import pandas as pd

# Define the processed data file path
data_file = "data/processed/netherlands_data.csv"

def load_data(file_path):
    """
    Load the processed data into a Pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def summary_statistics(data):
    """
    Print summary statistics for numerical and categorical features.
    """
    # Print overall statistics for numerical features
    numerical_features = ['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'frp']
    print("Summary Statistics (Numerical Features):\n", data[numerical_features].describe())

    # Count records by month and day (mm_dd)
    date_counts = data['mm_dd'].value_counts().sort_index()
    print("\nNumber of Records by Date (mm_dd):\n", date_counts)

    # Count records by type
    if 'type' in data.columns:
        type_counts = data['type'].value_counts()
        print("\nNumber of Records by Type:\n", type_counts)

    # Count records by day/night
    if 'daynight' in data.columns:
        daynight_counts = data['daynight'].value_counts()
        print("\nDay/Night Distribution:\n", daynight_counts)

    # Count records by confidence levels
    if 'confidence' in data.columns:
        confidence_counts = data['confidence'].value_counts()
        print("\nConfidence Level Distribution:\n", confidence_counts)

def main():
    # Load the data
    data = load_data(data_file)
    
    # Perform EDA
    summary_statistics(data)

# Run the script
if __name__ == "__main__":
    main()
