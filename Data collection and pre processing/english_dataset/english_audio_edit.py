import csv
import os

csv_file = 'english_final_output.csv'  # Your CSV file containing file paths
source_folder = 'source_folder/'  # Folder where the files are currently located
destination_folder = 'cv-other-dev/'  # Folder where you want to move the files

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Open the CSV file and retrieve file paths
with open(csv_file, 'r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present
    for row in reader:
        file_path = row[0]  # Assuming the file path is in the first column of the CSV
        
        # Extract filename from the path
        filename = os.path.basename(file_path)
        
        # Construct source and destination paths
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        
        # Move the file to the destination folder using os.rename or os.replace
        # If os.replace is used, existing files in the destination folder will be replaced
        os.rename(source_path, destination_path)
        print(f"Moved {filename} to {destination_folder}")
