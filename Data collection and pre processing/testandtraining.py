from sklearn.model_selection import train_test_split
import pandas as pd

# Input file path
input_file = input("Enter the input file name: ")

# Output file paths for training and test datasets
output_train_file = input("Enter the output file name for the training set: ")
output_test_file = input("Enter the output file name for the test set: ")

# Load the dataset from the input file (assuming a CSV file)
# Replace 'sep' with the appropriate separator (e.g., ',' for CSV, '\t' for TSV)
# Replace 'header' with None if there's no header in the file
data = pd.read_csv(input_file, sep=',')  # Replace 'sep' as needed

# Splitting the dataset into training and test sets (60% training, 40% test)
train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

# Writing training and test datasets to new files
train_data.to_csv(output_train_file, index=False)  # Writing training data to a file
test_data.to_csv(output_test_file, index=False)  # Writing test data to a file

print(f"Training set saved to {output_train_file}")
print(f"Test set saved to {output_test_file}")
