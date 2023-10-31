import pandas as pd

# Get the input TSV file name from the user
input_file_name = input("Enter the input TSV file name: ")

# Read the TSV file into a Pandas DataFrame
df = pd.read_csv(input_file_name, sep='\t')

# Remove the first, fifth, sixth, and seventh columns
df = df.drop(df.columns[[0, 4, 5, 6]], axis=1)

# Get the output TSV file name from the user
output_file_name = input("Enter the output TSV file name: ")

# Write the modified DataFrame to the new TSV file
df.to_csv(output_file_name, sep='\t', index=False)

print(f"Modified data saved to {output_file_name}")