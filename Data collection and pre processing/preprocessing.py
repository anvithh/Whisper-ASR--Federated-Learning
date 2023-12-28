import csv

# Function to clean the data in the second column
def clean_data(data):
    # Replace commas, dots, and other characters as needed
    cleaned_data = data.replace(',', '').replace('.', '').replace('|', '').replace('-', '')
    return cleaned_data

# Input file path
input_file = input("Enter the input TSV file name: ")

# Prompt user for output file name
output_file = input("Enter the output file name: ")

# Read the TSV file, clean the second column, and save to a new file
with open(input_file, 'r', newline='', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')

    # Iterate through each row, clean the second column, and write to the output file
    for row in reader:
        if len(row) >= 2:
            # Clean the data in the second column
            row[1] = clean_data(row[1])
        
        # Write the modified row to the output file
        writer.writerow(row)

print("Data cleaning completed. Check the output file:", output_file)
