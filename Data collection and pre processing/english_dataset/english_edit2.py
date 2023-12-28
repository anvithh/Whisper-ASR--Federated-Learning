import csv

input_file = 'english_filtered_output.csv'  # The previously created output file
output_file = 'english_final_output.csv'   # The new output file

# Open the previously created output file and extract columns 1 and 2
with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Read rows and extract columns 1 and 2
    for row in reader:
        if len(row) >= 2:
            data_to_write = [row[0], row[1]]  # Extracting columns 1 and 2
            writer.writerow(data_to_write)

print("Columns 1 and 2 written to", output_file)
