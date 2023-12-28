import csv

input_file = 'cv-other-dev.csv'
output_file = 'english_filtered_output.csv'

# Open the input file and read rows that meet the condition
with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write header from input file
    header = next(reader)
    writer.writerow(header)

    # Filter rows based on the condition (row 7 value is 'indian')
    for row in reader:
        if len(row) >= 7 and row[6].strip().lower() == 'indian':
            writer.writerow(row)

print("Filtered rows written to", output_file)
