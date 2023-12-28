import csv

# Function to convert TSV to CSV
def tsv_to_csv(input_tsv_file, output_csv_file):
    with open(input_tsv_file, 'r', newline='', encoding='utf-8') as tsvfile, open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        csv_writer = csv.writer(csvfile, delimiter=',')

        for row in tsv_reader:
            csv_writer.writerow(row)

    print("TSV to CSV conversion completed. Check the output file:", output_csv_file)

# Input file paths (input TSV and output CSV)
input_tsv = input("Enter the input TSV file name: ")
output_csv = input("Enter the output CSV file name: ")

# Call the function to convert TSV to CSV
tsv_to_csv(input_tsv, output_csv)
