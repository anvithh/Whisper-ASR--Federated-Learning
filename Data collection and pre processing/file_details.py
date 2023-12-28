import pandas as pd

# Get the input TSV file name from the user
input_file_name = input("Enter the input TSV file name: ")

# Read the TSV file into a Pandas DataFrame
try:
    df = pd.read_csv(input_file_name, sep='\t')

    # Print the number of rows and columns
    num_rows, num_columns = df.shape
    print(f'Number of rows: {num_rows}')
    print(f'Number of columns: {num_columns}')

    # Print column attribute names
    print('\nColumn Attribute Names:')
    for column_name in df.columns:
        print(column_name)


except FileNotFoundError:
    print(f"File '{input_file_name}' not found. Please make sure the file exists and try again.")
except pd.errors.EmptyDataError:
    print(f"File '{input_file_name}' is empty.")
except pd.errors.ParserError:
    print(f"Error parsing the TSV file. Please check if the file is a valid TSV file.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
