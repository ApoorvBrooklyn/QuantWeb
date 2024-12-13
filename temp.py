import pandas as pd

# File paths
input_file = "global_market_interactions.csv"  # Replace with your input file path
output_file = "cleaned_dataset.csv"  # Replace with your desired output file path

# Define a chunk size
chunk_size = 10000  # Number of rows per chunk

# Open the output file in write mode
with open(output_file, "w", encoding="utf-8", newline="") as output:
    # Process the file in chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Remove country names from the "headline" column
        chunk['headline'] = chunk['headline'].str.replace(r'^[A-Za-z]+:\s*', '', regex=True)
        
        # Append processed chunk to the output file
        chunk.to_csv(output, index=False, header=output.tell() == 0)  # Write header only once
