import pandas as pd

# Load the data from the CSV file.
# The `keep_default_na=False` argument prevents pandas from automatically
# treating empty cells as NaN, allowing the .replace method to find them.
df = pd.read_csv('main_with_NULL.csv', keep_default_na=False)

# Replace any NaN values with the string 'NULL'.
df.fillna('NULL', inplace=True)

# Save the updated DataFrame to a new CSV file.
df.to_csv('output_with_nulls.csv', index=False)

print("Empty cells have been replaced with 'NULL' and the file is saved as 'output_with_nulls.csv'")