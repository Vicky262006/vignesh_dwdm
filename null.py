import pandas as pd

# Load your dataset
df = pd.read_csv("main.csv")

# Fill all empty or whitespace-only fields with the string 'NULL'
df = df.replace(r'^\s*$', 'NULL', regex=True)

# Also fill real NaN values (if any) with 'NULL'
df = df.fillna('NULL')

# Save the cleaned file
df.to_csv("main_cleaned.csv", index=False)

print("✅ All empty and NaN fields have been filled with 'NULL' and saved as main_cleaned.csv")
