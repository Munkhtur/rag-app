import json

# Path to your JSON file
file_path = 'history-data2.json'

# Step 1: Read the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # Load JSON content into a Python object

# Step 2: Assuming the JSON content is an array
if isinstance(data, list):
    array_length = len(data)  # Get the length of the array
else:
    print("The JSON content is not an array.")

print(f"The length of the JSON array is: {array_length}")