import os

# Directory containing the files
directory = './songs/songs/'

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Split the filename and extension
    name, extension = os.path.splitext(filename)
    # Check if the filename length exceeds 7 characters
    if len(name) > 7 and name[7:].isalpha():
        # Create the new filename by taking the first 7 characters
        new_name = name[:7] + extension
        # Get the full path of the old and new filenames
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_name}')
