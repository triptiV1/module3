import random

def generate_large_file(filename, num_lines=1000000000):
    with open(filename, 'w') as f:
        for _ in range(num_lines):
            f.write(f"{random.randint(1000, 9999)}\n")
    
    # After file is written, confirm the creation and count the lines
    print(f"{filename} has been regenerated.")
    
    # Count the number of lines in the file
    with open(filename, 'r') as f:
        line_count = sum(1 for line in f)
    
    print(f"{filename} contains {line_count} lines.")

# Generate the files
generate_large_file('hugefile1.txt')
generate_large_file('hugefile2.txt')
