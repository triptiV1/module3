def split_file(input_file, num_chunks=10):
    # Get the total number of lines in the file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Calculate the number of lines per chunk
    chunk_size = len(lines) // num_chunks
    
    # Create the smaller files
    base_name = input_file.split('.')[0]
    for i in range(num_chunks):
        chunk_filename = f"{base_name}_chunk{i+1}.txt"
        with open(chunk_filename, 'w') as chunk_file:
            # Determine the start and end indices for this chunk
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(lines)
            chunk_file.writelines(lines[start:end])
            print(f"Created {chunk_filename}")

# Split hugefile1.txt and hugefile2.txt into 10 smaller files
split_file("hugefile1.txt")
split_file("hugefile2.txt")
