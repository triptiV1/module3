import time
import threading

# Function to process a chunk of lines
def process_chunk(start, end, file1, file2, output_file, thread_id):
    print(f"Thread-{thread_id} started processing lines from {start} to {end}")
    
    # Open the files for reading and the output file for appending results
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'a') as out:
        f1.seek(start)
        f2.seek(start)

        # Loop over the lines in the current chunk
        for i in range(start, end):
            line1 = f1.readline().strip()
            line2 = f2.readline().strip()
            if line1 and line2:
                # Add the corresponding lines from both files
                result = int(line1) + int(line2)
                out.write(f"{result}\n")
            
            # Optional: Print progress every 1 million lines
            if i % 1000000 == 0:
                print(f"Thread-{thread_id} processed {i} lines...")

    print(f"Thread-{thread_id} finished processing lines from {start} to {end}")


# Function to process the chunked files using threads
def process_file_chunk(file1, file2, output_file, start, end, num_threads):
    # Calculate chunk size
    chunk_size = (end - start) // num_threads
    threads = []
    
    print(f"Starting processing of the file from line {start} to {end} with {num_threads} threads...")
    
    # Create and start threads
    for i in range(num_threads):
        chunk_start = start + i * chunk_size
        chunk_end = chunk_start + chunk_size if i < num_threads - 1 else end
        thread = threading.Thread(target=process_chunk, args=(chunk_start, chunk_end, file1, file2, output_file, i + 1))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print(f"Finished processing from line {start} to {end}")


# Main function to start processing the chunked files
def main(file1, file2, output_file, start, end, num_threads):
    start_time = time.time()
    process_file_chunk(file1, file2, output_file, start, end, num_threads)
    end_time = time.time()
    print(f"Processing took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    num_threads = 4  # Number of threads to use
    
    # Process each chunk in parallel
    for i in range(1, 11):  # 10 chunks
        file1_chunk = f"hugefile1_chunk{i}.txt"
        file2_chunk = f"hugefile2_chunk{i}.txt"
        output_file = f"totalfile_chunk{i}.txt"
        
        # Process each pair of chunked files
        main(file1_chunk, file2_chunk, output_file, 0, 100000000, num_threads)  # Adjust line range as necessary
