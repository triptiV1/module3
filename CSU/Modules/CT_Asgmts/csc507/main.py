import subprocess
import time
import os

def run_processes():
    start_time = time.time()
    
    print("Starting the processing of chunked files...")

    processes = []

    # Start processes for all 10 chunks
    for i in range(1, 11):
        process1 = subprocess.Popen(['python3', 'process_half.py', '--start', '0', '--end', '100000000', '--file1', f'hugefile1_chunk{i}.txt', '--file2', f'hugefile2_chunk{i}.txt', '--output', f'totalfile_chunk{i}.txt'])
        processes.append(process1)
        print(f"Started process for chunk {i}...")

    # Wait for all processes to complete
    for i, process in enumerate(processes, 1):
        print(f"Waiting for process {i} (chunk {i}) to finish...")
        process.wait()
        print(f"Process {i} (chunk {i}) completed.")
    
    # Measure the end time
    end_time = time.time()
    print(f"Total processing time for all chunks: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Ensure the files exist before proceeding
    if not os.path.exists('hugefile1.txt'):
        print("Error: 'hugefile1.txt' does not exist!")
        exit(1)
    if not os.path.exists('hugefile2.txt'):
        print("Error: 'hugefile2.txt' does not exist!")
        exit(1)

    print("Both input files found. Proceeding with processing...")

    # Run the processes that will execute the halves of the task
    run_processes()

    print("Main script completed.")
