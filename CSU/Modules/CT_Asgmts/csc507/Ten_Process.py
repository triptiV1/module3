def split_file(filename, num_parts=10):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    chunk_size = len(lines) // num_parts
    for i in range(num_parts):
        with open(f"{filename}_part_{i}", 'w') as part_file:
            part_file.writelines(lines[i * chunk_size : (i + 1) * chunk_size])

def sum_chunk_lines(chunk_id, file1, file2, output_file):
    with open(f"{file1}_part_{chunk_id}", 'r') as f1, open(f"{file2}_part_{chunk_id}", 'r') as f2, open(output_file, 'a') as out:
        for line1, line2 in zip(f1, f2):
            num1 = int(line1.strip())
            num2 = int(line2.strip())
            out.write(f"{num1 + num2}\n")

def run_in_parallel_10_parts():
    # Split files into 10 parts
    split_file('hugefile1.txt')
    split_file('hugefile2.txt')
    
    processes = []
    for i in range(10):
        p = multiprocessing.Process(target=sum_chunk_lines, args=(i, 'hugefile1.txt', 'hugefile2.txt', 'totalfile.txt'))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

run_in_parallel_10_parts()
