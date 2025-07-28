def first_fit(memory_blocks, processes):
    allocation = [-1] * len(processes)
    
    for i in range(len(processes)):
        for j in range(len(memory_blocks)):
            if memory_blocks[j] >= processes[i]:
                allocation[i] = j
                memory_blocks[j] -= processes[i]
                break

    return allocation

# Example memory blocks and processes
memory_blocks = [100, 500, 200, 300, 600]
processes = [212, 417, 112, 426]

# Run the first-fit algorithm
allocation = first_fit(memory_blocks, processes)

# Display results
for i in range(len(processes)):
    if allocation[i] != -1:
        print(f"Process {i} of size {processes[i]} is allocated to memory block {allocation[i]}")
    else:
        print(f"Process {i} of size {processes[i]} could not be allocated")
