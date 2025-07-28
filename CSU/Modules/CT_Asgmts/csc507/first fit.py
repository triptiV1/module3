class MemoryBlock:
    def __init__(self, size):
        self.size = size
        self.allocated = False
        self.process_id = None

class FirstFitAllocator:
    def __init__(self, memory_blocks):
        self.memory = [MemoryBlock(size) for size in memory_blocks]

    def allocate(self, process_id, process_size):
        for block in self.memory:
            if not block.allocated and block.size >= process_size:
                block.allocated = True
                block.process_id = process_id
                print(f"Process {process_id} allocated {process_size}KB in block of {block.size}KB")
                return True
        print(f"Process {process_id} could not be allocated {process_size}KB")
        return False

    def deallocate(self, process_id):
        for block in self.memory:
            if block.allocated and block.process_id == process_id:
                block.allocated = False
                block.process_id = None
                print(f"Process {process_id} deallocated from block of {block.size}KB")
                return True
        print(f"Process {process_id} not found in memory")
        return False

    def display_memory(self):
        print("Memory Blocks:")
        for i, block in enumerate(self.memory):
            status = f"Allocated to Process {block.process_id}" if block.allocated else "Free"
            print(f"Block {i}: {block.size}KB - {status}")
