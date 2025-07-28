class OpenAddressingHashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        index = self._hash(key)
        initial_index = index
        
        while self.table[index] is not None:
            # Linear probing: move to the next slot
            index = (index + 1) % self.size
            if index == initial_index:
                raise Exception("Hash table is full")
        
        self.table[index] = (key, value)
    
    def search(self, key):
        index = self._hash(key)
        initial_index = index
        
        while self.table[index] is not None:
            k, v = self.table[index]
            if k == key:
                return v
            index = (index + 1) % self.size
            if index == initial_index:
                break
        
        return None

# Example usage
ht = OpenAddressingHashTable(10)
ht.insert("apple", 1)
ht.insert("banana", 2)
print(ht.search("apple"))  # Output: 1
print(ht.search("banana")) # Output: 2
