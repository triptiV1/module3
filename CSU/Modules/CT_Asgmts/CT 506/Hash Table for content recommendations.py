class HashTable:
    def __init__(self, size=100):
        """Initialize the hash table with a specified size."""
        self.size = size
        self.table = [[] for _ in range(size)]  # Create a list of empty lists for chaining

    def _hash_function(self, key):
        """Compute the hash value for a given key to find its index in the table."""
        return hash(key) % self.size

    def insert(self, key, value):
        """Insert a key-value pair into the hash table.
        
        If the key already exists, update its value.
        """
        index = self._hash_function(key)
        for pair in self.table[index]:
            if pair[0] == key:
                pair[1] = value  # Update existing key
                return
        self.table[index].append([key, value])  # Add new key-value pair

    def search(self, key):
        """Search for a value by its key in the hash table.
        
        Returns the value if found, otherwise returns None.
        """
        index = self._hash_function(key)
        for pair in self.table[index]:
            if pair[0] == key:
                return pair[1]  # Return value if key is found
        return None  # Key not found

    def delete(self, key):
        """Delete a key-value pair from the hash table.
        
        Returns True if the key was found and deleted, otherwise returns False.
        """
        index = self._hash_function(key)
        for i, pair in enumerate(self.table[index]):
            if pair[0] == key:
                del self.table[index][i]  # Remove key-value pair
                return True
        return False  # Key not found

def main():
    """Interactively use the HashTable class."""
    hash_table = HashTable(size=10)

    while True:
        action = input("Choose an action (insert, search, delete, exit): ").strip().lower()

        if action == "exit":
            print("Exiting the program.")
            break
        elif action == "insert":
            key = input("Enter key (e.g., user ID): ").strip()
            value = input("Enter value (e.g., content): ").strip()
            hash_table.insert(key, value)
            print(f"Inserted key-value pair ({key}, {value}).")
        elif action == "search":
            key = input("Enter key to search: ").strip()
            result = hash_table.search(key)
            if result is not None:
                print(f"Value found for key '{key}': {result}")
            else:
                print(f"No value found for key '{key}'.")
        elif action == "delete":
            key = input("Enter key to delete: ").strip()
            if hash_table.delete(key):
                print(f"Key '{key}' deleted successfully.")
            else:
                print(f"Key '{key}' not found.")
        else:
            print("Invalid action. Please choose 'insert', 'search', 'delete', or 'exit'.")

if __name__ == "__main__":
    main()
