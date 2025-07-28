def linear_search(items, target_id):
    for index, item in enumerate(items):
        if item['id'] == target_id:
            return index
    return -1

# Example usage
items = [
    {'id': '11111', 'name': 'Laptop'},
    {'id': '12345', 'name': 'Smartphone'},
    {'id': '67890', 'name': 'Tablet'}
]

target_id = '67890'
result = linear_search(items, target_id)

if result != -1:
    print(f"Item found at index: {result}")
else:
    print("Item not found")
