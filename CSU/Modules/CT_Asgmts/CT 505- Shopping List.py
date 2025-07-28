class Prototype:
    def __init__(self, name, num_pages, flow):
        self.name = name
        self.num_pages = num_pages
        self.flow = flow
    
    def display_details(self):
        print(f"Prototype: {self.name}")
        print(f"Number of pages: {self.num_pages}")
        print("Page flow:")
        for page in self.flow:
            print(page)


# Creating a prototype instance
prototype = Prototype("Shopping List App", 5, ["Home", "Shopping List", "Add Item", "Edit Item", "Settings"])

# Displaying prototype details
prototype.display_details()
