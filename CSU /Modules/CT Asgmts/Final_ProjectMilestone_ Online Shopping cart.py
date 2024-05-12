class ItemToPurchase:
    def __init__(self, item_name="none", item_price =0, item_quantity=0, item_description="none"):
        self.item_name = item_name
        self.item_price = item_price
        self.item_quantity = item_quantity
        self.item_description = item_description

    def print_item_cost(self):
      
        print (f'{self.item_name} {self.item_quantity} @ ${self.item_price} = ${self.item_price * self.item_quantity}')
                


class ShoppingCart:
    def __init__ (self, customer_name="none" , current_date="January 1, 2020" ):
        self.customer_name = customer_name
        self.current_date = current_date
        self.cart_items = []

    def add_item(self, ItemToPurchase):
        self.cart_items.append(ItemToPurchase)

    def remove_item(self, item_name):
        item_found = False
        for item in self.cart_items:
            if item.item_name == item_name:
                self.cart_items.remove(item)
                item_found = True
                break
        if not item_found:
            print("Item not found in cart.Nothing removed.")

    def modify_item(self, ItemToPurchase):
        for item in self.cart_items:
            if item.item_name == ItemToPurchase.item_name:
                if ItemToPurchase.item_price != 0:
                    item.item_price = ItemToPurchase.item_price
                if ItemToPurchase.item_quantity != 0:
                    item.item_quantity = ItemToPurchase.item_quantity
                if ItemToPurchase.item_description !=0:
                    item.item_description = ItemToPurchase.item_description
                return
        print("Item not found in cart. Nothing modified.")



    def get_num_items_in_cart(self):
        return sum(item.item_quantity for item in self.cart_items)

    def get_cost_of_cart(self):
        return sum(item.item_price * item.item_quantity for item in self.cart_items)

    def print_total(self):
        print(f"{self.customer_name}'s Shopping Cart - {self.current_date}")
        print(f"Number of Items: {self.get_num_items_in_cart()}")
        if len(self.cart_items) == 0:
            print("SHOPPING CART IS EMPTY")
        else:
            total_cost = 0
            for item in self.cart_items:
                item.print_item_cost()
                total_cost += item.item_price * item.item_quantity

            print(f"Total: ${total_cost}")


    def print_descriptions(self):
        print(f"{self.customer_name}'s Shopping Cart - {self.current_date}")
        print("\n Item Descriptions")
        for item in self.cart_items:
            print(f"Item name: {item.item_name}\n  Description: {item.item_description}\n Quantity: {item.item_quantity} \n")

            

def print_menu(cart):
    customer_menu = (
        "\nMENU\n"
        "a - Add item to cart\n"
        "r - Remove item from cart\n"
        "c - Change item quantity\n"
        "i - Output items' descriptions\n"
        "o - Output shopping cart\n"
        "q - Quit\n")
    option = ''
    while option != 'q':
        print(customer_menu)
        option = input('Choose an option:\n')
        if option =='a':
            print("\n ADD ITEM TO CART")
            item_name = input("Enter the item name : \n")
            item_price = float(input("Enter the item price:\n"))
            item_quantity = int(input("Enter the item quantity: \n"))
            item_description = input("Enter the item description: \n")
            cart.add_item(ItemToPurchase(item_name, item_price, item_quantity, item_description))
        elif option == 'r':
            print("\n REMOVE ITEM FROM THE CART")
            item_name = input("Enter the item name to remove: \n")
            cart.remove_item(item_name)
        elif option == 'c':
            print("\n CHANGE ITEM QUANTITY\n")
            item_name = input(" Enter the item name: \n")
            item_price = float(input("Enter the item price: \n"))
            item_quantity = int(input("Enter the new quantity: \n"))
            cart.modify_item(ItemToPurchase(item_name, item_price, item_quantity))
        elif option == 'i':
            print("\n OUTPUT ITEMS' DESCRIPTIONS")
            cart.print_descriptions()
        elif option == 'o':
            print("\n OUTPUT SHOPPING CART")
            cart.print_total()
        elif option == 'q':
            print("QUITTING...")
        else:
            print(" Invalid Option Please Try Again!!! ")


def main():

    print("Enter customer name: ")
    customer_name = input()
    print("Enter todays' date: ")
    current_date = input()
    print(f'Customer Name : {customer_name}')
    print(f'Todays Date : {current_date}')
    cart = ShoppingCart(customer_name, current_date)
    print_menu(cart)

    
             
if __name__ == "__main__":
    main()
                  
        
        
        
