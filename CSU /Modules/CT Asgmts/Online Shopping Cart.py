class ItemToPurchase:
    def __init__(self):
        self.item_name = "none"
        self.item_price = 0
        self.item_quantity = 0

    def print_item_cost(self):
        #print('Name of the Item: ', self.item_name)
        #print('Price of the item is : ', self.item_price)
        #print('number of items needed:', self.item_quantity)
        print (f'{self.item_name} {self.item_quantity} @ ${self.item_price} = ${self.item_price * self.item_quantity}')
                


Item1 = ItemToPurchase()

Item1.item_name = input('Enter the name of item:')


Item1.item_price = float(input('Enter the Item price: '))


Item1.item_quantity = int(input('Enter the quantity: '))


print(Item1.item_name)
print(Item1.item_price)
print(Item1.item_quantity)

Item1.print_item_cost()


Item2 = ItemToPurchase()
Item2.item_name = input('Enter the next Item :')
Item2.item_price = float(input('Enter price of the item: '))
Item2.item_quantity = int(input('Enter the quantity: '))

Item2.print_item_cost()

print('\nTOTAL COST')

Item1.print_item_cost()
Item2.print_item_cost()

total_cost = Item1.item_price * Item1.item_quantity + Item2.item_price * Item2.item_quantity

print('Total : $ ', total_cost) 



