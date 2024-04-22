#Bookstore

points_earned = 0
books_purchased = int(input('Enter the number of books purchased this month:  '))

if books_purchased in range(0,2):
    print('points earned = 0')
    
if books_purchased in range(2,4):
    print('points earned is = 5')
    #points_earned = points earned + 5
if books_purchased in range(4,6):
    print(' points earned = 15')

if books_purchased in range(6,8):
    print('points earned = 30')
if books_purchased >= 8:
    print('points earned = 60')


    

    
                      
