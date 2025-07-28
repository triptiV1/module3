# Restaurant Meal Total Calculation

# Ask the user to enter the charge for the food
food_charge = float(input("Please enter the charge for the food: $"))

# Constants for the tip and sales tax rates
tip_rate = 0.18
sales_tax_rate = 0.07

# Calculate the tip and sales tax
tip_amount = food_charge * tip_rate
tax_amount = food_charge * sales_tax_rate

# Calculate the total amount
total_price = food_charge + tip_amount + tax_amount

# Display the results
print(f"Food Charge: ${food_charge:.2f}")
print(f"Tip (18%): ${tip_amount:.2f}")
print(f"Sales Tax (7%): ${tax_amount:.2f}")
print(f"Total Amount Due: ${total_price:.2f}")
