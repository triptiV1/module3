def convert_to_words(amount):
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    scales = ["", "Thousand", "Million", "Billion"]

    def convert_hundreds(n):
        word = ""
        if n >= 100:
            word += ones[n // 100] + " Hundred "
            n %= 100
        if 10 <= n < 20:
            word += teens[n - 10]
        else:
            word += tens[n // 10]
            n %= 10
            word += " " + ones[n]
        return word.strip()

    def convert_whole_number(n):
        if n == 0:
            return "Zero"
        words = []
        scale_index = 0
        while n > 0:
            n, remainder = divmod(n, 1000)
            if remainder > 0:
                words.append(convert_hundreds(remainder) + " " + scales[scale_index])
            scale_index += 1
        return ', '.join(reversed(words)).strip()

    def convert_cents(n):
        return convert_hundreds(n) + " Cents" if n else "Zero Cents"

    def convert_amount_to_words(amount):
        dollars, cents = divmod(int(round(amount * 100)), 100)
        words = convert_whole_number(dollars) + " Dollars"
        if cents:
            words += " and " + convert_cents(cents)
        return words

    return convert_amount_to_words(amount)

# Prompt user for input
user_input = input("Please enter the amount in numeric form (e.g., 1234.56): ")

try:
    # Convert user input to float
    amount = float(user_input)
    
    # Validate input
    if amount < 0:
        raise ValueError("Amount cannot be negative.")
    
    # Convert amount to words
    amount_in_words = convert_to_words(amount)
    
    # Output the result
    print("Amount in words:", amount_in_words)
    
except ValueError as e:
    print("Invalid input:", e)
