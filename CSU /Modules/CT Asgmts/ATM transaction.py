class ATM:
    def __init__(self):
        self.pin = "1234"
        self.attempt_counter = 0
        self.account_balance = 100  # Example balance

    def insert_card(self):
        print("Card inserted.")
        self.enter_pin()

    def enter_pin(self):
        entered_pin = input("Please enter your PIN: ")
        self.check_pin(entered_pin)

    def check_pin(self, entered_pin):
        if entered_pin == self.pin:
            print("PIN correct.")
            self.proceed_to_transaction()
        else:
            self.attempt_counter += 1
            print("Incorrect PIN.")
            if self.attempt_counter < 3:
                self.enter_pin()
            else:
                print("Too many incorrect attempts. Card ejected.")
                self.lock_account()

    def proceed_to_transaction(self):
        print("Proceeding to transactions.")
        self.withdraw_money()

    def withdraw_money(self):
        amount = float(input("Enter amount to withdraw: "))
        if self.account_balance > amount:
            self.account_balance -= amount
            print(f"${amount} dispensed. Current balance: ${self.account_balance}")
            self.complete_transaction()
        elif self.account_balance == 0:
            print("Account balance is zero. Closing account.")
            self.close_account()
        else:
            print("Insufficient funds.")
            self.complete_transaction()

    def complete_transaction(self):
        print("Please take your card.")
        print("Transaction complete.")

    def lock_account(self):
        print("Account locked due to too many incorrect attempts.")

    def close_account(self):
        print("Account closed.")

# Create an ATM instance and start the process
atm = ATM()
atm.insert_card()
