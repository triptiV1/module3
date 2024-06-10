class SoftwareDeveloper:
    def __init__(self, sense_of_fairness, resilience_under_pressure, attention_to_detail):
        self.sense_of_fairness = sense_of_fairness
        self.resilience_under_pressure = resilience_under_pressure
        self.attention_to_detail = attention_to_detail

    def display_traits(self):
        print("Software Developer Traits:")
        print(f"Sense of Fairness: {'Yes' if self.sense_of_fairness else 'No'}")
        print(f"Resilience Under Pressure: {'Yes' if self.resilience_under_pressure else 'No'}")
        print(f"Attention to Detail: {'Yes' if self.attention_to_detail else 'No'}")

class SoftwareDeveloperBuilder:
    def __init__(self):
        self.sense_of_fairness = False
        self.resilience_under_pressure = False
        self.attention_to_detail = False

    def with_sense_of_fairness(self, value):
        self.sense_of_fairness = value
        return self

    def with_resilience_under_pressure(self, value):
        self.resilience_under_pressure = value
        return self

    def with_attention_to_detail(self, value):
        self.attention_to_detail = value
        return self

    def build(self):
        return SoftwareDeveloper(self.sense_of_fairness, self.resilience_under_pressure, self.attention_to_detail)

def get_boolean_input(prompt):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ['yes', 'y']:
            return True
        elif user_input in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")

# Main program
if __name__ == "__main__":
    # Step 1: Create a builder
    builder = SoftwareDeveloperBuilder()

    # Step 2: Configure the builder with user input
    builder.with_sense_of_fairness(get_boolean_input("Does the developer have a sense of fairness? (yes/no): "))
    builder.with_resilience_under_pressure(get_boolean_input("Does the developer have resilience under pressure? (yes/no): "))
    builder.with_attention_to_detail(get_boolean_input("Does the developer have attention to detail? (yes/no): "))

    # Step 3: Build the object
    developer = builder.build()

    # Step 4: Use the constructed object
    developer.display_traits()

    # Print the steps information
    print("\nImportant Steps in the Program:")
    print("1. Create a builder instance.")
    print("2. Configure the builder with user input for desired traits.")
    print("3. Build the object.")
    print("4. Use the constructed object.")
