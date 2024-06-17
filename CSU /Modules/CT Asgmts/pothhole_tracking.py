class PotholeTrackingSystem:
    def __init__(self):
        self.potholes = []
        self.damage_reports = []
        self.next_pothole_id = 1
        self.next_workorder_id = 1
        self.next_damage_report_id = 1

    def report_pothole(self, street_address, size, location):
        district = self.determine_district(street_address)
        repair_priority = self.calculate_repair_priority(size)
        
        pothole = {
            'id': self.next_pothole_id,
            'street_address': street_address,
            'size': size,
            'location': location,
            'district': district,
            'repair_priority': repair_priority,
            'status': 'reported'
        }
        self.potholes.append(pothole)
        self.next_pothole_id += 1
        return pothole

    def assign_repair_crew(self, pothole_id, crew_id):
        pothole = self.find_pothole_by_id(pothole_id)
        if pothole:
            workorder = {
                'id': self.next_workorder_id,
                'pothole_id': pothole_id,
                'crew_id': crew_id,
                'status': 'assigned'
            }
            self.next_workorder_id += 1
            return workorder
        else:
            return None

    def track_repair_status(self):
        return self.potholes

    def log_damage_report(self, citizen_name, address, phone_number, damage_type, damage_amount):
        damage_report = {
            'id': self.next_damage_report_id,
            'citizen_name': citizen_name,
            'address': address,
            'phone_number': phone_number,
            'damage_type': damage_type,
            'damage_amount': damage_amount
        }
        self.damage_reports.append(damage_report)
        self.next_damage_report_id += 1
        return damage_report

    def generate_repair_reports(self):
        return self.potholes, self.damage_reports

    def determine_district(self, street_address):
        # Simplified district determination logic based on street address
        # Example: Extract district from street address or use a default district
        return "District X"

    def calculate_repair_priority(self, size):
        # Simplified repair priority calculation based on pothole size
        # Example: Larger potholes get higher repair priority
        if size >= 8:
            return "High"
        elif size >= 5:
            return "Medium"
        else:
            return "Low"

    def find_pothole_by_id(self, pothole_id):
        for pothole in self.potholes:
            if pothole['id'] == pothole_id:
                return pothole
        return None


# Interactive script to simulate use cases

def main():
    system = PotholeTrackingSystem()

    while True:
        print("\n===== Pothole Tracking and Repair System Menu =====")
        print("1. Report Pothole")
        print("2. Assign Repair Crew")
        print("3. Track Repair Status")
        print("4. Log Damage Report")
        print("5. Generate Repair Reports")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            print("\n=== Report Pothole ===")
            street_address = input("Enter street address of pothole: ")
            size = int(input("Enter size of pothole (on a scale of 1 to 10): "))
            location = input("Enter location of pothole (e.g., middle, curb): ")
            pothole = system.report_pothole(street_address, size, location)
            print(f"Pothole reported successfully with ID: {pothole['id']}")

        elif choice == '2':
            print("\n=== Assign Repair Crew ===")
            pothole_id = int(input("Enter pothole ID to assign a repair crew: "))
            crew_id = int(input("Enter crew ID to assign: "))
            workorder = system.assign_repair_crew(pothole_id, crew_id)
            if workorder:
                print(f"Repair crew assigned successfully with Work Order ID: {workorder['id']}")
            else:
                print("Pothole ID not found.")

        elif choice == '3':
            print("\n=== Track Repair Status ===")
            potholes = system.track_repair_status()
            if potholes:
                print("Current repair status of potholes:")
                for pothole in potholes:
                    print(f"Pothole ID: {pothole['id']}, Status: {pothole['status']}")
            else:
                print("No potholes reported.")

        elif choice == '4':
            print("\n=== Log Damage Report ===")
            citizen_name = input("Enter your name: ")
            address = input("Enter your address: ")
            phone_number = input("Enter your phone number: ")
            damage_type = input("Enter type of damage: ")
            damage_amount = float(input("Enter estimated dollar amount of damage: "))
            damage_report = system.log_damage_report(citizen_name, address, phone_number, damage_type, damage_amount)
            print("Damage report logged successfully.")

        elif choice == '5':
            print("\n=== Generate Repair Reports ===")
            potholes, damage_reports = system.generate_repair_reports()
            if potholes:
                print("Repair Reports:")
                print("Potholes:")
                for pothole in potholes:
                    print(f"Pothole ID: {pothole['id']}, Status: {pothole['status']}")
                print("Damage Reports:")
                for damage_report in damage_reports:
                    print(f"Damage Report ID: {damage_report['id']}, Type: {damage_report['damage_type']}")
            else:
                print("No repair reports available.")

        elif choice == '6':
            print("Exiting the Pothole Tracking and Repair System. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main()
