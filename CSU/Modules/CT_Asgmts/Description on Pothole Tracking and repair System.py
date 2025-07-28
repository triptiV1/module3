class PHTRS:
    def __init__(self):
        self.actors = {
            'Citizen': ['Report Pothole', 'Log Damage Report'],
            'Public Works Staff': ['Assign Repair Crew', 'Track Repair Status', 'Generate Repair Reports'],
            'Repair Crew': []  # Assuming no specific use cases directly for Repair Crew
        }
    
    def print_use_cases(self):
        print("--------------------------------------")
        print("|        PotholeTrackingSystem         |")
        print("--------------------------------------")
        print("|                                    |")
        for actor, use_cases in self.actors.items():
            print(f"|              {actor: <20} |")
            for use_case in use_cases:
                print(f"|   - {use_case: <30} |")
            print("|                                    |")
        print("--------------------------------------")

# Usage example
if __name__ == "__main__":
    phtrs_system = PHTRS()
    phtrs_system.print_use_cases()
