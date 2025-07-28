# Simulated data for diagrams
diagrams_info = {
    "conflict_resolution_diagram": {
        "stakeholders": ["Stakeholder 1", "Stakeholder 2", "Stakeholder 3"],
        "communication_pathways": [
            ("Stakeholder 1", "ConflictResolutionSystem"),
            ("Stakeholder 2", "ConflictResolutionSystem"),
            ("Stakeholder 3", "ConflictResolutionSystem"),
            ("ConflictResolutionSystem", "NegotiationInterface"),
            ("NegotiationInterface", "Stakeholder 1"),
            ("NegotiationInterface", "Stakeholder 2"),
            ("NegotiationInterface", "Stakeholder 3"),
        ]
    },
    # Add more diagrams as needed
}

def get_diagram_info(diagram_name):
    diagram = diagrams_info.get(diagram_name)
    if diagram:
        stakeholders = diagram["stakeholders"]
        num_pathways = len(diagram["communication_pathways"])
        return stakeholders, num_pathways
    else:
        return None, None

# Main function to run the script
def main():
    # Input the name of the diagram
    diagram_name = input("Enter the name of the diagram: ")
    stakeholders, num_pathways = get_diagram_info(diagram_name)
    
    # Output the results
    if stakeholders is not None:
        print(f"Stakeholders in the '{diagram_name}':")
        for stakeholder in stakeholders:
            print(f"- {stakeholder}")
        print(f"Number of communication pathways: {num_pathways}")
    else:
        print(f"The diagram '{diagram_name}' was not found.")

if __name__ == "__main__":
    main()
