class Stakeholder:
    def __init__(self, name):
        self.name = name

    def provide_requirements(self, resolution_system, requirements):
        print(f"{self.name} provides requirements: {requirements}")
        resolution_system.receive_requirements(self, requirements)

    def receive_outcome(self, resolved_requirements):
        print(f"{self.name} receives resolved outcome: {resolved_requirements}")


class ConflictResolutionSystem:
    def __init__(self):
        self.requirements_database = {}
        self.negotiation_interface = NegotiationInterface(self)

    def receive_requirements(self, stakeholder, requirements):
        self.requirements_database[stakeholder] = requirements
        self.process_requirements()

    def process_requirements(self):
        conflicts_detected = self.detect_conflicts()
        if conflicts_detected:
            self.negotiation_interface.notify_stakeholders_of_conflict()
        else:
            self.negotiation_interface.notify_stakeholders_of_resolution(self.requirements_database)

    def detect_conflicts(self):
        # Simple conflict detection: if more than one unique requirement, there's a conflict
        unique_requirements = set(self.requirements_database.values())
        return len(unique_requirements) > 1

    def resolve_conflicts(self):
        # Simple conflict resolution strategy: concatenate requirements
        resolved_requirements = " & ".join(set(self.requirements_database.values()))
        return resolved_requirements


class NegotiationInterface:
    def __init__(self, resolution_system):
        self.resolution_system = resolution_system

    def notify_stakeholders_of_conflict(self):
        print("Conflict detected. Resolving...")
        # Resolve conflicts using the resolution system's strategy
        resolved_requirements = self.resolution_system.resolve_conflicts()
        self.notify_stakeholders_of_resolution(resolved_requirements)

    def notify_stakeholders_of_resolution(self, resolved_requirements):
        print(f"Resolution outcome: {resolved_requirements}")
        for stakeholder in self.resolution_system.requirements_database.keys():
            stakeholder.receive_outcome(resolved_requirements)


# Example usage
stakeholder1 = Stakeholder("Stakeholder 1")
stakeholder2 = Stakeholder("Stakeholder 2")
resolution_system = ConflictResolutionSystem()

# Stakeholders provide their requirements
stakeholder1.provide_requirements(resolution_system, "Requirement A")
stakeholder2.provide_requirements(resolution_system, "Requirement B")
