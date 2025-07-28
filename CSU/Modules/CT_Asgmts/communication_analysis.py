import re
import sys

def extract_stakeholders(diagram_file):
    with open(diagram_file, 'r') as file:
        content = file.read()
    
    # Extract stakeholders
    stakeholders = re.findall(r'\|\s+([^\n]+)\s+\|\n', content)
    
    return stakeholders

def count_communication_pathways(diagram_file):
    with open(diagram_file, 'r') as file:
        content = file.read()
    
    # Count the number of communication pathways
    num_communication_pathways = content.count('----')
    
    return num_communication_pathways

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <diagram_file>")
        return
    
    diagram_file = sys.argv[1]
    
    stakeholders = extract_stakeholders(diagram_file)
    num_communication_pathways = count_communication_pathways(diagram_file)
    
    print("Stakeholders:")
    for stakeholder in stakeholders:
        print("-", stakeholder)
    
    print("\nNumber of Communication Pathways:", num_communication_pathways)

if __name__ == "__main__":
    main()
