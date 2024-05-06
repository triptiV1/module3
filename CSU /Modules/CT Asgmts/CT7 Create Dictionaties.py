# Dictionary mapping course numbers to room numbers
course_to_room = {
    'CSC101': '3004',
    'CSC102': '4501',
    'CSC103': '6755',
    'NET110': '1244',
    'COM241': '1411'
}

# Dictionary mapping course numbers to instructors
course_to_instructor = {
    'CSC101': 'Haynes',
    'CSC102': 'Alvarado',
    'CSC103': 'Rich',
    'NET110': 'Burke',
    'COM241': 'Lee'
}

# Dictionary mapping course numbers to meeting times
course_to_meeting_time = {
    'CSC101': '8:00 a.m.',
    'CSC102': '9:00 a.m.',
    'CSC103': '10:00 a.m.',
    'NET110': '11:00 a.m.',
    'COM241': '1:00 p.m.'
}

# Function to display course information
def display_course_info(course_number):
    room = course_to_room.get(course_number)
    instructor = course_to_instructor.get(course_number)
    meeting_time = course_to_meeting_time.get(course_number)
    
    if room and instructor and meeting_time:
        print(f"Course Number: {course_number}")
        print(f"Room Number: {room}")
        print(f"Instructor: {instructor}")
        print(f"Meeting Time: {meeting_time}")
    else:
        print("Course not found.")

# Main program loop
def main():
    while True:
        course_number = input("Enter a course number (or 'exit' to quit): ")
        if course_number.lower() == 'exit':
            break
        display_course_info(course_number.upper())

# Run the main program
if __name__ == "__main__":
    main()
