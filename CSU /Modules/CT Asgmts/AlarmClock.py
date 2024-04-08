

#Alarm Clock Time Calculation

# Ask the user for the current time and the number of hours to wait
current_time = int(input("What is the current time (in hours, 0-23)? "))
wait_hours = int(input("How many hours do you want to wait for the alarm? "))

# Calculate the time when the alarm goes off
alarm_time = (current_time + wait_hours) % 24

# Display the result
print(f"The alarm will go off at {alarm_time:02d}:00 hours.")

