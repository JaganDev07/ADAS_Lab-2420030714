# PROGRAM 2: Parking Assistance System
# Displays warning messages based on the distance between the vehicle and an obstacle.

# Distance readings from the rear ultrasonic sensor (in cm)
distances = [250, 170, 130, 95, 60, 35, 18, 8]

for distance in distances:

    if distance > 150:
        message = "CLEAR - keep reversing"
    elif distance > 100:
        message = "OBSTACLE AHEAD - reverse slowly"
    elif distance > 50:
        message = "CAUTION - obstacle close"
    elif distance > 25:
        message = "WARNING - very close"
    else:
        message = "STOP! - obstacle too close"

    print(f"Distance: {distance:4d} cm  ->  {message}")
