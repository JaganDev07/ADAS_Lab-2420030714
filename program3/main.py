# PROGRAM 3: Forward Collision Warning System
# Alerts the driver when the distance to the vehicle ahead becomes unsafe.

speed_kmph = 60.0          # own vehicle speed
reaction_time = 1.5        # driver reaction time in seconds

speed_mps = speed_kmph * 1000 / 3600
safe_distance = speed_mps * reaction_time

print(f"Speed: {speed_kmph} km/h   Safe distance: {safe_distance:.1f} m\n")

# Distance to the vehicle ahead (in metres), one reading per frame
distances = [80.0, 60.0, 45.0, 30.0, 20.0, 12.0, 6.0]

for distance in distances:

    time_to_collision = distance / speed_mps

    if distance < safe_distance * 0.5:
        warning = "CRITICAL: COLLISION WARNING!"
    elif distance < safe_distance:
        warning = "CAUTION: REDUCE SPEED"
    else:
        warning = "SAFE"

    print(f"Distance: {distance:5.1f} m   TTC: {time_to_collision:4.1f} s   ->  {warning}")
