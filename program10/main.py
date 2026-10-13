"""PROGRAM 10: Adaptive Cruise Control with PID speed adjustment."""


def main() -> None:
    desired_distance = 40
    actual_distance = 30
    our_speed = 80
    lead_speed = 60

    kp = 0.5
    ki = 0.1
    kd = 0.2

    accumulated_error = 5
    rate_of_error = 2

    error = desired_distance - actual_distance
    proportional = kp * error
    integral = ki * accumulated_error
    derivative = kd * rate_of_error
    pid_output = proportional + integral + derivative
    new_speed = our_speed - pid_output

    print("Distance Error =", error, "m")
    print("P =", proportional)
    print("I =", integral)
    print("D =", derivative)
    print("PID Output =", pid_output)
    print("Lead Vehicle Speed =", lead_speed, "km/h")
    print("Our New Speed =", new_speed, "km/h")


if __name__ == "__main__":
    main()
