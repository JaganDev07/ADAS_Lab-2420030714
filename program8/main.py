# PROGRAM 8: Driver Drowsiness Detection System
# Classifies the driver as Alert or Drowsy from eye closure data using a Decision Tree.

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Features: [eye aspect ratio, blink duration (s), blinks per minute]
X_train = [
    [0.32, 0.15, 14], [0.30, 0.18, 16], [0.29, 0.20, 15], [0.33, 0.12, 12],
    [0.31, 0.17, 18], [0.28, 0.22, 19], [0.30, 0.16, 13], [0.34, 0.14, 11],
    [0.18, 0.75, 28], [0.16, 0.90, 31], [0.19, 0.68, 26], [0.15, 1.10, 34],
    [0.17, 0.82, 29], [0.20, 0.65, 25], [0.14, 1.25, 36], [0.19, 0.72, 27],
]
y_train = ["Alert"] * 8 + ["Drowsy"] * 8

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# Test readings from the driver monitoring camera
X_test = [
    [0.31, 0.16, 13],
    [0.17, 0.88, 30],
    [0.27, 0.25, 20],
    [0.13, 1.30, 38],
]
y_test = ["Alert", "Drowsy", "Alert", "Drowsy"]

predictions = model.predict(X_test)

for features, predicted in zip(X_test, predictions):
    ear, duration, rate = features
    alert = "WAKE UP! TAKE A BREAK" if predicted == "Drowsy" else "Driver OK"
    print(f"EAR={ear:.2f}  blink={duration:.2f}s  rate={rate:2d}/min  ->  {predicted:6s} | {alert}")

print("\nAccuracy:", accuracy_score(y_test, predictions))
