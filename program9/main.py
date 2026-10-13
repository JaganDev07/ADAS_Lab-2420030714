"""PROGRAM 9: Road Surface Classification System.

Classifies a road surface from vehicle speed and vibration measurements using
a distance-weighted K-nearest-neighbours model.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path(__file__).parent / "road_sensor.csv"
FEATURES = ["speed", "accel_rms", "vibration_rms", "roughness_index"]


def main() -> None:
    data = pd.read_csv(DATA_FILE)
    missing_columns = set(FEATURES + ["surface"]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data["surface"],
        test_size=0.25,
        random_state=42,
        stratify=data["surface"],
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    new_reading = pd.DataFrame(
        [[60, 0.35, 0.42, 3.1]],
        columns=FEATURES,
    )
    print("Predicted road surface:", model.predict(new_reading)[0])


if __name__ == "__main__":
    main()
