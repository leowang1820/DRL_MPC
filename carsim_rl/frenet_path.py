from pathlib import Path

import numpy as np


class FrenetPath:
    def __init__(self, path_file):
        self.path_file = Path(path_file)
        points = []
        reading_table = False

        with self.path_file.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            for raw_line in file:
                line = raw_line.strip()

                if line == "SEGMENT_XY_TABLE":
                    reading_table = True
                    continue

                if reading_table and line == "ENDTABLE":
                    break

                if reading_table:
                    parts = [part.strip() for part in line.split(",")]

                    if len(parts) >= 3:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            station = float(parts[2])
                            points.append((x, y, station))
                        except ValueError:
                            pass

        if len(points) < 2:
            raise RuntimeError(
                f"No valid path points found: {self.path_file}"
            )

        data = np.asarray(points, dtype=np.float64)

        self.x = data[:, 0]
        self.y = data[:, 1]
        self.s = data[:, 2]
        self.path_length = float(self.s[-1])

        self.p0 = np.column_stack((self.x[:-1], self.y[:-1]))
        self.segment = np.column_stack(
            (
                np.diff(self.x),
                np.diff(self.y),
            )
        )

        self.segment_length_squared = np.sum(
            self.segment * self.segment,
            axis=1,
        )

    def project(self, vehicle_x, vehicle_y, vehicle_yaw_deg):
        vehicle_position = np.array(
            [vehicle_x, vehicle_y],
            dtype=np.float64,
        )

        relative = vehicle_position - self.p0

        denominator = np.maximum(
            self.segment_length_squared,
            1e-12,
        )

        projection_ratio = np.sum(
            relative * self.segment,
            axis=1,
        ) / denominator

        projection_ratio = np.clip(
            projection_ratio,
            0.0,
            1.0,
        )

        projected_points = (
            self.p0
            + projection_ratio[:, None] * self.segment
        )

        difference = vehicle_position - projected_points
        distance_squared = np.sum(difference * difference, axis=1)

        index = int(np.argmin(distance_squared))

        dx = self.segment[index, 0]
        dy = self.segment[index, 1]

        lateral_vector = difference[index]

        cross_product = (
            dx * lateral_vector[1]
            - dy * lateral_vector[0]
        )

        lateral_error = np.sqrt(distance_squared[index])

        if cross_product < 0.0:
            lateral_error = -lateral_error

        station = (
            self.s[index]
            + projection_ratio[index]
            * (self.s[index + 1] - self.s[index])
        )

        path_yaw_deg = np.degrees(np.arctan2(dy, dx))

        heading_error_deg = (
            vehicle_yaw_deg - path_yaw_deg + 180.0
        ) % 360.0 - 180.0

        return {
            "station": float(station),
            "lateral_error": float(lateral_error),
            "path_yaw_deg": float(path_yaw_deg),
            "heading_error_deg": float(heading_error_deg),
            "segment_index": index,
        }
