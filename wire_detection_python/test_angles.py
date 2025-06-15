import numpy as np

import wire_detection_utils as wdu

angle_range = 360
angles = np.arange(-angle_range, angle_range+1, 45)
angels_rad = np.deg2rad(angles)

converted_angles = wdu.fold_angles_from_0_to_pi(angels_rad)

for angle, converted_angle in zip(angles, converted_angles):
    print(f"Angle: {angle}, Converted: {np.rad2deg(converted_angle):.2f}")
