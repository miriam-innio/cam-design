import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import find_peaks
import os

file_path = 'csp-683423.txt'
window = 50 #points for plateau
base_radius = 35  # Base radius for cam in mm
#set how many files you want
desired_squish = 2
desired_stretch = 2
a=0.95 #random scaling factors for x data (from)
b=1.05 #random scaling factors for x data (to)

c=0 #random scaling factors for y data stretch (from)
d=0.0001 #random scaling factors for y data strech(to)

e=0 #random scaling factors for y data squish (from)
h=0.0001 #random scaling factors for y data squish (to)

#check ration: abs(peak positive acceleration / peak negative acceleration)
ratio_lower= 0.3
ratio_upper=1.2

#check for peak cam lift / duration
lift_boundary= 0.05

#negative radius of curvature 
peak_max= 0 #boundary for finding peaks
peak_min=-100

problem_peak_max=0 #boundary for problem peaks that get filtered out
problem_peak_min= -60

x_data = [] #cam angle in deg
y_data = [] #cam lift in mm


with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        x_data.append(float(parts[0]))
        y_data.append(float(parts[1]))

x_data = np.array(x_data)
y_data = np.array(y_data)

output_folder = r'C:\Users\HagerMi00\Documents\Python\cam design\plot_data'
output_file_original = os.path.join(output_folder, 'plot_data_original.txt')

with open(output_file_original, 'w') as f:
    f.write('x_data\ty_data\n')
    for i in range(len(x_data)):
        f.write(f"{x_data[i]}\t{y_data[i]}\n")
print(f"Saved original data as {output_file_original}")


velocity = np.gradient(y_data, x_data)
acceleration = np.gradient(velocity, x_data)


zero_threshold = 0.1 * np.max(np.abs(acceleration))


# find 1. plateau
plateau_idx = None
for i in range(len(acceleration) - window):
    if np.all(np.abs(acceleration[i:i+window]) < zero_threshold):
        plateau_idx = i
        break

# acc not 0
start_idx = None
if plateau_idx is not None:
    for i in range(plateau_idx + window, len(acceleration)):
        if np.abs(acceleration[i]) >= zero_threshold:
            start_idx = i
            break

#find last plateau
end_plateau_idx = None
for i in range(len(acceleration) - window, 0, -1):
    if np.all(np.abs(acceleration[i:i+window]) < zero_threshold):
        end_plateau_idx = i
        break

#acc not 0
end_start_idx = None
if end_plateau_idx is not None:
    for i in range(end_plateau_idx - 1, 0, -1):
        if np.abs(acceleration[i]) >= zero_threshold:
            end_start_idx = i
            break


# exp scaling y data
def exponential_stretch(y, scale_factor, mode):
    """
    mode: 'stretch' for exponential stretching, 'squish' for logarithmic squishing.
    scale_factor: positive for effect strength, 0 for no change.
    """
    if mode == 'stretch':
        y_scaled = y + scale_factor * (np.exp(y) - y - 1)
    elif mode == 'squish':
        epsilon = 1e-8  # Small value to avoid log(0)
        y_safe = np.where(y <= 0, epsilon, y)
        y_scaled = y + scale_factor * (np.log(y_safe + 1) - y)
    else:
        raise ValueError("mode must be 'stretch' or 'squish'")
    return y_scaled

scaled_x_list = []
scaled_y_list = []
scaled_velocity_list = []
scaled_acceleration_list = []


ok_squish = 0
ok_stretch = 0
run = 0
ok_files_saved=0

while ok_squish < desired_squish or ok_stretch < desired_stretch:
    run += 1
    # Decide which mode to use
    if ok_squish < desired_squish and ok_stretch < desired_stretch:
        mode = random.choice(['squish', 'stretch'])
    elif ok_squish < desired_squish:
        mode = 'squish'
    else:
        mode = 'stretch'
    # Random scaling factors
    x_scale_factor = random.uniform(a, b)

    if mode == 'stretch':
        y_scale_factor = random.uniform(c, d)
    else:  # mode == 'squish'
        y_scale_factor = random.uniform(e, h)

    y_data_scaled = y_data.copy()
    x_data_scaled = x_data.copy()

    # Only middle part changed
    if start_idx is not None and end_start_idx is not None and end_start_idx > start_idx:
        y_region = y_data[start_idx:end_start_idx]
        y_region_scaled = exponential_stretch(y_region, y_scale_factor, mode)

        offset_start = y_region_scaled[0] - y_region[0]
        offset_end = y_region_scaled[-1] - y_region[-1]
        n = len(y_region)
        blend = np.linspace(0, 1, n)
        offset = offset_start * (1 - blend) + offset_end * blend

        y_region_scaled -= offset
        y_data_scaled[start_idx:end_start_idx] = y_region_scaled

        # lin scaling x data
        x_region = x_data[start_idx:end_start_idx]
        x0 = x_region[0]
        x1 = x_region[-1]
        region_length = x1 - x0

        x_data_scaled[start_idx:end_start_idx] = x0 + x_scale_factor * (x_region - x0)
        shift = x_data_scaled[end_start_idx-1] - x1
        x_data_scaled[end_start_idx:] += shift

    velocity_scaled = np.gradient(y_data_scaled, x_data_scaled)
    acceleration_scaled = np.gradient(velocity_scaled, x_data_scaled)

    # Calculate peak positive and negative acceleration
    peak_pos = np.max(acceleration_scaled)
    peak_neg = np.min(acceleration_scaled)
    if peak_neg == 0:
        ratio = np.inf  # :0
    else:
        ratio = abs(peak_pos / peak_neg)

    # Check ratio: abs(peak positive acceleration / peak negative acceleration)
    ratio_ok = ratio_lower <= ratio <= ratio_upper
    print(f"Run {run+1}: Ratio = {ratio:.3f} -> {'OK' if ratio_ok else 'NOT OK'}")
    
    #check for peak cam lift / duration
    peak_lift = np.max(y_data_scaled)
    duration = np.max(x_data_scaled) - np.min(x_data_scaled)
    lift_per_deg = peak_lift / duration

    lift_ok = lift_per_deg >= lift_boundary
    print(f"Run {run}: Peak lift/duration = {lift_per_deg:.4f} mm/deg -> {'OK' if lift_ok else 'NOT OK'}")


    theta_data = np.deg2rad(x_data)  # Convert to radians if needed
    r_actual = base_radius + y_data

    # Polar signed radius of curvature
    dr_dtheta = np.gradient(r_actual, theta_data)
    d2r_dtheta2 = np.gradient(dr_dtheta, theta_data)
    numerator = (r_actual**2 + dr_dtheta**2)**1.5
    denominator = r_actual**2 + 2*dr_dtheta**2 - r_actual*d2r_dtheta2
    radius_of_curvature_signed = numerator / denominator

    # Find local maxima
    peaks, _ = find_peaks(radius_of_curvature_signed)
    # Filter peaks between -100 and 0
    filtered_peaks = [i for i in peaks if peak_min < radius_of_curvature_signed[i] < peak_max]

    # maxima between 0 and -60
    problematic_peaks = [i for i in filtered_peaks if problem_peak_min < radius_of_curvature_signed[i] < problem_peak_max]

    curvature_ok = not problematic_peaks
    if curvature_ok:
        print("OK: No local maxima of radius of curvature between 0 and -60 detected.")
    else:
        max_of_maximums = np.max([radius_of_curvature_signed[i] for i in problematic_peaks])
        print(f"NOT OK: Local maximum of radius of curvature between 0 and -60 detected. Maximum value: {max_of_maximums:.2f}")
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, radius_of_curvature_signed, label='Signed Radius of Curvature', color='b')
    plt.scatter(
        np.array(x_data)[filtered_peaks],
        radius_of_curvature_signed[filtered_peaks],
        color='r',
        label='Local Maxima (-100 to 0)',
        zorder=5
    )

    plt.xlabel('Cam Angle (deg)')
    plt.ylabel('Signed Radius of Curvature (mm)')
    plt.title('Signed Radius of Curvature vs Cam Angle')
    plt.legend()
    plt.grid(True)
    plt.ylim(-350, 350)
    plt.tight_layout()
    plt.show()
    '''
    # Save only if all checks are OK
    if ratio_ok and lift_ok and curvature_ok:
        output_folder = r'C:\Users\HagerMi00\Documents\Python\cam design\plot_data'
        output_file = os.path.join(output_folder, f'plot_data_scaled_{ok_files_saved+1}.txt')

        with open(output_file, 'w') as f:
            f.write('x_data_scaled\ty_data_scaled\n')
            for i in range(len(x_data_scaled)):
                f.write(f"{x_data_scaled[i]}\t{y_data_scaled[i]}\n")

        scaled_x_list.append(x_data_scaled.copy())
        scaled_y_list.append(y_data_scaled.copy())
        scaled_velocity_list.append(velocity_scaled.copy())
        scaled_acceleration_list.append(acceleration_scaled.copy())
        ok_files_saved += 1
        if mode == 'squish':
            ok_squish += 1
        else:
            ok_stretch += 1
        print(f"Saved OK file {ok_files_saved} as {output_file} ({mode})")

velocity_scaled = np.gradient(y_data_scaled, x_data_scaled)
acceleration_scaled = np.gradient(velocity_scaled, x_data_scaled)


# Plot all scaled datasets
fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

axs[0].plot(x_data, y_data, label='Original', linewidth=2)
for i in range(len(scaled_x_list)):
    axs[0].plot(scaled_x_list[i], scaled_y_list[i], label=f'Scaled {i+1}', linewidth=1)
    axs[0].set_ylabel('Y Data (mm)')
    axs[0].set_title('Original and Scaled Data')
    axs[0].legend()
    axs[0].grid(True)

axs[1].plot(x_data, velocity, label='Original Velocity', linewidth=2)
for i in range(len(scaled_x_list)):
    axs[1].plot(scaled_x_list[i], scaled_velocity_list[i], label=f'Scaled Velocity {i+1}', linewidth=1)
    axs[1].set_ylabel('Velocity (mm/deg)')
    axs[1].set_title('Velocity')
    axs[1].legend()
    axs[1].grid(True)

axs[2].plot(x_data, acceleration, label='Original Acceleration', linewidth=2)
for i in range(len(scaled_x_list)):
    axs[2].plot(scaled_x_list[i], scaled_acceleration_list[i], label=f'Scaled Acceleration {i+1}', linewidth=1)
    axs[2].set_xlabel('X Data (deg)')
    axs[2].set_ylabel('Acceleration (mm/deg²)')
    axs[2].set_title('Acceleration')
    axs[2].legend()
    axs[2].grid(True)

plt.tight_layout()
plt.show()
