import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import math
import argparse


def extract_points_optimized(mask, sampling_rate):
    estimated_points = (mask.shape[0] * mask.shape[1]) // (sampling_rate * sampling_rate * 4)
    points = []
    for y in range(0, mask.shape[0], sampling_rate):
        for x in range(0, mask.shape[1], sampling_rate):
            if mask[y, x] > 0:
                points.append((float(x), float(y)))
    return points


def perform_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    if points.shape[0] == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.0
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.arctan2(principal_eigenvector[1], principal_eigenvector[0])
    angle_degrees = np.degrees(angle)
    return principal_eigenvector, centroid, angle_degrees


def get_extreme_points(seg_points, eig_vec, mean):
    seg_points = np.array(seg_points)
    mx, my = mean
    evx, evy = eig_vec
    dx = seg_points[:, 0] - mx
    dy = seg_points[:, 1] - my
    projections = dx * evx + dy * evy
    min_index = np.argmin(projections)
    max_index = np.argmax(projections)
    return tuple(seg_points[min_index]), tuple(seg_points[max_index])


def create_perpendicular_line_mask(image_shape, line_point, line_direction):
    mask = np.zeros(image_shape, dtype=np.uint8)  # Match the shape to the original mask
    line_point_f = np.array(line_point, dtype=np.float32)
    line_direction = np.array(line_direction, dtype=np.float32)
    line_length = np.sqrt(image_shape[1] ** 2 + image_shape[0] ** 2)  # Diagonal length
    start = line_point_f + line_direction * -line_length
    end = line_point_f + line_direction * line_length
    cv2.line(mask, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 255, 1, cv2.LINE_AA)
    return mask


def get_border_points(best_line_mask, mask):
    """
    Find the border points along the longest dimension (x or y) in the intersection of two masks.

    Parameters:
    - best_line_mask: A binary mask representing the best line (NumPy array).
    - mask: The original binary mask (NumPy array).

    Returns:
    - A tuple of two points (min_point, max_point) representing the border points.
    """
    combined_mask = np.logical_and(best_line_mask, mask).astype(np.uint8)
    min_x, max_x, min_y, max_y = combined_mask.shape[1], -1, combined_mask.shape[0], -1
    min_x_point, max_x_point, min_y_point, max_y_point = None, None, None, None
    found_any = False

    for y in range(combined_mask.shape[0]):
        for x in range(combined_mask.shape[1]):
            if combined_mask[y, x] > 0:
                found_any = True
                if x < min_x:
                    min_x = x
                    min_x_point = (x, y)
                if x > max_x:
                    max_x = x
                    max_x_point = (x, y)
                if y < min_y:
                    min_y = y
                    min_y_point = (x, y)
                if y > max_y:
                    max_y = y
                    max_y_point = (x, y)

    if not found_any:
        # Return default values if no points found
        return (-1, -1), (-1, -1)

    # Compare ranges and return the appropriate points
    x_diff = max_x - min_x
    y_diff = max_y - min_y
    if x_diff > y_diff:
        return min_x_point or (-1, -1), max_x_point or (-1, -1)
    else:
        return min_y_point or (-1, -1), max_y_point or (-1, -1)
    

def get_best_mask_per_optimized(length_start, length_end, image_shape, mask, num_points = 30):
    scale = 1.0
    max_dim = max(image_shape[1], image_shape[0])
    if max_dim > 500:
        scale = 500.0 / max_dim
    
    scaled_size = (int(image_shape[1] * scale), int(image_shape[0] * scale))
    scaled_mask = cv2.resize(mask, scaled_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    
    scaled_start = (length_start[0] * scale, length_start[1] * scale)
    scaled_end = (length_end[0] * scale, length_end[1] * scale)
    
    dx = scaled_end[0] - scaled_start[0]
    dy = scaled_end[1] - scaled_start[1]
    vector_length = math.sqrt(dx * dx + dy * dy)
    perpx = -dy / vector_length
    perpy = dx / vector_length
    
    best_line_mask = None
    max_length = 0
    
    for i in range(num_points):
        t = i / (num_points - 1)
        point = (
            scaled_start[0] + t * dx,
            scaled_start[1] + t * dy
        )
        
        local_mask = np.zeros((scaled_size[1], scaled_size[0]), dtype=np.uint8)
        diagonal_length = math.sqrt(scaled_size[0]**2 + scaled_size[1]**2)
        
        line_start = (point[0] + perpx * -diagonal_length, point[1] + perpy * -diagonal_length)
        line_end = (point[0] + perpx * diagonal_length, point[1] + perpy * diagonal_length)
        
        cv2.line(local_mask, (int(line_start[0]), int(line_start[1])), 
                (int(line_end[0]), int(line_end[1])), 65535, 1, cv2.LINE_AA)
        
        intersection = np.logical_and(local_mask, scaled_mask).astype(np.uint8)
        current_length = np.count_nonzero(intersection)
        
        if current_length > max_length:
            max_length = current_length
            best_line_mask = local_mask.copy()
    
    final_mask = cv2.resize(best_line_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
    return final_mask, max_length / scale


def main_compute(seg_res, sampling_rate):
    mask = seg_res
    seg_points = extract_points_optimized(mask, sampling_rate)
    principal_vector, mean, angle = perform_pca(np.array(seg_points))
    cv_mean = (mean[0], mean[1])
    length_start, length_end = get_extreme_points(seg_points, principal_vector, mean)
    width_line_mask = get_best_mask_per_optimized(length_start, length_end, mask.shape, mask, num_points = 100)[0]
    width_start, width_end = get_border_points(width_line_mask, mask)
    
    length_length = np.linalg.norm(np.array(length_end) - np.array(length_start))
    width_length = np.linalg.norm(np.array(width_end) - np.array(width_start))

    # Calculate lengths of the axes
    res =  {
        "length_start_point": length_start,
        "length_end_point": length_end,
        "width_start_point": width_start,
        "width_end_point": width_end,
        "length_length": length_length,
        "width_length": width_length,
        "angle": angle,
        "centroid": cv_mean
        }

    return res


def draw_axes_on_image(image: np.ndarray, results: dict, centroid: Tuple[float, float]):
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # Extract major and minor axes points
    length_start = tuple(map(int, results["length_start_point"]))
    length_end = tuple(map(int, results["length_end_point"]))
    width_start = tuple(map(int, results["width_start_point"]))
    width_end = tuple(map(int, results["width_end_point"]))

    # Draw centroid using cv_mean
    centroid = tuple(map(int, centroid))  # Convert to integer coordinates
    cv2.circle(image_color, centroid, 10, (0, 0, 255), -1)  # Red circle for centroid

    # Draw major and minor axes
    cv2.line(image_color, length_start, length_end, (0, 255, 0), 5)  # Green for length
    cv2.line(image_color, width_start, width_end, (255, 0, 0), 5)  # Blue for width

    # Display lengths
    print(f"Length: {results['length_length']:.2f}")
    print(f"Width: {results['width_length']:.2f}")
    print(f"Centroid (cv_mean): {centroid}")

    # Visualize the axes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title("Length and Width with Centroid")
    plt.axis("off")
    plt.show()



def longest_length_and_width(mask):
    # Find contours
    contours = find_contours(mask, level=0.5)
    if not contours:
        return 0, 0, None
    
    # Assume the largest contour is the object of interest
    contour = max(contours, key=len)

    # Compute pairwise distances between all points on the contour
    pairwise_dists = distance.cdist(contour, contour, 'euclidean')
    
    # Find the maximum distance (longest length)
    max_dist_indices = np.unravel_index(np.argmax(pairwise_dists), pairwise_dists.shape)
    longest_length = pairwise_dists[max_dist_indices]
    
    # Get the points corresponding to the longest length
    point1, point2 = contour[max_dist_indices[0]], contour[max_dist_indices[1]]
    length_points = (point1, point2)
    mid_point = (point1 + point2) / 2

    # Compute the direction vector of the longest length
    direction_vector = point2 - point1
    direction_vector /= np.linalg.norm(direction_vector)
    
    # Compute the perpendicular direction
    perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])

    # Compute orthogonal width's points and length
    distances = np.dot(contour - mid_point, perpendicular_vector)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    width_length = max_distance - min_distance
    width_point1 = mid_point + min_distance * perpendicular_vector
    width_point2 = mid_point + max_distance * perpendicular_vector
    width_points = (width_point1, width_point2)

    return longest_length, width_length, (contour, length_points, width_points)

def plot_contour_length_width(mask, data):
    if data is None:
        print("No valid contour found in the mask.")
        return
    
    contour, length_points, width_points = data

    fig, ax = plt.subplots()
    ax.imshow(mask, cmap=plt.cm.gray)
    
    # Plot the contour
    ax.plot(contour[:, 1], contour[:, 0], 'y-', linewidth=2, label='Contour')
    
    # Plot the longest length
    ax.plot([length_points[0][1], length_points[1][1]], [length_points[0][0], length_points[1][0]], 'r-', linewidth=2, label='Longest Length')
    
    # Plot the longest width perpendicular to the longest length
    if width_points[0] is not None and width_points[1] is not None:
        ax.plot([width_points[0][1], width_points[1][1]], 
                [width_points[0][0], width_points[1][0]], 
                'r-', linewidth=2, label='Perpendicular Width')
    
    # Highlight points
    ax.plot(length_points[0][1], length_points[0][0], 'go')
    ax.plot(length_points[1][1], length_points[1][0], 'go')
    
    ax.legend(loc=2)
    ax.axis('off')
    plt.show()



def main(path):
    # Example binary mask
    seg_res = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Run the main function
    sampling_rate = 1
    results = main_compute(seg_res, sampling_rate)

    # Draw and visualize axes on the binary mask
    cv_mean = results["centroid"]  # Use this as centroid
    draw_axes_on_image(seg_res, results, cv_mean)


def draw_legth_width(mask):
    # Run the main function
    sampling_rate = 1
    results = main_compute(mask, sampling_rate)

    # Draw and visualize axes on the binary mask
    cv_mean = results["centroid"]  # Use this as centroid
    draw_axes_on_image(mask, results, cv_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='mask path')
    args = parser.parse_args()

    main(args.path)