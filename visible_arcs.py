import numpy as np
import matplotlib.pyplot as plt

def compute_visible_arcs(sdf_points, sdf_values):
    """
    For each circle (center=sdf_point, radius=|sdf_value|), compute the visible arcs
    that are not occluded by other circles.
    
    Parameters:
    -----------
    sdf_points : (N, 2) array
        Centers of circles
    sdf_values : (N,) array
        Signed radii of circles (use absolute value)
    
    Returns:
    --------
    visible_arcs : list of lists
        For each circle i, visible_arcs[i] contains [(start_angle1, end_angle1), (start_angle2, end_angle2), ...]
        Angles are in radians, in range [0, 2π]
    """
    n_circles = len(sdf_points)
    radii = np.abs(sdf_values)
    visible_arcs = []
    arc_points = []
    
    for i in range(n_circles):
        center_i = sdf_points[i]
        radius_i = radii[i]
        
        # Start with full circle [0, 2π]
        visible_intervals = [(0.0, 2.0 * np.pi)]
        
        # Check occlusion from all other circles
        for j in range(n_circles):
            if i == j:
                continue
                
            center_j = sdf_points[j]
            radius_j = radii[j]
            
            # Calculate occlusion intervals
            occluded = compute_occlusion_interval(center_i, radius_i, center_j, radius_j)
            
            if occluded is not None:
                # Subtract occluded interval from visible intervals
                visible_intervals = subtract_intervals(visible_intervals, occluded)
        
        visible_arcs.append(visible_intervals)
        
        # Convert intervals to actual points
        # points_i = intervals_to_points(center_i, radius_i, visible_intervals, num_points_per_arc=50)
        # arc_points.append(points_i)
    
    return visible_arcs


def compute_occlusion_interval(center_i, radius_i, center_j, radius_j):
    """
    Compute the angular interval on circle i that is occluded by circle j.
    
    Returns:
    --------
    (start_angle, end_angle) or None if no occlusion
    Angles are in [0, 2π], may wrap around
    """
    # Vector from i to j
    vec_ij = center_j - center_i
    dist_ij = np.linalg.norm(vec_ij)
    
    if dist_ij < 1e-10:
        # Circles are concentric
        if radius_j >= radius_i:
            # Circle j completely covers circle i
            return (0.0, 2.0 * np.pi)
        else:
            return None
    
    # Check if circle j occludes any part of circle i
    # A point on circle i is occluded if it's inside circle j
    
    # If circles don't intersect or touch, check if one contains the other
    if dist_ij >= radius_i + radius_j:
        # Circles are separate, no occlusion
        return None
    
    if dist_ij + radius_i <= radius_j:
        # Circle i is completely inside circle j
        return (0.0, 2.0 * np.pi)
    
    if dist_ij + radius_j <= radius_i:
        # Circle j is completely inside circle i, no occlusion
        return None
    
    # Circles intersect - compute the angular range of occlusion
    # Points on circle i that are inside circle j
    
    # Angle to center of circle j from center of circle i
    angle_to_j = np.arctan2(vec_ij[1], vec_ij[0])
    
    # Using law of cosines to find the angular half-width of occlusion
    # For a point at angle θ on circle i, distance to center j is:
    # d^2 = radius_i^2 + dist_ij^2 - 2*radius_i*dist_ij*cos(θ - angle_to_j)
    # Point is inside circle j if d < radius_j
    
    # At the boundary: radius_i^2 + dist_ij^2 - 2*radius_i*dist_ij*cos(θ) = radius_j^2
    # cos(θ) = (radius_i^2 + dist_ij^2 - radius_j^2) / (2*radius_i*dist_ij)
    
    cos_half_angle = (radius_i**2 + dist_ij**2 - radius_j**2) / (2 * radius_i * dist_ij)
    
    # Clamp to valid range
    cos_half_angle = np.clip(cos_half_angle, -1.0, 1.0)
    half_angle = np.arccos(cos_half_angle)
    
    # The occluded interval is centered at angle_to_j
    start_angle = angle_to_j - half_angle
    end_angle = angle_to_j + half_angle
    
    # Normalize to [0, 2π]
    start_angle = normalize_angle(start_angle)
    end_angle = normalize_angle(end_angle)
    
    return (start_angle, end_angle)


def normalize_angle(angle):
    """Normalize angle to [0, 2π]"""
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle


def subtract_intervals(intervals, occluded):
    """
    Subtract occluded interval from list of visible intervals.
    Handles wrap-around at 0/2π.
    """
    if occluded is None:
        return intervals
    
    start_occ, end_occ = occluded
    new_intervals = []
    
    # Handle wrap-around case
    wraps = end_occ < start_occ
    
    for start_vis, end_vis in intervals:
        if wraps:
            # Occluded interval wraps around: [start_occ, 2π] and [0, end_occ]
            # This is equivalent to NOT being in (end_occ, start_occ)
            if start_vis >= end_occ and end_vis <= start_occ:
                # Completely in the visible gap
                new_intervals.append((start_vis, end_vis))
            elif start_vis < end_occ and end_vis <= start_occ:
                # Starts in occluded, ends in visible
                if end_vis > end_occ:
                    new_intervals.append((end_occ, end_vis))
            elif start_vis >= end_occ and end_vis > start_occ:
                # Starts in visible, ends in occluded
                if start_vis < start_occ:
                    new_intervals.append((start_vis, start_occ))
            else:
                # Completely occluded or split
                if start_vis < end_occ and end_vis > start_occ:
                    # Split into two parts
                    if end_vis > end_occ:
                        new_intervals.append((end_occ, start_occ))
        else:
            # Normal case: occluded interval doesn't wrap
            if end_vis <= start_occ or start_vis >= end_occ:
                # No overlap
                new_intervals.append((start_vis, end_vis))
            elif start_vis < start_occ and end_vis > end_occ:
                # Visible interval contains occluded - split into two
                new_intervals.append((start_vis, start_occ))
                new_intervals.append((end_occ, end_vis))
            elif start_vis < start_occ and end_vis > start_occ:
                # Partial overlap on right
                new_intervals.append((start_vis, start_occ))
            elif start_vis < end_occ and end_vis > end_occ:
                # Partial overlap on left
                new_intervals.append((end_occ, end_vis))
            # else: completely occluded, don't add
    
    return new_intervals


def intervals_to_points(center, radius, intervals, num_points_per_arc=50):
    """
    Convert angular intervals to actual 2D points on the circle.
    """
    all_points = []
    
    for start_angle, end_angle in intervals:
        # Handle wrap-around
        if end_angle < start_angle:
            end_angle += 2 * np.pi
        
        # Generate points
        n_pts = max(3, int(num_points_per_arc * (end_angle - start_angle) / (2 * np.pi)))
        angles = np.linspace(start_angle, end_angle, n_pts)
        
        points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        all_points.append(points)
    
    if all_points:
        return np.vstack(all_points)
    else:
        return np.empty((0, 2))
    
def find_arcs_neighbors(centers, radii, arcs, tol=1e-2):
    """
    For each circle, find neighboring circles that have spatially overlapping visible arcs.
    
    Parameters:
    -----------
    centers : (N, 2) array
        Centers of circles
    radii : (N,) array
        Radii of circles
    arcs : list of lists
        For each circle, list of (start_angle, end_angle) tuples
    tol : float
        Tolerance for spatial proximity
    
    Returns:
    --------
    neighbors : list of lists
        For each circle i, list of indices j of neighboring circles
    """
    n_circles = len(centers)
    neighbors = [[] for _ in range(n_circles)]
    
    def angle_in_arc(angle, start, end):
        """Check if angle is within arc [start, end], handling wrap-around."""
        angle = normalize_angle(angle)
        if end < start:  # Wrap-around case
            return angle >= start or angle <= end
        else:
            return start <= angle <= end
    
    for i in range(n_circles):
        arcs_i = arcs[i]
        center_i = centers[i]
        radius_i = radii[i]
        
        for j in range(n_circles):
            if i == j:
                continue
            if np.linalg.norm(centers[i] - centers[j]) > (radii[i] + radii[j] + tol):
                continue  # Too far apart to overlap
            arcs_j = arcs[j]
            center_j = centers[j]
            radius_j = radii[j]
            
            # Calculate angle from i to j and from j to i
            vec_ij = center_j - center_i
            angle_i_to_j = np.arctan2(vec_ij[1], vec_ij[0])
            angle_i_to_j = normalize_angle(angle_i_to_j)
            angle_j_to_i = normalize_angle(angle_i_to_j + np.pi)
            
            # Check if any arc from i is close to any arc from j
            found_overlap = False
            for start_i, end_i in arcs_i:
                if found_overlap:
                    break
                
                # Get boundary points of arc i
                pt_start_i = center_i + radius_i * np.array([np.cos(start_i), np.sin(start_i)])
                pt_end_i = center_i + radius_i * np.array([np.cos(end_i), np.sin(end_i)])
                
                for start_j, end_j in arcs_j:
                    # Get boundary points of arc j
                    pt_start_j = center_j + radius_j * np.array([np.cos(start_j), np.sin(start_j)])
                    pt_end_j = center_j + radius_j * np.array([np.cos(end_j), np.sin(end_j)])
                    
                    # Check if closest point on arc i towards j is in arc i
                    # and closest point on arc j towards i is in arc j
                    if angle_in_arc(angle_i_to_j, start_i, end_i) and angle_in_arc(angle_j_to_i, start_j, end_j):
                        neighbors[i].append(j)
                        found_overlap = True
                        break
                    
                    # Also check pairwise distances between boundary points
                    distances = [
                        np.linalg.norm(pt_start_i - pt_start_j),
                        np.linalg.norm(pt_start_i - pt_end_j),
                        np.linalg.norm(pt_end_i - pt_start_j),
                        np.linalg.norm(pt_end_i - pt_end_j)
                    ]
                    if min(distances) < tol:
                        neighbors[i].append(j)
                        found_overlap = True
                        break
    
    return neighbors

def visualize_visible_arcs(sdf_points, sdf_values, visible_arcs, original_shape=None):
    """
    Visualize the circles and their visible arcs.
    """
    from matplotlib.patches import Arc
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    radii = np.abs(sdf_values)
    
    # Draw all circles faintly
    for i, (center, radius) in enumerate(zip(sdf_points, radii)):
        circle = plt.Circle(center, radius, fill=False, edgecolor='lightgray', linestyle='--', alpha=0.3)
        ax.add_patch(circle)
    
    # Draw visible arcs using Arc patches
    colors = plt.cm.rainbow(np.linspace(0, 1, len(visible_arcs)))
    for i, arcs in enumerate(visible_arcs):
        center = sdf_points[i]
        radius = radii[i]
        
        for start_angle, end_angle in arcs:
            # Convert radians to degrees
            start_deg = np.degrees(start_angle)
            end_deg = np.degrees(end_angle)
            
            # Handle wrap-around: if end < start, the arc crosses 0
            if end_deg < start_deg:
                # Split into two arcs: [start, 360] and [0, end]
                arc1 = Arc(xy=center, width=2*radius, height=2*radius,
                          angle=0, theta1=start_deg, theta2=360,
                          color=colors[i], linewidth=2.5)
                arc2 = Arc(xy=center, width=2*radius, height=2*radius,
                          angle=0, theta1=0, theta2=end_deg,
                          color=colors[i], linewidth=2.5)
                ax.add_patch(arc1)
                ax.add_patch(arc2)
            else:
                arc = Arc(xy=center, width=2*radius, height=2*radius,
                         angle=0, theta1=start_deg, theta2=end_deg,
                         color=colors[i], linewidth=2.5)
                ax.add_patch(arc)
    
    # Draw centers
    ax.scatter(sdf_points[:, 0], sdf_points[:, 1], c='red', s=3, zorder=5, label='Centers')
    
    # Draw original shape if provided
    if original_shape is not None:
        ax.plot(original_shape[:, 0], original_shape[:, 1], 'b-', linewidth=1, label='Original Shape')
    
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig, ax


# Test function
if __name__ == "__main__":
    # Simple test with 3 circles
    sdf_points = np.array([
        [0.3, 0.5],
        [0.7, 0.5],
        [0.5, 0.7]
    ])
    sdf_values = np.array([0.2, 0.25, 0.15])
    
    visible_arcs = compute_visible_arcs(sdf_points, sdf_values)
    
    print("Visible arcs (in radians):")
    for i, arcs in enumerate(visible_arcs):
        print(f"Circle {i}: {arcs}")
    
    fig, ax = visualize_visible_arcs(sdf_points, sdf_values, visible_arcs)
    plt.show()