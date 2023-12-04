import numpy as np
import cv2
import typing


def get_focal_length(image_path: str) -> int:
    """Extract exif:FocalLengthIn35mmFilm from image

    Args:
        image_path (str): path to the image

    Returns:
        int: focal length in 35mm film
    """
    import exifread

    with open(image_path, "rb") as f:
        exif = exifread.process_file(f)
        focal_length = exif["EXIF FocalLengthIn35mmFilm"].values[0]
    return focal_length


def to_uv_points(points: np.ndarray) -> np.ndarray:
    """Convert points to uv coordinates

    Args:
        points (np.ndarray): points to convert (N, 2)

    Returns:
        np.ndarray: uv coordinates (N, 2)
    """
    return np.concatenate(
        [
            points,
            np.ones((points.shape[0], 1)),
        ],
        axis=1,
    )


def to_camera_coordinates(
    points: np.ndarray, intrinsic_matrix: np.ndarray, depth: float = 1
) -> np.ndarray:
    """Convert points to camera coordinates

    Args:
        points (np.ndarray): points to convert (N, 2)
        intrinsic_matrix (np.ndarray): intrinsic matrix (3, 3)
        depth (float, optional): depth of the points. Defaults to 1.

    Returns:
        np.ndarray: camera coordinates (N, 3)
    """
    return (np.linalg.inv(intrinsic_matrix) @ to_uv_points(points).T * depth).T


def cast_to(origin: np.ndarray, direction: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Cast ray to line in batch

    Args:
        origin (np.ndarray): origin of the ray (N, 2) or (2,)
        direction (np.ndarray): direction of the ray (N, 2)
        line (np.ndarray): line to cast to (N, 3), represent as ax + by + c = 0

    Returns:
        np.ndarray: intersection point (N, 2)
    """

    # Convert to homogeneous coordinates
    n = line[:, :2]  # (N, 2)
    t = -(np.sum(n * origin, axis=-1) + line[:, 2]) / np.sum(
        n * direction, axis=-1
    )  # (N, 1)
    intersection_points = origin + t.reshape(-1, 1) * direction  # (N, 2)
    assert np.allclose(np.sum(n * intersection_points, axis=-1) + line[:, 2], 0)
    return intersection_points


def read_points(points_path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Read points from file

    Args:
        point_path (str): path to the points file
    """
    points = np.loadtxt(points_path, delimiter=",")  # x1, y1, x2, y2, x3, y3
    # Convert to 3x2 matrix
    points = points.reshape((3, 2))  # (3, 2)
    # points = points / np.array([image.shape[1], image.shape[0]]) # Normalize points
    vanishing_point = points[2]  # (2,)
    # Convert 2 points to 4 points
    rectangle_points = np.concatenate(
        [
            np.array([[points[1, 0], points[0, 1]]]),
            points[:2],
            np.array([[points[0, 0], points[1, 1]]]),
        ],
        axis=0,
    )  # (4, 2)
    return vanishing_point.astype(np.int32), rectangle_points.astype(np.int32)


def compute_homography(
    correspondences_1: np.ndarray, correspondences_2: np.ndarray
) -> np.ndarray:
    """Compute homography matrix from two sets of corresponding points, solve using SVD

    Args:
        correspondences_1 (np.ndarray): (N, 2)
        correspondences_2 (np.ndarray): (N, 2)

    Returns:
        np.ndarray: homography matrix (3, 3)
    """

    N = correspondences_1.shape[0]
    A = np.zeros((2 * N, 9))
    for i in range(N):
        x1, y1 = correspondences_1[i]
        x2, y2 = correspondences_2[i]
        A[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H


def warp_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Inverse warping points using homography matrix

    Args:
        points (np.ndarray): (N, 2)
        H (np.ndarray): (3, 3)

    Returns:
        np.ndarray: (N, 2)
    """

    N = points.shape[0]
    points = np.concatenate([points, np.ones((N, 1))], axis=1)
    transformed_points = np.matmul(H, points.T).T
    transformed_points = transformed_points / transformed_points[:, -1:]
    return transformed_points[:, :2]


def warp_to_canvas(
    canvas: np.ndarray, image: np.ndarray, matrix_H: np.ndarray
) -> np.ndarray:
    """Warp image to canvas

    Args:
        canvas (np.ndarray): canvas to warp to, (N, M, 3)
        image (np.ndarray): image to warp, (H, W, 3)
        matrix_H (np.ndarray): homography matrix, (3, 3)
    """

    # Get canvas size
    height = canvas.shape[0]
    width = canvas.shape[1]

    # Get canvas points
    canvas_points = np.meshgrid(np.arange(width), np.arange(height))  # (2, H, W)
    canvas_points = np.stack(canvas_points, axis=-1).reshape(-1, 2)  # (N, 2)
    center_point = np.array([width // 2, height // 2])
    canvas_points = canvas_points

    # Warp canvas points to image
    warped_canvas_points = warp_points(canvas_points, matrix_H)
    warped_canvas_points = warped_canvas_points.astype(np.int32)
    mask_x = (warped_canvas_points[:, 0] >= 0) & (
        warped_canvas_points[:, 0] < image.shape[1]
    )
    mask_y = (warped_canvas_points[:, 1] >= 0) & (
        warped_canvas_points[:, 1] < image.shape[0]
    )
    mask = mask_x & mask_y

    masked_canvas_points = canvas_points[mask]
    masked_warped_points = warped_canvas_points[mask]
    # canvas[masked_canvas_points[:, 1], masked_canvas_points[:, 0]] = 1
    canvas[masked_canvas_points[:, 1], masked_canvas_points[:, 0]] = image[
        masked_warped_points[:, 1], masked_warped_points[:, 0]
    ]
    return canvas


def visualize_points(image, points):
    import matplotlib.pyplot as plt

    plt.imshow(image)
    plt.plot(points[:, 0], points[:, 1], "r-")
    plt.scatter(points[0, 0], points[0, 1], c="r")
    plt.show()


def dist(p1, p2):
    return np.sum((p1 - p2) ** 2) ** 0.5


image_path = "/Users/qingchengzhao/Nutstore Files/Nutstore/课程/Term-V/CS180/Final/TIP/data/corridor.jpg"
points_path = "/Users/qingchengzhao/Nutstore Files/Nutstore/课程/Term-V/CS180/Final/TIP/data/corridor.txt"
front_wall_z = 3000  # pixels

image = cv2.imread(image_path)[:, :, ::-1] / 255.0  # (H, W, 3)

# focal_length = get_focal_length(image_path)
# print(f"Focal Length: {focal_length}")
# To pixel unit
# focal_length_x = focal_length / 36 * image.shape[1]
# focal_length_y = focal_length / 36 * image.shape[0]
focal_length_x = 800  # pixels
focal_length_y = 800  # pixels
center_point = np.array([image.shape[1] / 2, image.shape[0] / 2])

# Create intrinsic camera matrix
intrinsic_matrix = np.array(
    [
        [focal_length_x, 0, image.shape[1] / 2],
        [0, focal_length_y, image.shape[0] / 2],
        [0, 0, 1],
    ]
)
vanishing_point, rectangle_points = read_points(points_path)

# Convert to camera coordinates
front_wall_camera_points = to_camera_coordinates(
    rectangle_points, intrinsic_matrix, front_wall_z
)  # (4, 3)
wall_width = front_wall_z
wall_height = dist(front_wall_camera_points[1], front_wall_camera_points[3])

# Compute front wall
front_wall_image = image[
    rectangle_points[1, 1] : rectangle_points[2, 1],
    rectangle_points[1, 0] : rectangle_points[2, 0],
]
# visualize_points(image, rectangle_points)
# visualize_points(front_wall_image, rectangle_points - rectangle_points[1])

# Compute left wall
left_border_points = cast_to(
    vanishing_point,
    rectangle_points[[1, 3]] - vanishing_point,
    np.array([[1, 0, 0], [1, 0, 0]]),
)
left_wall_points = np.concatenate(
    [
        rectangle_points[1],  # Up right corner, in image plane
        left_border_points[0, :2],  # Up left corner, casting to image plane
        rectangle_points[3],  # Down right corner, in image plane
        left_border_points[1, :2],  # Down left corner, casting to image plane
    ],
    axis=0,
).reshape(
    4, 2
)  # (4, 2)
# visualize_points(image, left_wall_points)
# visualize_points(image, front_wall_camera_points[:, :2] + center_point)
left_wall_image_points = np.array(
    [
        [wall_width, 0],
        [0, 0],
        [wall_width, wall_height],
        [0, wall_height],
    ]
)

# Compute homography matrix
left_wall_H = compute_homography(left_wall_image_points, left_wall_points)
left_wall_image = np.zeros((int(wall_height), int(wall_width), 3))
# Inverse warp points
left_wall_image = warp_to_canvas(left_wall_image, image, left_wall_H)
# visualize_points(left_wall_image, left_wall_image_points)
left_wall_camera_points = np.concatenate(
    [
        front_wall_camera_points[1],
        front_wall_camera_points[1] - np.array([0, 0, wall_width]),
        front_wall_camera_points[3],
        front_wall_camera_points[3] - np.array([0, 0, wall_width]),
    ],
    axis=0,
).reshape(4, 3)


# Compute right wall
right_border_points = cast_to(
    vanishing_point,
    rectangle_points[[0, 2]] - vanishing_point,
    np.array([[1, 0, -image.shape[1]], [1, 0, -image.shape[1]]]),
)
right_wall_points = np.concatenate(
    [
        rectangle_points[0],  # Up left corner, in image plane
        right_border_points[0, :2],  # Up right corner, casting to image plane
        rectangle_points[2],  # Down left corner, in image plane
        right_border_points[1, :2],  # Down right corner, casting to image plane
    ],
    axis=0,
).reshape(
    4, 2
)  # (4, 2)
# visualize_points(image, right_wall_points)
# visualize_points(image, front_wall_camera_points[:, :2] + center_point)
right_wall_image_points = np.array(
    [
        [0, 0],
        [wall_width, 0],
        [0, wall_height],
        [wall_width, wall_height],
    ]
)

# Compute homography matrix
right_wall_H = compute_homography(right_wall_image_points, right_wall_points)
right_wall_image = np.zeros((int(wall_height), int(wall_width), 3))
# Inverse warp points
right_wall_image = warp_to_canvas(right_wall_image, image, right_wall_H)
# visualize_points(right_wall_image, right_wall_image_points)
right_wall_camera_points = np.concatenate(
    [
        front_wall_camera_points[0],
        front_wall_camera_points[0] - np.array([0, 0, wall_width]),
        front_wall_camera_points[2],
        front_wall_camera_points[2] - np.array([0, 0, wall_width]),
    ],
    axis=0,
).reshape(4, 3)

# Compute top wall
top_border_points = cast_to(
    vanishing_point,
    rectangle_points[[0, 1]] - vanishing_point,
    np.array([[0, 1, 0], [0, 1, 0]]),
)
top_wall_points = np.concatenate(
    [
        rectangle_points[0],  # Up left corner, in image plane
        top_border_points[0, :2],  # Up right corner, casting to image plane
        rectangle_points[1],  # Down left corner, in image plane
        top_border_points[1, :2],  # Down right corner, casting to image plane
    ],
    axis=0,
).reshape(
    4, 2
)  # (4, 2)
# visualize_points(image, top_wall_points)
# visualize_points(image, front_wall_camera_points[:, :2] + center_point)
top_wall_image_points = np.array(
    [
        [0, 0],
        [wall_width, 0],
        [0, wall_height],
        [wall_width, wall_height],
    ]
)

# Compute homography matrix
top_wall_H = compute_homography(top_wall_image_points, top_wall_points)
top_wall_image = np.zeros((int(wall_height), int(wall_width), 3))
# Inverse warp points
top_wall_image = warp_to_canvas(top_wall_image, image, top_wall_H)
# visualize_points(top_wall_image, top_wall_image_points)

top_wall_camera_points = np.concatenate(
    [
        front_wall_camera_points[0],
        front_wall_camera_points[0] - np.array([0, 0, wall_width]),
        front_wall_camera_points[1],
        front_wall_camera_points[1] - np.array([0, 0, wall_width]),
    ],
    axis=0,
).reshape(4, 3)

# Compute bottom wall
bottom_border_points = cast_to(
    vanishing_point,
    rectangle_points[[2, 3]] - vanishing_point,
    np.array([[0, 1, -image.shape[0]], [0, 1, -image.shape[0]]]),
)
bottom_wall_points = np.concatenate(
    [
        rectangle_points[2],  # Up left corner, in image plane
        bottom_border_points[0, :2],  # Up right corner, casting to image plane
        rectangle_points[3],  # Down left corner, in image plane
        bottom_border_points[1, :2],  # Down right corner, casting to image plane
    ],
    axis=0,
).reshape(
    4, 2
)  # (4, 2)
# visualize_points(image, bottom_wall_points)
# visualize_points(image, front_wall_camera_points[:, :2] + center_point)
bottom_wall_image_points = np.array(
    [
        [wall_width, 0],
        [0, 0],
        [wall_width, wall_height],
        [0, wall_height],
    ]
)

# Compute homography matrix
bottom_wall_H = compute_homography(bottom_wall_image_points, bottom_wall_points)
bottom_wall_image = np.zeros((int(wall_height), int(wall_width), 3))
# Inverse warp points
bottom_wall_image = warp_to_canvas(bottom_wall_image, image, bottom_wall_H)
# visualize_points(bottom_wall_image, bottom_wall_image_points)
bottom_wall_camera_points = np.concatenate(
    [
        front_wall_camera_points[2],
        front_wall_camera_points[2] - np.array([0, 0, wall_width]),
        front_wall_camera_points[3],
        front_wall_camera_points[3] - np.array([0, 0, wall_width]),
    ],
    axis=0,
).reshape(4, 3)


import uuid
import os

# Create temporary directory
temp_dir = os.path.join("/tmp", str(uuid.uuid4()))
os.mkdir(temp_dir)
print(f"Temp dir: {temp_dir}")

# Save images
cv2.imwrite(
    os.path.join(temp_dir, "front_wall.jpg"), front_wall_image[:, :, ::-1] * 255
)
cv2.imwrite(os.path.join(temp_dir, "left_wall.jpg"), left_wall_image[:, :, ::-1] * 255)
cv2.imwrite(
    os.path.join(temp_dir, "right_wall.jpg"), right_wall_image[:, :, ::-1] * 255
)
cv2.imwrite(os.path.join(temp_dir, "top_wall.jpg"), top_wall_image[:, :, ::-1] * 255)
cv2.imwrite(
    os.path.join(temp_dir, "bottom_wall.jpg"), bottom_wall_image[:, :, ::-1] * 255
)

try:
    import bpy
except ImportError:
    print("This script must be run from Blender")
    quit()


def set_image(image_path, vertices):
    bpy.ops.import_image.to_plane(shader="SHADELESS", files=[{"name": image_path}])
    image_plane = bpy.context.selected_objects[0]
    image_plane.data.vertices[0].co = vertices[3]
    image_plane.data.vertices[1].co = vertices[2]
    image_plane.data.vertices[2].co = vertices[1]
    image_plane.data.vertices[3].co = vertices[0]
    return image_plane


# Delete all planes
for obj in bpy.data.objects:
    if obj.type == "MESH":
        bpy.data.objects.remove(obj)

set_image(os.path.join(temp_dir, "front_wall.jpg"), front_wall_camera_points / 50)
set_image(os.path.join(temp_dir, "left_wall.jpg"), left_wall_camera_points / 50)
set_image(os.path.join(temp_dir, "right_wall.jpg"), right_wall_camera_points / 50)
set_image(os.path.join(temp_dir, "top_wall.jpg"), top_wall_camera_points / 50)
set_image(os.path.join(temp_dir, "bottom_wall.jpg"), bottom_wall_camera_points / 50)
