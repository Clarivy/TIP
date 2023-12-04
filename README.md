# Tour into Picture

## Introduction

This project implements a "tour into picture" algorithm based on the paper [TIP](http://graphics.cs.cmu.edu/courses/15-463/2011_fall/Papers/TIP.pdf). The algorithm assumes the world is a box and the camera is perpendicular to one of its faces. The project includes a GUI for image annotation and integration with [Blender](https://www.blender.org/) for 3D visualization.

Example of a 2D image and its conversion to 3D using the algorithm:
| 3D Image | 2D Model |
| --- | --- |
| ![Moffit](result/moffit.gif) *Moffitt Library* | ![Moffit Image](data/moffit.jpg) |

## Usage

### Dependencies

- Python 3.7 or higher
- NumPy
- OpenCV-Python
- Matplotlib
- Blender (with "Import-Export: Import Images as Planes" addon)

### Running the Program

To use the GUI for annotation:
```bash
python app.py
```

To reconstruct the 3D model:
```bash
python main.py
```

For Blender integration, paste `main.py` into the Blender text editor and run it.

## GUI Interface

The GUI allows users to draw rectangles to indicate the front wall and select the vanishing point. Annotations can be saved for further processing. 

![GUI](images/gui.png)


## 3D Animation & Results

The program generates a 3D box model in Blender. Below are examples of the generated 3D scenes:

| Result | Original Image |
| --- | --- |
| ![Office](result/office.gif) *Berkeley School of Education* | ![Office Image](data/office.jpg) |
| ![Moffit](result/moffit.gif) *Moffitt Library* | ![Moffit Image](data/moffit.jpg) |
| ![Corridor](result/corridor.gif) *Moffitt Library Corridor* | ![Corridor Image](data/corridor.jpg) |

## Method

The core of the 'Tour into Picture' algorithm lies in its unique approach to 3D reconstruction from a single image. The method hinges on a fundamental assumption: the world is a 'box', and the camera capturing the image is perpendicular to one of the box's walls (usually the front wall).

## Vanishing Point and Perspective

In any given scene, lines that are parallel in the real world but appear to converge in an image, meet at what's called the vanishing point. The algorithm leverages this concept to reconstruct three-dimensional space from a two-dimensional photo. By identifying the vanishing point in an image, we can delineate the perspective and spatial orientation of the scene.

## Scale Ambiguity and Camera Calibration

Due to the inherent scale ambiguity in monocular images (images captured with a single lens), the algorithm employs a calibrated camera model. This allows us to select a suitable z-axis depth for the front wall. With the camera's calibration and the identified vanishing point, we can accurately map the 3D coordinates of the front wall.

## Spatial Reconstruction

Once the front wall's coordinates are established, we can extend the model to other walls: left, right, ceiling, and floor. We can treat the image plane as the rear wall of the 'box', enabling a complete 3D spatial model of the scene.

## Texture Mapping through Homography

The final step involves applying textures to the 3D model. This is achieved through homography. We can divide the image into five sections (as shown in the following figure) based on the vanishing point and front wall. For each wall (e.g., the left wall), a homography matrix is computed using the wall's four corners and corresponding image sections. This allows us to warp the image texture accurately onto each wall surface.

![room](data/room.jpg)

By following these steps, the algorithm transforms a flat image into a navigable 3D space, offering an immersive 'tour' into the picture.