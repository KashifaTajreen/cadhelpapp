import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

# ---------- GEOMETRY ENGINE ----------
def rotate_point(p, angle_deg):
    angle = math.radians(angle_deg)
    x, y, z = p
    xr = x*math.cos(angle) - y*math.sin(angle)
    yr = x*math.sin(angle) + y*math.cos(angle)
    return (xr, yr, z)

def project_first_angle(point):
    x, y, z = point
    front = (x, z)
    top = (x, -y)
    return front, top

def draw_line(draw, p1, p2, color, width):
    draw.line([p1, p2], fill=color, width=width)

# ---------- SHAPE MODELS ----------
def model_line(length=60, angle=30):
    p1 = (0,0,0)
    p2 = (length*math.cos(math.radians(angle)),
          length*math.sin(math.radians(angle)), 0)
    return [p1, p2]

def model_plane(sides=4, radius=40):
    pts = []
    for i in range(sides):
        ang = i * (360/sides)
        x = radius * math.cos(math.radians(ang))
        y = radius * math.sin(math.radians(ang))
        pts.append((x,y,0))
    return pts

def model_cube(edge=40):
    pts = []
    for z in [0, edge]:
        for y in [0, edge]:
            for x in [0, edge]:
                pts.append((x,y,z))
    return pts

# ---------- RENDER ENGINE ----------
def render_projection(obj_points):

    canvas = Image.new("RGB", (900, 600), "black")
    draw = ImageDraw.Draw(canvas)

    front_pts = []
    top_pts = []

    for p in obj_points:
        f, t = project_first_angle(p)
        front_pts.append((f[0]+150, 300 - f[1]))
        top_pts.append((t[0]+450, 300 - t[1]))

    # Draw front view
    for i in range(len(front_pts)-1):
        draw_line(draw, front_pts[i], front_pts[i+1], "white", 3)

    # Draw top view
    for i in range(len(top_pts)-1):
        draw_line(draw, top_pts[i], top_pts[i+1], "white", 3)

    draw.text((140, 350), "Front View", fill="white")
    draw.text((440, 350), "Top View", fill="white")

    return canvas

# ---------- STREAMLIT UI ----------
st.title("📐 CADhelp – Engineering Graphics Projector")
st.write("First-Angle 2D Orthographic Projections")

input_type = st.selectbox("Choose Input Type:", 
                          ["Type Question", "Upload Image", "Voice"])

if input_type == "Type Question":
    q = st.text_area("Enter question here:")

shape = st.selectbox("Choose object:",
                     ["Line", "Triangle Plane", "Square Plane", "Pentagon",
                      "Hexagon", "Circle Plane",
                      "Cube", "Cuboid", "Prism", "Pyramid", "Cylinder", "Cone"])

if st.button("Generate Projection"):

    if shape == "Line":
        pts = model_line()
    elif shape == "Triangle Plane":
        pts = model_plane(3)
    elif shape == "Square Plane":
        pts = model_plane(4)
    elif shape == "Pentagon":
        pts = model_plane(5)
    elif shape == "Hexagon":
        pts = model_plane(6)
    elif shape == "Cube":
        pts = model_cube()
    else:
        st.error("This shape is not fully implemented yet.")
        st.stop()

    img = render_projection(pts)
    st.image(img)
