# cadhelp_app.py
# CADhelp - Advanced 2D Orthographic Projection Generator (Streamlit)
# - No cv2 dependency
# - Uses Pillow + numpy
# Save as cadhelp_app.py
# pip install streamlit pillow numpy

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import math, io, re, json, base64
import numpy as np
from typing import List, Tuple

st.set_page_config(page_title="CADhelp", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------
# Constants & Styling
# ---------------------------
CANVAS_W = 1400
CANVAS_H = 900
MARGIN = 40
XY_Y = CANVAS_H // 2
BG = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)

THICK_HP = 5   # front view outlines
THICK_VP = 4   # top view outlines
THIN = 1       # construction/projector

FONT_PATH = None  # use default if not present

def load_font(size=14):
    try:
        if FONT_PATH:
            return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        pass
    return ImageFont.load_default()

# Helper boxes for drawing areas
FRONT_BOX = (MARGIN+10, MARGIN+10, CANVAS_W//2 - 10, XY_Y - 30)
TOP_BOX   = (MARGIN+10, XY_Y + 30, CANVAS_W//2 - 10, CANVAS_H - 30)
INFO_BOX  = (CANVAS_W//2 + 10, MARGIN+10, CANVAS_W - MARGIN - 10, CANVAS_H - 30)

# ---------------------------
# Low-level drawing helpers
# ---------------------------
def new_canvas():
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)
    return img, draw

def draw_xy(draw):
    draw.line([(MARGIN, XY_Y), (CANVAS_W - MARGIN, XY_Y)], fill=WHITE, width=1)
    f = load_font(18)
    draw.text((MARGIN+10, XY_Y - 28), "Front view (above XY) — First Angle", font=f, fill=WHITE)
    draw.text((MARGIN+10, XY_Y + 12), "Top view (below XY)", font=f, fill=WHITE)

def draw_info_box(draw, lines: List[str]):
    left, top, right, bottom = INFO_BOX
    draw.rectangle([left, top, right, bottom], outline=WHITE, width=1)
    f = load_font(14)
    y = top + 12
    for line in lines:
        draw.text((left + 10, y), line, font=f, fill=WHITE)
        y += 18

def mm_to_px_scale(max_mm, box_width_px):
    # choose scale so max_mm maps to box_width_px * 0.8
    if max_mm <= 0: max_mm = 100.0
    return (box_width_px * 0.8) / max_mm

def in_box_transform(x_mm, y_mm, box, scale):
    # center box coordinate system: x to right, y up
    left, top, right, bottom = box
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    px = cx + x_mm * scale
    py = cy - y_mm * scale
    return (px, py)

# ---------------------------
# Geometry utilities
# ---------------------------
def rotate_point_2d(x, y, deg):
    r = math.radians(deg)
    xr = x*math.cos(r) - y*math.sin(r)
    yr = x*math.sin(r) + y*math.cos(r)
    return xr, yr

def polygon_regular(n_sides, circumradius):
    pts = []
    for i in range(n_sides):
        theta = 2*math.pi*i / n_sides
        x = circumradius * math.cos(theta)
        y = circumradius * math.sin(theta)
        pts.append((x,y))
    return pts

def circle_points(radius, n=64):
    pts=[]
    for i in range(n):
        theta=2*math.pi*i/n
        pts.append((radius*math.cos(theta), radius*math.sin(theta)))
    return pts

# ---------------------------
# Parsers: Interpret user question text
# ---------------------------
def extract_numbers(text):
    nums = re.findall(r'([-+]?\d*\.?\d+)\s*(?:mm|MM|cm|CM|m|M)?', text)
    return [float(x) for x in nums]

def extract_angles(text):
    angs = re.findall(r'([-+]?\d*\.?\d+)\s*(?:deg|°|degree|degrees)', text, flags=re.I)
    return [float(x) for x in angs]

def parse_question(text):
    """
    Return a dict:
    {
      'type': 'point'/'line'/'lamina'/'prism'/'pyramid'/'cylinder'/'cone'/'cube'/'cuboid'/...
      'shape': for lamina: 'triangle','square','rectangle','pentagon','hexagon','circle'
      'params': dict of numeric params
    }
    Heuristics-based parser tailored to typical textbook inputs.
    """
    t = text.lower()
    nums = extract_numbers(text)
    angs = extract_angles(text)
    res = {'type': None, 'shape': None, 'params': {}, 'raw': text}

    # Points
    if 'point' in t or 'infront' in t or 'above' in t or 'below' in t:
        res['type'] = 'point'
        # heuristics: first number => infront (y), second => above (z)
        if len(nums) >= 1:
            res['params']['infront'] = nums[0]
        if len(nums) >= 2:
            res['params']['above'] = nums[1]
        return res

    # Line
    if 'line' in t or re.search(r'\btrue length\b', t) or 'inclined' in t:
        res['type'] = 'line'
        if len(nums) >= 1:
            res['params']['true_length'] = nums[0]
        # angles: assign if present (first angle to HP, second to VP)
        if len(angs) >= 1:
            res['params']['angle_hp'] = angs[0]
        if len(angs) >= 2:
            res['params']['angle_vp'] = angs[1]
        return res

    # Lamina / Plane
    if any(k in t for k in ['lamina','triangle','rectangular','square','circle','pentagon','hexagon']):
        res['type'] = 'lamina'
        if 'triangle' in t or 'triangular' in t:
            res['shape'] = 'triangle'
        elif 'square' in t:
            res['shape'] = 'square'
        elif 'rectangular' in t or 'rectangle' in t:
            res['shape'] = 'rectangle'
        elif 'pentagon' in t:
            res['shape'] = 'pentagon'
        elif 'hexagon' in t:
            res['shape'] = 'hexagon'
        elif 'circle' in t:
            res['shape'] = 'circle'
        # size heuristics
        if nums:
            # triangle/square: first numeric is side/radius accordingly
            res['params']['nums'] = nums
        if angs:
            res['params']['surface_angle'] = angs[0]
        return res

    # Solids
    if any(k in t for k in ['prism','pyramid','cone','cylinder','cube','cuboid','rectangular prism','solid','develop','development','lateral']):
        # detect specific solids
        if 'prism' in t:
            res['type'] = 'prism'
        elif 'pyramid' in t:
            res['type'] = 'pyramid'
        elif 'cylinder' in t:
            res['type'] = 'cylinder'
        elif 'cone' in t:
            res['type'] = 'cone'
        elif 'cube' in t:
            res['type'] = 'cube'
        elif 'cuboid' in t or 'rectangular prism' in t:
            res['type'] = 'cuboid'
        # parse numeric list
        if nums:
            res['params']['nums'] = nums
        # if mention 'develop' treat as development
        if 'develop' in t or 'development' in t or 'lateral' in t:
            res['type'] = 'develop'
        return res

    # fallback: maybe line
    res['type'] = 'line'
    if nums:
        res['params']['true_length'] = nums[0]
    return res

# ---------------------------
# Core drawing routines
# ---------------------------

def draw_base_layout(draw):
    # front box
    draw.rectangle(FRONT_BOX, outline=WHITE, width=1)
    draw.rectangle(TOP_BOX, outline=WHITE, width=1)
    # XY line
    draw.line([(MARGIN, XY_Y), (CANVAS_W - MARGIN, XY_Y)], fill=WHITE, width=1)
    f = load_font(16)
    draw.text((FRONT_BOX[0]+6, FRONT_BOX[1]-22), "Front View (above XY) — First Angle", font=f, fill=WHITE)
    draw.text((TOP_BOX[0]+6, TOP_BOX[3]+6), "Top View (below XY)", font=f, fill=WHITE)

def annotate_lengths_and_angles(params):
    lines=[]
    # common keys: true_length, angle_hp, angle_vp, apparent_top, apparent_front
    if 'true_length' in params:
        lines.append(f"True length = {params['true_length']:.2f} mm")
    if 'apparent_top' in params:
        lines.append(f"Top apparent = {params['apparent_top']:.2f} mm")
    if 'apparent_front' in params:
        lines.append(f"Front apparent = {params['apparent_front']:.2f} mm")
    if 'angle_hp' in params:
        lines.append(f"Inclination to HP (α) = {params['angle_hp']:.2f}°")
    if 'angle_vp' in params:
        lines.append(f"Inclination to VP (β) = {params['angle_vp']:.2f}°")
    return lines

# --- Draw point ---
def draw_point(img_draw, infront_mm=30, above_mm=30, label='A'):
    # scale based on a default range
    box_w = FRONT_BOX[2] - FRONT_BOX[0]
    box_h = FRONT_BOX[3] - FRONT_BOX[1]
    max_mm = max(infront_mm, above_mm, 100.0)
    scale = mm_to_px_scale(max_mm, min(box_w, box_h))
    # front point (x=0,z=above)
    front_pt = in_box_transform(0, above_mm, FRONT_BOX, scale)
    top_pt = in_box_transform(0, infront_mm, TOP_BOX, scale)
    # draw
    img_draw.ellipse([front_pt[0]-7, front_pt[1]-7, front_pt[0]+7, front_pt[1]+7], outline=WHITE, width=THICK_HP)
    img_draw.ellipse([top_pt[0]-5, top_pt[1]-5, top_pt[0]+5, top_pt[1]+5], outline=WHITE, width=THICK_VP)
    img_draw.line([front_pt, top_pt], fill=WHITE, width=THIN)
    f = load_font(14)
    img_draw.text((front_pt[0]+8, front_pt[1]-12), f"{label}''", font=f, fill=WHITE)
    img_draw.text((top_pt[0]+8, top_pt[1]-6), f"{label}'", font=f, fill=WHITE)
    info = [f"Point {label}: {infront_mm} mm infront of VP, {above_mm} mm above HP"]
    return info

# --- Draw line using true length and inclinations ---
def draw_line(img_draw, L=80.0, angle_hp=30.0, angle_vp=45.0):
    # compute apparent lengths
    apparent_top = L * abs(math.cos(math.radians(angle_hp)))
    apparent_front = L * abs(math.cos(math.radians(angle_vp)))
    # choose scale to fit largest
    max_mm = max(L, apparent_top, apparent_front, 120.0)
    box_w = FRONT_BOX[2] - FRONT_BOX[0]
    scale = mm_to_px_scale(max_mm, min(box_w, (FRONT_BOX[3]-FRONT_BOX[1])))
    # pick a start point A in 3D: Ax,Ay,Az
    Ax, Ay, Az = -L*0.15, 20.0, 20.0
    # compute plan length and vertical component
    dz = L * math.sin(math.radians(angle_hp))   # vertical component (to HP)
    plan_len = L * math.cos(math.radians(angle_hp)) # length in HP (in plan)
    # choose plan orientation angle for visual (so top view not colinear)
    plan_angle_deg = 25.0
    dx = plan_len * math.cos(math.radians(plan_angle_deg))
    dy = plan_len * math.sin(math.radians(plan_angle_deg))
    Bx = Ax + dx
    By = Ay + dy
    Bz = Az + dz
    # compute front & top points
    A_front = in_box_transform(Ax, Az, FRONT_BOX, scale)
    B_front = in_box_transform(Bx, Bz, FRONT_BOX, scale)
    A_top = in_box_transform(Ax, Ay, TOP_BOX, scale)
    B_top = in_box_transform(Bx, By, TOP_BOX, scale)
    # draw thick front
    img_draw.line([A_front, B_front], fill=WHITE, width=THICK_HP)
    img_draw.ellipse([A_front[0]-6, A_front[1]-6, A_front[0]+6, A_front[1]+6], outline=WHITE, width=THICK_HP)
    img_draw.ellipse([B_front[0]-6, B_front[1]-6, B_front[0]+6, B_front[1]+6], outline=WHITE, width=THICK_HP)
    # draw top
    img_draw.line([A_top, B_top], fill=WHITE, width=THICK_VP)
    img_draw.ellipse([A_top[0]-5, A_top[1]-5, A_top[0]+5, A_top[1]+5], outline=WHITE, width=THICK_VP)
    img_draw.ellipse([B_top[0]-5, B_top[1]-5, B_top[0]+5, B_top[1]+5], outline=WHITE, width=THICK_VP)
    # projectors
    img_draw.line([A_front, A_top], fill=WHITE, width=THIN)
    img_draw.line([B_front, B_top], fill=WHITE, width=THIN)
    f = load_font(14)
    img_draw.text((A_front[0]-18, A_front[1]-28), "A''", font=f, fill=WHITE)
    img_draw.text((B_front[0]-18, B_front[1]-28), "B''", font=f, fill=WHITE)
    img_draw.text((A_top[0]-18, A_top[1]+6), "A'", font=f, fill=WHITE)
    img_draw.text((B_top[0]-18, B_top[1]+6), "B'", font=f, fill=WHITE)
    info = annotate_lengths_and_angles({
        'true_length': L,
        'apparent_top': apparent_top,
        'apparent_front': apparent_front,
        'angle_hp': angle_hp,
        'angle_vp': angle_vp
    })
    return info

# --- Draw lamina (plane shapes) using first-angle projection ---
def draw_lamina(img_draw, shape='triangle', nums=None, surface_angle=60.0, edge_rot=30.0):
    # nums: list of numbers, interpretation depends on shape
    f = load_font(14)
    # compute a base size
    if nums and len(nums)>0:
        size = nums[0]
    else:
        size = 40.0
    # compute scale
    max_dim = size * 2.5
    scale = mm_to_px_scale(max_dim, min(FRONT_BOX[2]-FRONT_BOX[0], FRONT_BOX[3]-FRONT_BOX[1]))
    # FRONT view: true shape pitched by surface_angle -> foreshortened vertical component
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])//2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])//2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])//2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])//2
    if shape == 'triangle':
        # equilateral triangle side = size
        h_true = math.sqrt(3)/2 * size
        # front height foreshortened
        h_front = h_true * abs(math.cos(math.radians(surface_angle)))
        # points in front
        A = (cx_f, cy_f - h_front/2)
        B = (cx_f - size*scale, cy_f + h_front/2)
        C = (cx_f + size*scale, cy_f + h_front/2)
        img_draw.line([A,B,C,A], fill=WHITE, width=THICK_HP)
        # top view: triangle in plan rotated by edge_rot
        theta = math.radians(edge_rot)
        pts_top = []
        for px,py in [(0,-h_true*2/3),( -size/2, h_true/3),(size/2,h_true/3)]:
            rx = px*math.cos(theta) - py*math.sin(theta)
            ry = px*math.sin(theta) + py*math.cos(theta)
            pts_top.append((cx_t + rx*scale, cy_t + ry*scale))
        img_draw.line([pts_top[0],pts_top[1],pts_top[2],pts_top[0]], fill=WHITE, width=THICK_VP)
        # projectors connect A->pts_top[0], B->pts_top[1], C->pts_top[2]
        img_draw.line([A, pts_top[0]], fill=WHITE, width=THIN)
        img_draw.line([B, pts_top[1]], fill=WHITE, width=THIN)
        img_draw.line([C, pts_top[2]], fill=WHITE, width=THIN)
        info=[f"Triangular lamina side={size} mm, surface to HP={surface_angle}°"]
        return info

    elif shape == 'square' or shape == 'rectangle':
        if shape == 'square':
            w = size
            h = size
        else:
            if nums and len(nums) >= 2:
                w = nums[0]
                h = nums[1]
            else:
                w = size; h = size*1.6
        # front rectangle (foreshortened height)
        hw_px = (w/2)*scale
        hh_px = (h/2)*scale * abs(math.cos(math.radians(surface_angle)))
        left = cx_f - hw_px
        top = cy_f - hh_px
        right = cx_f + hw_px
        bottom = cy_f + hh_px
        img_draw.rectangle([left,top,right,bottom], outline=WHITE, width=THICK_HP)
        # top view rotated
        theta = math.radians(edge_rot)
        corners = [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
        pts_top=[]
        for px,py in corners:
            rx = px*math.cos(theta) - py*math.sin(theta)
            ry = px*math.sin(theta) + py*math.cos(theta)
            pts_top.append((cx_t + rx*scale, cy_t + ry*scale))
        img_draw.line([pts_top[0],pts_top[1],pts_top[2],pts_top[3],pts_top[0]], fill=WHITE, width=THICK_VP)
        # projectors corners
        for i,pt in enumerate([(left,top),(right,top),(right,bottom),(left,bottom)]):
            img_draw.line([pt, pts_top[i]], fill=WHITE, width=THIN)
        info=[f"{shape.title()} lamina {w} x {h} mm surface to HP={surface_angle}°"]
        return info

    elif shape == 'circle':
        # size[0] is radius or diameter? assume diameter if >0
        if nums and len(nums)>=1:
            diam = nums[0]
        else:
            diam = size
        r = (diam/2)
        # front view: foreshortened circle becomes ellipse
        rx = r*scale
        ry = r*scale * abs(math.cos(math.radians(surface_angle)))
        left = cx_f - rx; right = cx_f + rx
        top = cy_f - ry; bottom = cy_f + ry
        img_draw.ellipse([left,top,right,bottom], outline=WHITE, width=THICK_HP)
        # top view: circle true size
        rx_t = r*scale; ry_t = r*scale
        img_draw.ellipse([cx_t - rx_t, cy_t - ry_t, cx_t + rx_t, cy_t + ry_t], outline=WHITE, width=THICK_VP)
        img_draw.line([(cx_f - rx, cy_f),(cx_t - rx_t, cy_t)], fill=WHITE, width=THIN)
        info=[f"Circle diameter = {diam} mm, surface to HP = {surface_angle}°"]
        return info

    elif shape == 'pentagon' or shape == 'hexagon':
        n = 5 if shape == 'pentagon' else 6
        # circumradius approximates from given 'size' as side length
        side = size
        # approximate circumradius R for regular polygon given side s and n:
        R = side / (2 * math.sin(math.pi / n))
        # front: foreshortened in vertical direction
        pts_front = []
        pts_top = []
        for i,(px,py) in enumerate(polygon_regular(n, R)):
            pts_front.append((cx_f + px*scale, cy_f + py*scale * math.cos(math.radians(surface_angle))))
            # top rotated by edge_rot
            rx, ry = rotate_point_2d(px, py, edge_rot)
            pts_top.append((cx_t + rx*scale, cy_t + ry*scale))
        img_draw.line(pts_front + [pts_front[0]], fill=WHITE, width=THICK_HP)
        img_draw.line(pts_top + [pts_top[0]], fill=WHITE, width=THICK_VP)
        for i in range(n):
            img_draw.line([pts_front[i], pts_top[i]], fill=WHITE, width=THIN)
        info=[f"{n}-gon lamina side~{side} mm, surface to HP={surface_angle}°"]
        return info

    else:
        return [f"Unknown lamina shape: {shape}"]

# --- Solids: prism/pyramid/cylinder/cone/cube/cuboid ---
def draw_prism(img_draw, base_sides=4, base_dims=None, height=80.0):
    # base_sides: 3,4,5,6; base_dims: for rectangular base [w,d] or side length for regular polygon
    f = load_font(14)
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])//2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])//2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])//2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])//2
    if base_dims is None:
        side = 50.0
        depth = 40.0
    else:
        if isinstance(base_dims, (list,tuple)) and len(base_dims)>=2:
            side, depth = base_dims[0], base_dims[1]
        else:
            side = base_dims if base_dims else 50.0
            depth = side * 0.8
    max_mm = max(side, depth, height, 120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    # Front view: rectangle width=side, height=height (we draw as front face)
    left = cx_f - (side/2)*scale
    right = cx_f + (side/2)*scale
    top = cy_f - (height/2)*scale
    bottom = cy_f + (height/2)*scale
    img_draw.rectangle([left,top,right,bottom], outline=WHITE, width=THICK_HP)
    # Top view: rectangle side x depth
    tlx = cx_t - (side/2)*scale
    tly = cy_t - (depth/2)*scale
    brx = cx_t + (side/2)*scale
    bry = cy_t + (depth/2)*scale
    img_draw.rectangle([tlx,tly,brx,bry], outline=WHITE, width=THICK_VP)
    # projectors (4 corners)
    corners_front = [(left,top),(right,top),(right,bottom),(left,bottom)]
    corners_top = [(tlx,tly),(brx,tly),(brx,bry),(tlx,bry)]
    for a,b in zip(corners_front, corners_top):
        img_draw.line([a,b], fill=WHITE, width=THIN)
    # lateral development: strip of side rectangles on right
    dev_x = INFO_BOX[0] + 14
    dev_y = FRONT_BOX[1]
    face_w = (side)*scale
    face_h = (height)*scale
    for i in range(4):
        x0 = dev_x + i*(face_w + 4)
        img_draw.rectangle([x0, dev_y, x0+face_w, dev_y+face_h], outline=WHITE, width=THICK_HP)
    info=[f"Prism base {side} x {depth} mm height {height} mm; lateral development (strip) shown"]
    return info

def draw_pyramid(img_draw, base_sides=4, base_dims=None, height=80.0):
    # Simplified pyramid front & top approximations
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])//2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])//2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])//2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])//2
    if base_dims is None:
        side = 50.0
    else:
        side = base_dims if not isinstance(base_dims, (list,tuple)) else base_dims[0]
    max_mm = max(side, height, 120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    # front: triangle (apex above base mid)
    left = cx_f - (side/2)*scale
    right = cx_f + (side/2)*scale
    base_y = cy_f + (height/2)*scale
    apex = (cx_f, cy_f - (height/2)*scale)
    A = (left, base_y)
    B = (right, base_y)
    img_draw.line([A,B,apex,A], fill=WHITE, width=THICK_HP)
    # top: base polygon (regular n sides)
    pts_top=[]
    R = side / (2*math.sin(math.pi/base_sides))
    for px,py in polygon_regular(base_sides, R):
        pts_top.append((cx_t + px*scale, cy_t + py*scale))
    img_draw.line(pts_top + [pts_top[0]], fill=WHITE, width=THICK_VP)
    # projectors from base vertices (approx)
    # map first two for projection
    for i in range(min(len(pts_top),2)):
        # pick corresponding front base vertices A,B
        img_draw.line([(left + i*(right-left), base_y), pts_top[i]], fill=WHITE, width=THIN)
    info=[f"Pyramid base ~{side} mm, sides={base_sides}, height {height} mm"]
    return info

def draw_cylinder(img_draw, diameter=40.0, height=80.0):
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])//2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])//2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])//2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])//2
    max_mm = max(diameter, height, 120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    r = (diameter/2)*scale
    # front: rectangle with ellipses top/bottom? For simplicity show rectangle (height) and ellipse top as hidden line
    left = cx_f - r
    right = cx_f + r
    top = cy_f - (height/2)*scale
    bottom = cy_f + (height/2)*scale
    # ellipse top visible? In first-angle, top view below XY shows circle
    img_draw.rectangle([left, top, right, bottom], outline=WHITE, width=THICK_HP)
    # top view: circle
    img_draw.ellipse([cx_t-r, cy_t-r, cx_t+r, cy_t+r], outline=WHITE, width=THICK_VP)
    # projectors
    img_draw.line([(left,top),(cx_t-r,cy_t-r)], fill=WHITE, width=THIN)
    img_draw.line([(right,bottom),(cx_t+r,cy_t+r)], fill=WHITE, width=THIN)
    info=[f"Cylinder diameter {diameter} mm, height {height} mm"]
    return info

def draw_cone(img_draw, diameter=40.0, height=80.0):
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])//2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])//2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])//2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])//2
    max_mm = max(diameter, height, 120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    r = (diameter/2)*scale
    left = cx_f - r
    right = cx_f + r
    apex = (cx_f, cy_f - (height/2)*scale)
    base_left = (left, cy_f + (height/2)*scale)
    base_right = (right, cy_f + (height/2)*scale)
    img_draw.line([base_left, base_right], fill=WHITE, width=THICK_HP)
    img_draw.line([base_left, apex, base_right], fill=WHITE, width=THICK_HP)
    # top view: circle
    img_draw.ellipse([cx_t-r, cy_t-r, cx_t+r, cy_t+r], outline=WHITE, width=THICK_VP)
    # projectors
    img_draw.line([base_left, (cx_t-r, cy_t)], fill=WHITE, width=THIN)
    img_draw.line([base_right, (cx_t+r, cy_t)], fill=WHITE, width=THIN)
    info=[f"Cone base diameter {diameter} mm, height {height} mm"]
    return info

def draw_cube_cuboid(img_draw, w=50.0, d=40.0, h=50.0):
    cx_f = (FRONT_BOX[0]+FRONT_BOX[2])//2
    cy_f = (FRONT_BOX[1]+FRONT_BOX[3])//2
    cx_t = (TOP_BOX[0]+TOP_BOX[2])//2
    cy_t = (TOP_BOX[1]+TOP_BOX[3])//2
    max_mm = max(w,d,h,120.0)
    scale = mm_to_px_scale(max_mm, FRONT_BOX[2]-FRONT_BOX[0])
    # front rectangle w x h
    left = cx_f - (w/2)*scale
    right = cx_f + (w/2)*scale
    top = cy_f - (h/2)*scale
    bottom = cy_f + (h/2)*scale
    img_draw.rectangle([left,top,right,bottom], outline=WHITE, width=THICK_HP)
    # top view: w x d
    tlx = cx_t - (w/2)*scale
    tly = cy_t - (d/2)*scale
    brx = cx_t + (w/2)*scale
    bry = cy_t + (d/2)*scale
    img_draw.rectangle([tlx,tly,brx,bry], outline=WHITE, width=THICK_VP)
    # projectors
    img_draw.line([(left,top),(tlx,tly)], fill=WHITE, width=THIN)
    img_draw.line([(right,top),(brx,tly)], fill=WHITE, width=THIN)
    img_draw.line([(left,bottom),(tlx,bry)], fill=WHITE, width=THIN)
    img_draw.line([(right,bottom),(brx,bry)], fill=WHITE, width=THIN)
    info=[f"Cuboid {w} x {d} x {h} mm (top: w x d ; front: w x h)"]
    return info

# ---------------------------
# Utility: image <-> bytes
# ---------------------------
def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def download_button_bytes(data_bytes: bytes, filename: str, label: str):
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------------------------
# Voice helper (browser) HTML widget
# ---------------------------
VOICE_HTML = """
<div style="background:#111;padding:10px;border:1px solid #444;color:white;">
  <p style="margin:0 0 6px 0"><strong>Voice input helper</strong> — click Record, speak your question clearly in English. After stop, copy the transcript into the question box.</p>
  <button id="recBtn">🎤 Record</button>
  <button id="stopBtn" disabled>⏹ Stop</button>
  <div id="status" style="margin-top:6px;color:#ddd">Idle</div>
  <div id="transcript" style="margin-top:10px;background:#000;padding:8px;border:1px solid #333;min-height:60px;color:#fff;"></div>
</div>
<script>
const recBtn = document.getElementById('recBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
const transcriptDiv = document.getElementById('transcript');
let recognition;
if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
  status.innerText = 'Speech Recognition not supported in this browser. Use Chrome on desktop or Android.';
} else {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';
  recognition.onstart = () => { status.innerText = 'Listening...'; recBtn.disabled=true; stopBtn.disabled=false; };
  recognition.onerror = (e) => { status.innerText = 'Error: ' + e.error; recBtn.disabled=false; stopBtn.disabled=true; };
  recognition.onend = () => { status.innerText = 'Stopped'; recBtn.disabled=false; stopBtn.disabled=true; };
  recognition.onresult = (event) => {
    let text = '';
    for (let i=0;i<event.results.length;i++){
      text += event.results[i][0].transcript;
    }
    transcriptDiv.innerText = text;
  }
}
recBtn.onclick = () => { if (recognition) recognition.start(); };
stopBtn.onclick = () => { if (recognition) recognition.stop(); };
</script>
"""

# ---------------------------
# Streamlit UI layout
# ---------------------------
st.title("CADhelp — Orthographic Projection Generator")
st.markdown("Type or paste a textbook-style question (points, lines, lamina, solids). You can also upload a photo (display only) and use voice helper to record text (copy the transcript into the question box). Output is a CAD-like 2D orthographic (first-angle for planes, auxiliary-plane style for solids).")

# Sidebar controls
st.sidebar.header("Input & Options")
q_text = st.sidebar.text_area("Type question here (example):\nLine AB 80 mm long inclined 30 deg to HP and 45 deg to VP\nPoint A 30 mm infront of VP and 20 mm above HP\nTriangular lamina side 40 mm surface 60 deg to HP", height=200)
uploaded_file = st.sidebar.file_uploader("Upload photo of question (optional)", type=['png','jpg','jpeg'])
st.sidebar.markdown("Voice input helper (use Chrome):")
#st.sidebar.components.v1.html(VOICE_HTML, height=170)

force_type = st.sidebar.selectbox("Force question type (optional)", ["Auto","Point","Line","Lamina","Prism","Pyramid","Cylinder","Cone","Cube/Cuboid","Develop"])
st.sidebar.markdown("---")
st.sidebar.markdown("Override numeric params (comma separated): e.g. true_length=80, angle_hp=30, angle_vp=45, infront=30, above=20, base_w=50, base_d=40, height=80")
override_text = st.sidebar.text_input("Overrides", value="")

# parse override into dict
overrides = {}
if override_text.strip():
    try:
        for piece in override_text.split(','):
            if '=' in piece:
                k,v = piece.split('=',1)
                k=k.strip(); v=v.strip()
                try:
                    overrides[k]=float(v)
                except:
                    overrides[k]=v
    except Exception:
        st.sidebar.error("Could not parse overrides. Use key=value, ...")

st.markdown("### Uploaded image (if any)")
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded question image (for reference). OCR not auto-run.", use_column_width=True)

# decide q
if q_text.strip():
    parsed = parse_question(q_text)
else:
    parsed = {'type': None, 'params': {}, 'raw': ''}

# apply forced type
if force_type != "Auto":
    parsed['type'] = force_type.lower() if force_type != "Cube/Cuboid" else "cube/cuboid"

# merge overrides
params = parsed.get('params', {})
params.update(overrides)

# Main controls area
st.markdown("### Parsed question")
st.json(parsed)

# Buttons and generation
st.markdown("### Generate Projection")
col_a, col_b = st.columns([1,1])
with col_a:
    label = st.text_input("Label (for points/lines)", value="A")
with col_b:
    filename = st.text_input("Output filename", value="cad_projection.png")

generate_btn = st.button("Generate Projection Image")

if generate_btn:
    # create canvas and draw base
    img, draw = new_canvas()
    draw_base_layout(draw)
    info_lines = []

    qtype = parsed.get('type')
    try:
        if qtype == 'point' or (force_type=="Point"):
            infront = params.get('infront', 30.0)
            above = params.get('above', 30.0)
            info_lines = draw_point(draw, infront_mm:=float(infront) if infront is not None else 30.0,
                                    above_mm:=float(above) if above is not None else 30.0, label=label)
        elif qtype == 'line' or (force_type=="Line"):
            L = params.get('true_length', overrides.get('true_length', 80.0))
            a = params.get('angle_hp', overrides.get('angle_hp', 30.0))
            b = params.get('angle_vp', overrides.get('angle_vp', 45.0))
            L = float(L) if L is not None else 80.0
            a = float(a) if a is not None else 30.0
            b = float(b) if b is not None else 45.0
            info_lines = draw_line(draw, L=L, angle_hp=a, angle_vp=b)
        elif qtype == 'lamina' or (force_type=="Lamina"):
            shape = parsed.get('shape', 'triangle') or 'triangle'
            nums = params.get('nums', [])
            surface_angle = params.get('surface_angle', overrides.get('surface_angle', 60.0))
            edge_rot = params.get('edge_rot', overrides.get('edge_rot', 30.0))
            info_lines = draw_lamina(draw, shape=shape, nums=nums, surface_angle=float(surface_angle), edge_rot=float(edge_rot))
        elif qtype == 'prism' or (force_type=="Prism"):
            nums = params.get('nums', [])
            if nums and len(nums)>=3:
                base_w, base_d, height = float(nums[0]), float(nums[1]), float(nums[2])
            else:
                base_w = overrides.get('base_w', params.get('base_w', 50.0))
                base_d = overrides.get('base_d', params.get('base_d', 40.0))
                height = overrides.get('height', params.get('height', 80.0))
            info_lines = draw_prism(draw, base_sides=4, base_dims=(float(base_w), float(base_d)), height=float(height))
        elif qtype == 'pyramid' or (force_type=="Pyramid"):
            nums = params.get('nums', [])
            base_sides = int(overrides.get('base_sides', params.get('base_sides', 4)))
            base_size = nums[0] if nums else overrides.get('base_size', 50.0)
            height = overrides.get('height', params.get('height', 80.0))
            info_lines = draw_pyramid(draw, base_sides=base_sides, base_dims=base_size, height=float(height))
        elif qtype == 'cylinder' or (force_type=="Cylinder"):
            nums = params.get('nums', [])
            diam = nums[0] if nums else overrides.get('diameter', 40.0)
            h = overrides.get('height', params.get('height', 80.0))
            info_lines = draw_cylinder(draw, diameter=float(diam), height=float(h))
        elif qtype == 'cone' or (force_type=="Cone"):
            nums = params.get('nums', [])
            diam = nums[0] if nums else overrides.get('diameter', 40.0)
            h = overrides.get('height', params.get('height', 80.0))
            info_lines = draw_cone(draw, diameter=float(diam), height=float(h))
        elif qtype == 'cube' or qtype == 'cuboid' or (force_type=="Cube/Cuboid"):
            nums = params.get('nums', [])
            if nums and len(nums)>=3:
                w,d,h = float(nums[0]), float(nums[1]), float(nums[2])
            else:
                w = overrides.get('w', params.get('w', 50.0))
                d = overrides.get('d', params.get('d', 40.0))
                h = overrides.get('h', params.get('h', 50.0))
            info_lines = draw_cube_cuboid(draw, w=float(w), d=float(d), h=float(h))
        elif qtype == 'develop' or (force_type=="Develop"):
            # ask for prism params
            nums = params.get('nums', [])
            base_w = nums[0] if nums else overrides.get('base_w', 50.0)
            base_d = nums[1] if len(nums)>1 else overrides.get('base_d', 40.0)
            height = nums[2] if len(nums)>2 else overrides.get('height', 80.0)
            info_lines = draw_prism(draw, base_sides=4, base_dims=(float(base_w), float(base_d)), height=float(height))
            # note: draw_prism includes lateral development
        else:
            # fallback: attempt line
            L = params.get('true_length', overrides.get('true_length', 80.0))
            a = params.get('angle_hp', overrides.get('angle_hp', 30.0))
            b = params.get('angle_vp', overrides.get('angle_vp', 45.0))
            info_lines = draw_line(draw, L=float(L), angle_hp=float(a), angle_vp=float(b))
    except Exception as e:
        st.error(f"Error while generating drawing: {e}")
        info_lines = [f"Error: {e}"]

    # draw info box
    draw_info_box(draw, info_lines)

    # show image & download
    st.image(img, use_column_width=True)
    bytes_img = pil_to_bytes(img)
    download_button_bytes(bytes_img, filename, "Download PNG")

# Footer help
st.markdown("---")
st.markdown("Notes:\n- If you see errors about cv2: this app does not use cv2. `cv2` is OpenCV (use `pip install opencv-python` if you need it). I intentionally avoided cv2 so deployment on Streamlit Cloud is simpler.\n- Voice helper requires Chrome browser (desktop or Android). Record and copy-paste transcript into the question box for best results.\n- For fully automatic OCR of uploaded photos, I can add `pytesseract` support, but that requires installing the Tesseract engine on the server (I can provide instructions or code).")


