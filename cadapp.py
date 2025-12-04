# cadhelp_app.py
# CADhelp — Orthographic projection generator (First-angle for planes, Auxiliary-plane for solids)
# Save as cadhelp_app.py
# Requirements:
#   pip install streamlit pillow numpy shapely opencv-python-headless SpeechRecognition

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math, io, re, json
from shapely.geometry import Point, Polygon
import cv2
import speech_recognition as sr

# ---------------------------
#  Config / Constants
# ---------------------------
st.set_page_config(page_title="CADhelp", layout="wide")

CANVAS_W, CANVAS_H = 1400, 900
MARGIN = 36
XY_Y = CANVAS_H // 2
BG = (0, 0, 0)
WHITE = (255, 255, 255)

LINE_HP = 4   # thick outlines (front view)
LINE_VP = 3   # medium (top view)
LINE_CONST = 1  # thin projector / construction

FONT = None
def load_font(sz=14):
    global FONT
    if FONT is None:
        try:
            FONT = ImageFont.truetype("DejaVuSans.ttf", sz)
        except:
            FONT = ImageFont.load_default()
    return FONT

# Boxes positions (front view area left top, top view area left bottom), right side for info
FRONT_BOX = (MARGIN, MARGIN, CANVAS_W//2 - MARGIN, XY_Y - 20)
TOP_BOX = (MARGIN, XY_Y + 20, CANVAS_W//2 - MARGIN, CANVAS_H - MARGIN)
INFO_BOX = (CANVAS_W//2 + 10, MARGIN, CANVAS_W - MARGIN, CANVAS_H - MARGIN)

# ---------------------------
# Helper drawing functions
# ---------------------------
def new_canvas():
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)
    return img, draw

def draw_xy_line(draw):
    draw.line([(MARGIN, XY_Y), (CANVAS_W - MARGIN, XY_Y)], fill=WHITE, width=1)
    f = load_font(18)
    draw.text((MARGIN+6, XY_Y-30), "Front view (above XY) — First angle", font=f, fill=WHITE)
    draw.text((MARGIN+6, XY_Y+8), "Top view (below XY)", font=f, fill=WHITE)

def draw_info_panel(draw, text):
    left,top,right,bottom = INFO_BOX
    draw.rectangle([left, top, right, bottom], outline=WHITE, width=1)
    f = load_font(14)
    y = top + 8
    for line in text.splitlines():
        draw.text((left+8, y), line, font=f, fill=WHITE)
        y += 18

def fit_scale_for_box(max_mm, box):
    # returns mm->px scale to fit max_mm into box width or height (approx)
    left,top,right,bottom = box
    w = right - left - 20
    h = bottom - top - 20
    # Use width as primary
    if max_mm == 0:
        return 1.0
    sx = w / max_mm
    sy = h / max_mm
    return min(sx, sy)

def transform_to_box_coords(x_mm, y_mm, box, scale):
    # center mapping: logical mm coordinates with origin at 0 -> box center
    left,top,right,bottom = box
    cx = (left + right)/2
    cy = (top + bottom)/2
    px = cx + x_mm * scale
    py = cy - y_mm * scale
    return (px, py)

# ---------------------------
# Parsing user input (text)
# ---------------------------
def parse_text_question(text):
    """
    Very flexible heuristic parser. Recognizes:
    - Point A 20 mm infront of VP and 30 mm above HP
    - Line AB true length 80 mm inclined 30 deg to HP and 45 deg to VP
    - Triangular lamina 30 mm sides resting on HP surface 60 deg to HP nearest corner 30 mm infront of VP
    - Prism base 50x40 height 80 develop lateral surfaces
    """
    if not text:
        return {}
    t = text.lower()
    nums = [float(x) for x in re.findall(r'([-+]?\d*\.?\d+)\s*(?:mm)?', text)]
    angs = [float(x) for x in re.findall(r'([-+]?\d*\.?\d+)\s*(?:deg|°|degree|degrees)', text)]
    out = {'raw': text, 'type': None, 'params': {}}
    # detect point
    if 'point' in t or re.search(r'\b[a-z]\b.*?(infront|above|behind|below|mm)', t):
        out['type'] = 'point'
        # heuristics: first numeric = infront (y), second = above (z)
        if nums: out['params']['infront'] = nums[0]
        if len(nums) > 1: out['params']['above'] = nums[1]
        # allow "behind" or "infront" detection
        # return early
        return out
    # detect line
    if 'line' in t or ('true length' in t) or re.search(r'\b[a-z]{1,2}\b.*\b[a-z]{1,2}\b.*\bmm\b', t):
        out['type'] = 'line'
        L = None; a=None; b=None
        # look for "true length" or "xx mm long"
        m = re.search(r'(?:true length|length|long)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*mm', text, flags=re.I)
        if not m:
            m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*mm\s*(?:long|length)', text, flags=re.I)
        if m:
            L = float(m.group(1))
        else:
            # fallback to first number
            if nums: L = nums[0]
        # angles detection
        if angs:
            a = angs[0]
            if len(angs) > 1: b = angs[1]
        # also pick up patterns "inclined at 30 deg to HP and 45 deg to VP"
        m2 = re.search(r'(\d+(?:\.\d+)?)\s*(?:deg|°).*hp', text, flags=re.I)
        if m2: a = float(m2.group(1))
        m3 = re.search(r'(\d+(?:\.\d+)?)\s*(?:deg|°).*vp', text, flags=re.I)
        if m3: b = float(m3.group(1))
        out['params'].update({'true_length': L, 'angle_hp': a, 'angle_vp': b})
        return out
    # detect lamina / plane
    if 'lamina' in t or 'triangle' in t or 'rectangular' in t or 'square' in t or 'polygon' in t or 'pentagon' in t or 'hexagon' in t or 'circle' in t:
        out['type'] = 'lamina'
        # shape
        if 'triangle' in t or 'triangular' in t: shape='triangle'
        elif 'hexagon' in t: shape='hexagon'
        elif 'pentagon' in t: shape='pentagon'
        elif 'square' in t: shape='square'
        elif 'rectangular' in t or 'rectangle' in t: shape='rectangle'
        elif 'circle' in t: shape='circle'
        else: shape='polygon'
        out['params']['shape'] = shape
        # sizes
        if nums:
            out['params']['sizes'] = nums
        # surface angle
        if angs:
            out['params']['surface_angle'] = angs[0]
        return out
    # detect solids / development
    if 'prism' in t or 'pyramid' in t or 'cone' in t or 'cylinder' in t or 'cube' in t or 'cuboid' in t or 'develop' in t or 'development' in t or 'lateral' in t:
        # classify
        if 'prism' in t: typ='prism'
        elif 'pyramid' in t: typ='pyramid'
        elif 'cone' in t: typ='cone'
        elif 'cylinder' in t: typ='cylinder'
        elif 'cube' in t: typ='cube'
        elif 'cuboid' in t: typ='cuboid'
        else: typ='solid'
        out['type'] = 'solid'
        out['params']['solid_type'] = typ
        if nums: out['params']['sizes'] = nums
        if 'develop' in t or 'development' in t or 'lateral' in t:
            out['params']['develop'] = True
        return out
    # fallback: nothing
    out['type'] = 'unknown'
    return out

# ---------------------------
# Geometry helpers & calculations
# ---------------------------
def apparent_length(true_len, angle_deg):
    """apparent length when inclined by angle w.r.t. plane: L * cos(angle)"""
    if true_len is None or angle_deg is None:
        return None
    return abs(true_len * math.cos(math.radians(angle_deg)))

def true_from_apparent(apparent, angle_deg):
    if apparent is None or angle_deg is None: return None
    if math.cos(math.radians(angle_deg)) == 0: return None
    return apparent / math.cos(math.radians(angle_deg))

# ---------------------------
# Drawing implementations for many shapes
# ---------------------------
def draw_point(draw, params, label='A'):
    infront = float(params.get('infront', 30))
    above = float(params.get('above', 30))
    # scale:
    max_mm = max(infront, above, 100)
    scale = fit_scale_for_box(max_mm*2.0, FRONT_BOX)
    # choose x=0 (center)
    fx, fz = transform_to_box_coords(0, above, FRONT_BOX, scale)
    tx, ty = transform_to_box_coords(0, infront, TOP_BOX, scale)
    # draw point
    draw.ellipse([fx-6, fz-6, fx+6, fz+6], outline=WHITE, width=LINE_HP)
    draw.ellipse([tx-5, ty-5, tx+5, ty+5], outline=WHITE, width=LINE_VP)
    # projector
    draw.line([(fx,fz),(tx,ty)], fill=WHITE, width=LINE_CONST)
    draw.text((fx+10, fz-10), label + "''", font=load_font(14), fill=WHITE)
    draw.text((tx+10, ty-10), label + "'", font=load_font(14), fill=WHITE)

def draw_line(draw, params):
    """
    Draw line AB projections. Params: true_length, angle_hp (deg), angle_vp (deg)
    We'll construct a 3D line from A to B using heuristics and project to front/top boxes.
    """
    L = params.get('true_length', 80.0) or 80.0
    alpha = params.get('angle_hp', 30.0) or 30.0  # with HP
    beta = params.get('angle_vp', 45.0) or 45.0   # with VP

    # Apparent lengths:
    top_ap = apparent_length(L, alpha)   # top view apparent length
    front_ap = apparent_length(L, beta)  # front view apparent length

    # choose XY-plane orientation so top and front aren't collinear
    plan_angle = 25  # degrees rotation in plan for aesthetic
    # compute vector components in 3D approximating chosen angles:
    # Component perpendicular to HP (z) = L * sin(alpha)
    dz = L * math.sin(math.radians(alpha))
    # remaining projected length on HP = L * cos(alpha)
    plan_len = L * math.cos(math.radians(alpha))
    dx = plan_len * math.cos(math.radians(plan_angle))
    dy = plan_len * math.sin(math.radians(plan_angle))
    # pick A at small negative x to center line
    Ax, Ay, Az = -L*0.1, 0.0, 0.0
    Bx, By, Bz = Ax + dx, Ay + dy, Az + dz

    # scale considering max extents
    max_mm = max(abs(Ax), abs(Bx), abs(Ay), abs(By), abs(Az), abs(Bz), L, 150)
    scale = fit_scale_for_box(max_mm*2.2, FRONT_BOX)
    # get front coords (x,z)
    A_front = transform_to_box_coords(Ax, Az, FRONT_BOX, scale)
    B_front = transform_to_box_coords(Bx, Bz, FRONT_BOX, scale)
    # top coords (x,y)
    A_top = transform_to_box_coords(Ax, Ay, TOP_BOX, scale)
    B_top = transform_to_box_coords(Bx, By, TOP_BOX, scale)
    # draw front (thick)
    draw.line([A_front, B_front], fill=WHITE, width=LINE_HP)
    draw.ellipse([A_front[0]-5,A_front[1]-5,A_front[0]+5,A_front[1]+5], outline=WHITE, width=LINE_HP)
    draw.ellipse([B_front[0]-5,B_front[1]-5,B_front[0]+5,B_front[1]+5], outline=WHITE, width=LINE_HP)
    # draw top (medium)
    draw.line([A_top, B_top], fill=WHITE, width=LINE_VP)
    draw.ellipse([A_top[0]-4,A_top[1]-4,A_top[0]+4,A_top[1]+4], outline=WHITE, width=LINE_VP)
    draw.ellipse([B_top[0]-4,B_top[1]-4,B_top[0]+4,B_top[1]+4], outline=WHITE, width=LINE_VP)
    # projectors
    draw.line([A_front, A_top], fill=WHITE, width=LINE_CONST)
    draw.line([B_front, B_top], fill=WHITE, width=LINE_CONST)
    # labels and info
    draw.text((A_front[0]-18, A_front[1]-26), "A''", font=load_font(14), fill=WHITE)
    draw.text((B_front[0]-18, B_front[1]-26), "B''", font=load_font(14), fill=WHITE)
    draw.text((A_top[0]-18, A_top[1]+8), "A'", font=load_font(14), fill=WHITE)
    draw.text((B_top[0]-18, B_top[1]+8), "B'", font=load_font(14), fill=WHITE)

    # compute and return dimension texts:
    info_lines = []
    info_lines.append(f"True length L = {L:.2f} mm")
    info_lines.append(f"Apparent (top) = {top_ap:.2f} mm (L*cos α), α={alpha}°")
    info_lines.append(f"Apparent (front) = {front_ap:.2f} mm (L*cos β), β={beta}°")
    # apparent inclinations with XY (angles of projection lines)
    # compute angle in front view relative to horizontal (positive right)
    dx_front = B_front[0] - A_front[0]
    dy_front = A_front[1] - B_front[1]  # inverted since y increases downward
    if dx_front == 0:
        theta_front = 90.0
    else:
        theta_front = math.degrees(math.atan2(abs(dy_front), abs(dx_front)))
    dx_top = B_top[0] - A_top[0]
    dy_top = A_top[1] - B_top[1]
    if dx_top == 0:
        theta_top = 90.0
    else:
        theta_top = math.degrees(math.atan2(abs(dy_top), abs(dx_top)))
    info_lines.append(f"Apparent angle in front view (with X) = {theta_front:.1f}°")
    info_lines.append(f"Apparent angle in top view (with X) = {theta_top:.1f}°")
    draw_info_panel(draw, "\n".join(info_lines))

def regular_polygon_vertices(n, side):
    # Build regular polygon vertices centered at origin (plan)
    # For polygon of side length s, circumradius R = s / (2*sin(pi/n))
    n = int(n)
    if n < 3: n = 3
    s = side
    R = s / (2 * math.sin(math.pi / n))
    pts = []
    # start one vertex at angle -90 so polygon "point" upwards for triangle
    for i in range(n):
        ang = -math.pi/2 + 2*math.pi*i/n
        x = R * math.cos(ang)
        y = R * math.sin(ang)
        pts.append((x,y))
    return pts

def draw_lamina(draw, params):
    shape = params.get('shape','triangle')
    sizes = params.get('size') or (params.get('sizes') and params.get('sizes')[0]) or 30.0
    surface_angle = params.get('surface_angle', 60.0)
    edge_angle = params.get('edge_angle_to_VP', 45.0)
    s = float(sizes)
    fnt = load_font(14)
    # center positions
    # compute max dimension for scale
    max_mm = max(s, 100)
    scale = fit_scale_for_box(max_mm*2.2, FRONT_BOX)
    # FRONT: show true shape foreshortened by cos(surface_angle) in vertical direction (z)
    center_f = ((FRONT_BOX[0]+FRONT_BOX[2])//2, (FRONT_BOX[1]+FRONT_BOX[3])//2)
    center_t = ((TOP_BOX[0]+TOP_BOX[2])//2, (TOP_BOX[1]+TOP_BOX[3])//2)
    if shape == 'triangle':
        # equilateral base
        side = s
        h = math.sqrt(3)/2 * side
        # front view: height foreshortened by cos(surface_angle)
        h_f = h * math.cos(math.radians(surface_angle))
        pts_f = [(center_f[0], center_f[1] - h_f/2),
                 (center_f[0] - side/2*scale, center_f[1] + h_f/2),
                 (center_f[0] + side/2*scale, center_f[1] + h_f/2)]
        # top view: use actual polygon rotated by edge_angle
        polygon_pts = regular_polygon_vertices(3, side)
        pts_t = []
        theta = math.radians(edge_angle)
        for (px,py) in polygon_pts:
            rx = px*math.cos(theta) - py*math.sin(theta)
            ry = px*math.sin(theta) + py*math.cos(theta)
            pts_t.append((center_t[0] + rx*scale, center_t[1] + ry*scale))
        draw.line([pts_f[0],pts_f[1],pts_f[2],pts_f[0]], fill=WHITE, width=LINE_HP)
        draw.line([pts_t[0],pts_t[1],pts_t[2],pts_t[0]], fill=WHITE, width=LINE_VP)
        # projectors: map each front vertex to nearest top vertex by index (heuristic)
        for i in range(3):
            draw.line([pts_f[i], pts_t[i]], fill=WHITE, width=LINE_CONST)
        draw_info_panel(draw, f"Triangle side={side} mm, surface angle={surface_angle}° to HP, edge rotated {edge_angle}° to VP")
    elif shape == 'square' or shape == 'rectangle':
        # rectangle: sizes maybe [w,h]
        if isinstance(sizes, (list,tuple)) and len(sizes)>=2:
            w = float(sizes[0]); h = float(sizes[1])
        else:
            w = s; h = s*1.5 if shape=='rectangle' else s
        # front: height foreshortened
        h_f = h * math.cos(math.radians(surface_angle))
        leftf = center_f[0] - (w/2)*scale; rightf = center_f[0] + (w/2)*scale
        topf = center_f[1] - (h_f/2); botf = center_f[1] + (h_f/2)
        draw.rectangle([leftf, topf, rightf, botf], outline=WHITE, width=LINE_HP)
        # top: rotate rectangle by edge_angle
        theta = math.radians(edge_angle)
        corners = [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
        pts_t = []
        for px,py in corners:
            rx = px*math.cos(theta) - py*math.sin(theta)
            ry = px*math.sin(theta) + py*math.cos(theta)
            pts_t.append((center_t[0]+rx*scale, center_t[1]+ry*scale))
        draw.line([pts_t[0],pts_t[1],pts_t[2],pts_t[3],pts_t[0]], fill=WHITE, width=LINE_VP)
        # projectors
        front_pts = [(leftf,topf),(rightf,topf),(rightf,botf),(leftf,botf)]
        for i in range(4):
            draw.line([front_pts[i], pts_t[i]], fill=WHITE, width=LINE_CONST)
        draw_info_panel(draw, f"Rectangle {w} x {h} mm, surface angle={surface_angle}°")
    elif shape in ('pentagon','hexagon','polygon'):
        n = 6 if shape=='hexagon' else 5 if shape=='pentagon' else 6
        polygon_pts = regular_polygon_vertices(n, s)
        pts_f = []
        pts_t = []
        theta = math.radians(edge_angle)
        for i,(px,py) in enumerate(polygon_pts):
            # front: foreshorten y (vertical) by cos(surface angle) and scale
            rx_f = center_f[0] + px*scale
            ry_f = center_f[1] + py*math.cos(math.radians(surface_angle))*scale
            pts_f.append((rx_f, ry_f))
            # top: rotate by edge angle
            rx = px*math.cos(theta) - py*math.sin(theta)
            ry = px*math.sin(theta) + py*math.cos(theta)
            pts_t.append((center_t[0] + rx*scale, center_t[1] + ry*scale))
        # draw
        draw.line(pts_f + [pts_f[0]], fill=WHITE, width=LINE_HP)
        draw.line(pts_t + [pts_t[0]], fill=WHITE, width=LINE_VP)
        # projectors pairwise
        for i in range(len(pts_f)):
            draw.line([pts_f[i], pts_t[i]], fill=WHITE, width=LINE_CONST)
        draw_info_panel(draw, f"{shape.capitalize()} side={s} mm, surface angle={surface_angle}°")
    elif shape == 'circle':
        r = s/2
        # front: circle foreshortened vertically => ellipse
        leftf = center_f[0] - r*scale
        rightf = center_f[0] + r*scale
        topf = center_f[1] - r*math.cos(math.radians(surface_angle))*scale
        botf = center_f[1] + r*math.cos(math.radians(surface_angle))*scale
        draw.ellipse([leftf, topf, rightf, botf], outline=WHITE, width=LINE_HP)
        # top: true circle
        leftt = center_t[0] - r*scale; rightt = center_t[0] + r*scale
        topt = center_t[1] - r*scale; bott = center_t[1] + r*scale
        draw.ellipse([leftt, topt, rightt, bott], outline=WHITE, width=LINE_VP)
        draw.line([(center_f[0], center_f[1]), (center_t[0], center_t[1])], fill=WHITE, width=LINE_CONST)
        draw_info_panel(draw, f"Circle diameter={s} mm, surface angle={surface_angle}°")
    else:
        draw_info_panel(draw, "Shape not recognized for lamina.")

def draw_prism(draw, params):
    # supports regular prism (n-sided) or cube/cuboid/cylinder/cone/pyramid
    typ = params.get('solid_type','prism')
    sizes = params.get('sizes', [])
    # Basic handling for common solids:
    # prism: sizes: [side, depth, height] or base side and height
    f = load_font(14)
    if typ == 'cube':
        a = sizes[0] if sizes else 50
        draw_info_panel(draw, f"Cube side {a} mm")
        # front: square a x a; top: square a x a
        scale = fit_scale_for_box(a*3, FRONT_BOX)
        center_f = ((FRONT_BOX[0]+FRONT_BOX[2])//2, (FRONT_BOX[1]+FRONT_BOX[3])//2)
        center_t = ((TOP_BOX[0]+TOP_BOX[2])//2, (TOP_BOX[1]+TOP_BOX[3])//2)
        l = a*scale/2
        draw.rectangle([center_f[0]-l, center_f[1]-l, center_f[0]+l, center_f[1]+l], outline=WHITE, width=LINE_HP)
        draw.rectangle([center_t[0]-l, center_t[1]-l, center_t[0]+l, center_t[1]+l], outline=WHITE, width=LINE_VP)
    elif typ == 'cuboid':
        if len(sizes)>=3:
            w,d,h = sizes[0], sizes[1], sizes[2]
        else:
            w,d,h = 80,50,60
        draw_info_panel(draw, f"Cuboid {w} x {d} base, height {h} mm")
        scale = fit_scale_for_box(max(w,d,h)*3, FRONT_BOX)
        center_f = ((FRONT_BOX[0]+FRONT_BOX[2])//2, (FRONT_BOX[1]+FRONT_BOX[3])//2)
        center_t = ((TOP_BOX[0]+TOP_BOX[2])//2, (TOP_BOX[1]+TOP_BOX[3])//2)
        lw = w*scale/2; lh = h*scale/2; ld = d*scale/2
        # front: rectangle w x h
        draw.rectangle([center_f[0]-lw, center_f[1]-lh, center_f[0]+lw, center_f[1]+lh], outline=WHITE, width=LINE_HP)
        # top: rectangle w x d
        draw.rectangle([center_t[0]-lw, center_t[1]-ld, center_t[0]+lw, center_t[1]+ld], outline=WHITE, width=LINE_VP)
    elif typ == 'cylinder':
        # sizes: [diameter, height]
        if len(sizes)>=2:
            diam, h = sizes[0], sizes[1]
        else:
            diam, h = 50, 80
        draw_info_panel(draw, f"Cylinder D={diam} mm, height={h} mm")
        scale = fit_scale_for_box(max(diam,h)*2.5, FRONT_BOX)
        center_f = ((FRONT_BOX[0]+FRONT_BOX[2])//2, (FRONT_BOX[1]+FRONT_BOX[3])//2)
        center_t = ((TOP_BOX[0]+TOP_BOX[2])//2, (TOP_BOX[1]+TOP_BOX[3])//2)
        r = diam/2 * scale
        # front: rectangle with top & bottom as ellipses maybe; simple representation:
        draw.ellipse([center_f[0]-r, center_f[1]-h*scale/2 - r/3, center_f[0]+r, center_f[1]-h*scale/2 + r/3], outline=WHITE, width=LINE_HP)
        draw.rectangle([center_f[0]-r, center_f[1]-h*scale/2 + r/3, center_f[0]+r, center_f[1]+h*scale/2 - r/3], outline=WHITE, width=LINE_HP)
        draw.ellipse([center_f[0]-r, center_f[1]+h*scale/2 - r/3, center_f[0]+r, center_f[1]+h*scale/2 + r/3], outline=WHITE, width=LINE_HP)
        # top: circle
        draw.ellipse([center_t[0]-r, center_t[1]-r, center_t[0]+r, center_t[1]+r], outline=WHITE, width=LINE_VP)
    elif typ == 'cone':
        # sizes: [diameter, height]
        if len(sizes)>=2:
            diam, h = sizes[0], sizes[1]
        else:
            diam, h = 50, 80
        draw_info_panel(draw, f"Cone D={diam} mm, height={h} mm")
        scale = fit_scale_for_box(max(diam,h)*3, FRONT_BOX)
        center_f = ((FRONT_BOX[0]+FRONT_BOX[2])//2, (FRONT_BOX[1]+FRONT_BOX[3])//2)
        center_t = ((TOP_BOX[0]+TOP_BOX[2])//2, (TOP_BOX[1]+TOP_BOX[3])//2)
        r = diam/2 * scale
        draw.polygon([(center_f[0]-r, center_f[1]+h*scale/2),(center_f[0]+r, center_f[1]+h*scale/2),(center_f[0], center_f[1]-h*scale/2)], outline=WHITE)
        draw.ellipse([center_t[0]-r, center_t[1]-r, center_t[0]+r, center_t[1]+r], outline=WHITE, width=LINE_VP)
    elif typ == 'pyramid':
        # assume square base pyramid sizes: base, height
        if len(sizes)>=2:
            base, height = sizes[0], sizes[1]
        else:
            base, height = 60, 80
        draw_info_panel(draw, f"Pyramid base={base} mm, height={height} mm")
        scale = fit_scale_for_box(base*3, FRONT_BOX)
        center_f = ((FRONT_BOX[0]+FRONT_BOX[2])//2, (FRONT_BOX[1]+FRONT_BOX[3])//2)
        center_t = ((TOP_BOX[0]+TOP_BOX[2])//2, (TOP_BOX[1]+TOP_BOX[3])//2)
        b = base*scale/2
        # top: square base
        draw.line([(center_t[0]-b, center_t[1]-b),(center_t[0]+b, center_t[1]-b),(center_t[0]+b, center_t[1]+b),(center_t[0]-b, center_t[1]+b),(center_t[0]-b, center_t[1]-b)], fill=WHITE, width=LINE_VP)
        # front: triangle elevation
        draw.polygon([(center_f[0]-b, center_f[1]+b),(center_f[0]+b, center_f[1]+b),(center_f[0], center_f[1]-height*scale)], outline=WHITE)
    elif typ == 'prism':
        # sizes: [n_sides(optional), side(optional), depth(optional), height(optional)]
        # We'll support n-sided regular prism: sizes = [n, side, height]
        if len(sizes) >= 3:
            n = int(sizes[0]); side = sizes[1]; height = sizes[2]
        elif len(sizes) == 2:
            n = 4; side = sizes[0]; height = sizes[1]
        elif len(sizes) == 1:
            n = 4; side = sizes[0]; height = sizes[0]*1.2
        else:
            n, side, height = 4, 50, 80
        draw_info_panel(draw, f"Regular prism n={n}, side={side} mm, height={height} mm\nLateral development shown below")
        # build polygon (plan) and draw top as plan, front as elevation rectangle for prism height
        scale = fit_scale_for_box(max(side*2, height)*3, FRONT_BOX)
        center_f = ((FRONT_BOX[0]+FRONT_BOX[2])//2, (FRONT_BOX[1]+FRONT_BOX[3])//2)
        center_t = ((TOP_BOX[0]+TOP_BOX[2])//2, (TOP_BOX[1]+TOP_BOX[3])//2)
        poly = regular_polygon_vertices(n, side)
        pts_t = [(center_t[0] + px*scale, center_t[1] + py*scale) for px,py in poly]
        # top
        draw.line(pts_t + [pts_t[0]], fill=WHITE, width=LINE_VP)
        # front: draw rectangle height x width approximate by bounding box width of polygon
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        width_mm = max(xs) - min(xs)
        w_px = width_mm * scale
        h_px = height * scale
        leftf = center_f[0] - w_px/2
        rightf = center_f[0] + w_px/2
        topf = center_f[1] - h_px/2
        botf = center_f[1] + h_px/2
        draw.rectangle([leftf, topf, rightf, botf], outline=WHITE, width=LINE_HP)
        # lateral development: draw n rectangles beside right
        dev_x = INFO_BOX[0] + 20
        dev_y = FRONT_BOX[1]
        face_w = side * scale
        face_h = height * scale
        for i in range(n):
            x0 = dev_x + i*(face_w + 6)
            y0 = dev_y
            draw.rectangle([x0, y0, x0+face_w, y0+face_h], outline=WHITE, width=LINE_HP)
    else:
        draw_info_panel(draw, "Solid type not yet implemented.")

# ---------------------------
# Utilities: audio transcription (optional)
# ---------------------------
def transcribe_audio_file(uploaded_file):
    # uses SpeechRecognition with Google Web Speech (requires internet)
    try:
        recognizer = sr.Recognizer()
        audio_bytes = uploaded_file.read()
        audio_data = sr.AudioFile(io.BytesIO(audio_bytes))
        with audio_data as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        return f"[Transcription failed: {e}]"

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("CADhelp — 2D Orthographic Projection Generator (First-angle / Auxiliary-plane)")
st.markdown("Type your textbook-style question, or upload a photo of the question, or upload an audio file (voice question). "
            "Examples of supported text: `Point A 20 mm infront of VP and 30 mm above HP`, "
            "`Line AB 80 mm long inclined 30 deg to HP and 45 deg to VP`, "
            "`Triangular lamina 30 mm sides surface 60° to HP`, "
            "`Prism 5 30 80` (5-sided prism, side=30mm, height=80mm)`.")

with st.sidebar:
    st.header("Input")
    input_text = st.text_area("Type question here (preferred for accuracy)", height=140)
    uploaded_img = st.file_uploader("Or upload photo of question (optional)", type=["png","jpg","jpeg"])
    uploaded_audio = st.file_uploader("Or upload audio file (optional) - wav/mp3", type=["wav","mp3","m4a","ogg"])
    force_type = st.selectbox("Force question type (optional)", ["Auto-detect","Point","Line","Lamina","Solid","Development"])
    st.markdown("Optional overrides (key=value comma separated). Examples: true_length=80, angle_hp=30, angle_vp=45, infront=20, above=30, shape=triangle, size=30, solid_type=prism, sizes=4,30,80")
    overrides = st.text_input("Overrides")

st.markdown("## Uploaded / Recognized Question")
if uploaded_img:
    st.image(uploaded_img, caption="Uploaded question image (display only). Type or override parameters for generation.", use_column_width=True)

recognized_text = ""
if uploaded_audio:
    st.info("Transcribing uploaded audio (uses Google Web Speech API). If transcription fails, type your question.")
    recognized_text = transcribe_audio_file(uploaded_audio)
    st.text_area("Transcribed text (edit if incorrect)", value=recognized_text, height=120)

# use typed text primarily; else use transcribed; else empty
text_for_parse = input_text.strip() if input_text.strip() else (recognized_text if recognized_text else "")

parsed = parse_text_question(text_for_parse)

# apply force option
if force_type != "Auto-detect":
    parsed['type'] = force_type.lower() if force_type!="Auto-detect" else parsed.get('type')

# apply overrides
if overrides.strip():
    try:
        for kv in overrides.split(','):
            if '=' in kv:
                k,v = kv.split('=',1); k=k.strip(); v=v.strip()
                # numeric?
                if re.match(r'^-?\d+(\.\d+)?(,\s*-?\d+(\.\d+)?)*$', v):
                    # single or comma-separated numbers
                    if ',' in v:
                        parsed.setdefault('params', {})[k] = [float(x) for x in v.split(',')]
                    else:
                        parsed.setdefault('params', {})[k] = float(v)
                else:
                    parsed.setdefault('params', {})[k] = v
    except Exception as e:
        st.warning("Could not parse overrides: " + str(e))

st.write("Detected:", parsed.get('type', 'None'), parsed.get('params', {}))
st.markdown("---")

# Generate button
if st.button("Generate projection"):
    img, draw = new_canvas()
    draw_xy_line(draw)
    typ = parsed.get('type')
    params = parsed.get('params', {})
    # route to drawing functions
    if typ == 'point' or force_type=="Point":
        draw_point(draw, params, label='A')
        # info
        infront = params.get('infront', 30); above = params.get('above', 30)
        info = f"Point projection\nInfront of VP (y) = {infront} mm\nAbove HP (z) = {above} mm"
        draw_info_panel(draw, info)
    elif typ == 'line' or force_type=="Line":
        draw_line(draw, params)
    elif typ == 'lamina' or force_type=="Lamina":
        draw_lamina(draw, params)
    elif typ == 'solid' or force_type=="Solid":
        draw_prism(draw, params)
    else:
        # fallback: prompt user
        draw_info_panel(draw, "Could not auto-detect question type. Try typing clearly or choose Force question type (sidebar). Example inputs:\n"
                              "- Point A 20 mm infront of VP and 30 mm above HP\n"
                              "- Line AB 80 mm long inclined 30 deg to HP and 45 deg to VP\n"
                              "- Triangular lamina 30 mm sides resting on HP surface 60° to HP\n"
                              "- Prism 5 30 80  (5-sided prism, side=30, height=80)")
    # show image
    st.image(img, use_column_width=True)
    # provide download
    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    st.download_button("Download projection PNG", data=buf, file_name="projection.png", mime="image/png")
else:
    st.info("Type a question, optionally upload a photo or audio, then click Generate projection.")

st.caption("Notes: This app aims to reproduce textbook CAD-style orthographic projections (first-angle for lamina, auxiliary-plane heuristics for solids). "
           "If you want automatic OCR on uploaded photos, or live voice recording from the browser (instead of uploading an audio file), tell me and I will add it (OCR requires Tesseract and extra dependencies).")
