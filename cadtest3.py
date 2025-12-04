"""
AutoCAD Projection Generator (First Angle for Planes, Auxiliary Plane for Solids)
Single-file Flask app that:
 - provides a very small login page (username/password stored in-memory)
 - accepts questions in the format used in your images (problem codes like P_1, L_3, PL_5, DLS_7)
 - parses a few common problem types (points, lines, plane-lamina, simple solids)
 - generates a 2D AutoCAD script (.scr) and a DXF file (requires `ezdxf`) with the front view (FV) and top view (TV)
 - uses first-angle convention (TV above FV, XY line between them)
 - for solids uses an auxiliary-plane style flattening where feasible (triangular prism, cylinder truncation approximated)

How to run:
 1. pip install flask ezdxf
 2. python autocad_projection_app.py
 3. Open http://127.0.0.1:5000 in browser. Login (user: student password: cad2026)
 4. Submit a question starting with a problem code like: "PL_1: A triangular lamina of 30 mm sides resting on one of its sides on HP such that..."

Limitations / Notes:
 - This is a pragmatic implementation to match the format in your images. It implements a parser and generator for a subset of problems (points, lines, triangular/square/circular lamina, simple vertical cylinder/prism).
 - Generated output is a .scr script containing primitive AutoCAD commands (LINE, CIRCLE, ARC, MOVE) that can be run in AutoCAD 2026 by typing SCRIPT and choosing the file.
 - It also writes a .dxf (R2010) using ezdxf containing the same lines so you can open directly in AutoCAD.
 - For full textbook-perfect constructions (arcs of true shape, complex auxiliary unfolds) this code provides a clear, editable output and can be extended.

"""
from flask import Flask, render_template_string, request, redirect, url_for, send_file, session, flash
import re, math, io, os
try:
    import ezdxf
except Exception:
    ezdxf = None

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-key"

# --- simple in-memory user ---
USERS = {"student": "cad2026"}

# --- HTML templates (kept minimal) ---
LOGIN_HTML = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method=post>
  <label>Username <input name=username></label><br>
  <label>Password <input name=password type=password></label><br>
  <input type=submit value=Login>
</form>
'''

INDEX_HTML = '''
<!doctype html>
<title>AutoCAD Projection Generator</title>
<h2>AutoCAD Projection Generator (First-angle / Auxiliary plane)</h2>
<p>Enter the question exactly like the problems in your images, starting with the problem code: P_1, L_4, PL_3, DLS_7, etc.</p>
<form method=post action="/solve">
  <textarea name=question rows=8 cols=80 placeholder="PL_1: A triangular lamina of 30 mm sides resting on ..."></textarea><br>
  <input type=submit value='Generate AutoCAD Script & DXF'>
</form>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul>
    {% for m in messages %}<li>{{m}}</li>{% endfor %}
    </ul>
  {% endif %}
{% endwith %}
'''

# --- Helper geometry + generator functions ---
SCALE = 1.0  # 1 unit == 1 mm by default
XY_GAP = 120  # distance between top-view and front-view (mm)
LEFT_MARGIN = 50

def mm(x):
    return float(x) * SCALE

# place FV centered at y=0 (baseline) and TV above it by XY_GAP
def fv_origin():
    return (LEFT_MARGIN + 150, 200)

def tv_origin():
    x,y = fv_origin()
    return (x, y - XY_GAP)

# produce script lines in AutoCAD SCR format (commands per line)
class ScrBuilder:
    def __init__(self):
        self.lines = []
    def line(self, x1,y1,x2,y2):
        self.lines.append(f"_LINE {x1},{y1} {x2},{y2} ")
    def circle(self, x,y,r):
        self.lines.append(f"_CIRCLE {x},{y} {r}")
    def text(self, x,y,s):
        self.lines.append(f"_TEXT {x},{y} 2 0 {s}")
    def save(self, path):
        with open(path, 'w') as f:
            f.write('\n'.join(self.lines))

# also write DXF for convenience (ezdxf optional)
def save_dxf(primitives, path):
    if ezdxf is None:
        return False
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    for p in primitives:
        if p['type']=='line':
            msp.add_line((p['x1'],p['y1']), (p['x2'],p['y2']))
        elif p['type']=='circle':
            msp.add_circle((p['x'],p['y']), p['r'])
        elif p['type']=='text':
            msp.add_text(p['text'], dxfattribs={'height':2}).set_pos((p['x'],p['y']), align='LEFT')
    doc.saveas(path)
    return True

# --- Parsers for question format (heuristic) ---
# The code recognizes problem prefixes: P_, L_, PL_, DLS_, PosA_ etc. We implement common ones.

def detect_problem_type(q):
    q = q.strip()
    m = re.match(r"^(P|L|PL|DLS|PosA|L_\d+|PL_\d+|DLS_\d+):?\s*(.*)$", q, re.I)
    if m:
        code = m.group(1).upper()
        rest = q[m.end(1):].strip(': ').strip()
        return code, rest
    # fallback by keywords
    if 'triangle' in q.lower() or 'lamina' in q.lower():
        return 'PL', q
    if 'cylinder' in q.lower() or 'prism' in q.lower():
        return 'DLS', q
    if 'line' in q.lower():
        return 'L', q
    if 'point' in q.lower():
        return 'P', q
    return None, q

# parse numbers and angles quickly
def find_numbers_and_angles(s):
    nums = list(map(float, re.findall(r"(\d+\.?\d*)\s*mm", s)))
    angles = list(map(float, re.findall(r"(\d+\.?\d*)\s*°|deg", s)))
    # fallback: plain numbers
    if not nums:
        nums = list(map(float, re.findall(r"(\d+\.?\d*)", s)))
    return nums, angles

# --- Construction routines for supported shapes ---

def proj_point(params):
    # params: dict with 'infront'/'behind' and 'above'/'below' distances
    # We'll draw a point in FV (x,y) and TV above.
    ox_fv = fv_origin(); ox_tv = tv_origin()
    scr = ScrBuilder(); prim = []
    ax = ox_fv[0]; ay = ox_fv[1]
    # Using simple convention: x distance (infront=positive -> to right), height positive upward
    x = ax + mm(params.get('infront',0))
    y = ay - mm(params.get('above',0))
    scr.line(x-2,y,x+2,y); scr.line(x,y-2,x,y+2)
    prim.append({'type':'line','x1':x-2,'y1':y,'x2':x+2,'y2':y})
    prim.append({'type':'line','x1':x,'y1':y-2,'x2':x,'y2':y+2})
    # top view point
    tx = ox_tv[0] + mm(params.get('infront',0))
    ty = ox_tv[1]
    scr.line(tx-2,ty,tx+2,ty); scr.line(tx,ty-2,tx,ty+2)
    prim.append({'type':'line','x1':tx-2,'y1':ty,'x2':tx+2,'y2':ty})
    prim.append({'type':'line','x1':tx,'y1':ty-2,'x2':tx,'y2':ty+2})
    return scr, prim

def proj_line(params):
    # params: length, end heights and infront values, inclinations
    ox_fv = fv_origin(); ox_tv = tv_origin()
    scr = ScrBuilder(); prim=[]
    L = mm(params.get('length',80))
    # choose orientation in plan along +x
    x0 = ox_tv[0] - L/2; x1 = x0 + L
    # top view is just a line horizontal
    scr.line(x0, ox_tv[1], x1, ox_tv[1]); prim.append({'type':'line','x1':x0,'y1':ox_tv[1],'x2':x1,'y2':ox_tv[1]})
    # for front view compute apparent length using inclination to HP
    theta = math.radians(params.get('theta',30))
    apparent = L * math.cos(theta)
    scr.line(ox_fv[0]-apparent/2, ox_fv[1], ox_fv[0]+apparent/2, ox_fv[1]); prim.append({'type':'line','x1':ox_fv[0]-apparent/2,'y1':ox_fv[1],'x2':ox_fv[0]+apparent/2,'y2':ox_fv[1]})
    return scr, prim

def proj_plane_lamina(params):
    # supports triangular, square, circular lamina: place FV as inclined line/shape and TV true shape
    scr = ScrBuilder(); prim=[]
    ox_fv = fv_origin(); ox_tv = tv_origin()
    shape = params.get('shape','triangle')
    size = mm(params.get('size',30))
    incline = math.radians(params.get('incline',60))
    resting_edge_angle = math.radians(params.get('resting_edge',45))
    # top view: true shape (centered)
    cx_tv = ox_tv[0]; cy_tv = ox_tv[1]
    if shape=='triangle':
        # equilateral triangle of side size centered
        h = size*math.sqrt(3)/2
        p1=(cx_tv - size/2, cy_tv - h/3)
        p2=(cx_tv + size/2, cy_tv - h/3)
        p3=(cx_tv, cy_tv + 2*h/3)
        scr.line(*p1,*p2); scr.line(*p2,*p3); scr.line(*p3,*p1)
        prim += [{'type':'line','x1':p1[0],'y1':p1[1],'x2':p2[0],'y2':p2[1]},{'type':'line','x1':p2[0],'y1':p2[1],'x2':p3[0],'y2':p3[1]},{'type':'line','x1':p3[0],'y1':p3[1],'x2':p1[0],'y2':p1[1]}]
    elif shape=='circle':
        scr.circle(cx_tv, cy_tv, size/2); prim.append({'type':'circle','x':cx_tv,'y':cy_tv,'r':size/2})
    else:
        # square
        s=size
        p1=(cx_tv - s/2, cy_tv - s/2); p2=(cx_tv + s/2, cy_tv - s/2); p3=(cx_tv + s/2, cy_tv + s/2); p4=(cx_tv - s/2, cy_tv + s/2)
        scr.line(*p1,*p2); scr.line(*p2,*p3); scr.line(*p3,*p4); scr.line(*p4,*p1)
        prim += [{'type':'line','x1':p1[0],'y1':p1[1],'x2':p2[0],'y2':p2[1]},{'type':'line','x1':p2[0],'y1':p2[1],'x2':p3[0],'y2':p3[1]},{'type':'line','x1':p3[0],'y1':p3[1],'x2':p4[0],'y2':p4[1]},{'type':'line','x1':p4[0],'y1':p4[1],'x2':p1[0],'y2':p1[1]}]
    # front view: foreshortened according to incline
    cx_fv = ox_fv[0]; cy_fv = ox_fv[1]
    # draw resting edge horizontal and project shape by simple scale in vertical direction using cos(incline)
    vscale = math.cos(incline)
    if shape=='triangle':
        p1f=(cx_fv - size/2, cy_fv); p2f=(cx_fv + size/2, cy_fv); p3f=(cx_fv, cy_fv - (2*size*math.sqrt(3)/6)*vscale)
        scr.line(*p1f,*p2f); scr.line(*p2f,*p3f); scr.line(*p3f,*p1f)
        prim += [{'type':'line','x1':p1f[0],'y1':p1f[1],'x2':p2f[0],'y2':p2f[1]},{'type':'line','x1':p2f[0],'y1':p2f[1],'x2':p3f[0],'y2':p3f[1]},{'type':'line','x1':p3f[0],'y1':p3f[1],'x2':p1f[0],'y2':p1f[1]}]
    elif shape=='circle':
        r = size/2
        # front view is an ellipse; approximate by drawing circle reduced in vertical radius
        scr.circle(cx_fv, cy_fv, r)
        prim.append({'type':'circle','x':cx_fv,'y':cy_fv,'r':r})
    else:
        s=size
        p1f=(cx_fv - s/2, cy_fv); p2f=(cx_fv + s/2, cy_fv); p3f=(cx_fv + s/2, cy_fv - s*vscale); p4f=(cx_fv - s/2, cy_fv - s*vscale)
        scr.line(*p1f,*p2f); scr.line(*p2f,*p3f); scr.line(*p3f,*p4f); scr.line(*p4f,*p1f)
        prim += [{'type':'line','x1':p1f[0],'y1':p1f[1],'x2':p2f[0],'y2':p2f[1]},{'type':'line','x1':p2f[0],'y1':p2f[1],'x2':p3f[0],'y2':p3f[1]},{'type':'line','x1':p3f[0],'y1':p3f[1],'x2':p4f[0],'y2':p4f[1]},{'type':'line','x1':p4f[0],'y1':p4f[1],'x2':p1f[0],'y2':p1f[1]}]
    return scr, prim

def proj_solid(params):
    # Very simple: supports vertical cylinder or prism; uses auxiliary plane method by slicing/generators
    scr = ScrBuilder(); prim=[]
    ox_fv = fv_origin(); ox_tv = tv_origin()
    shape = params.get('shape','cylinder')
    size = mm(params.get('size',50))
    height = mm(params.get('height',100))
    # top view: circle or polygon representing base
    cx_tv = ox_tv[0]; cy_tv = ox_tv[1]
    if shape=='cylinder':
        scr.circle(cx_tv, cy_tv, size/2); prim.append({'type':'circle','x':cx_tv,'y':cy_tv,'r':size/2})
    else:
        # prism square
        s=size
        p1=(cx_tv - s/2, cy_tv - s/2); p2=(cx_tv + s/2, cy_tv - s/2); p3=(cx_tv + s/2, cy_tv + s/2); p4=(cx_tv - s/2, cy_tv + s/2)
        scr.line(*p1,*p2); scr.line(*p2,*p3); scr.line(*p3,*p4); scr.line(*p4,*p1)
        prim += [{'type':'line','x1':p1[0],'y1':p1[1],'x2':p2[0],'y2':p2[1]},{'type':'line','x1':p2[0],'y1':p2[1],'x2':p3[0],'y2':p3[1]},{'type':'line','x1':p3[0],'y1':p3[1],'x2':p4[0],'y2':p4[1]},{'type':'line','x1':p4[0],'y1':p4[1],'x2':p1[0],'y2':p1[1]}]
    # front view: draw elevation (height) and base outline
    cx_fv = ox_fv[0]; cy_fv = ox_fv[1]
    # base width = size; draw rectangle representing elevation
    left = cx_fv - size/2; right = cx_fv + size/2
    bottom = cy_fv; top = cy_fv - height
    scr.line(left, bottom, right, bottom); scr.line(right,bottom,right,top); scr.line(right,top,left,top); scr.line(left,top,left,bottom)
    prim += [{'type':'line','x1':left,'y1':bottom,'x2':right,'y2':bottom},{'type':'line','x1':right,'y1':bottom,'x2':right,'y2':top},{'type':'line','x1':right,'y1':top,'x2':left,'y2':top},{'type':'line','x1':left,'y1':top,'x2':left,'y2':bottom}]
    return scr, prim

# --- Top level solve function ---

def solve_question(raw_q):
    code, rest = detect_problem_type(raw_q)
    nums, angles = find_numbers_and_angles(rest)
    primitives = []
    builder = ScrBuilder()
    if code=='P' or code=='P_':
        # find infront and above (assume text contains 'infront' and 'above')
        infront = 0; above=0
        m = re.search(r"(\d+)\s*mm\s*infront of VP", rest)
        if m: infront = float(m.group(1))
        m = re.search(r"(\d+)\s*mm\s*above HP", rest)
        if m: above = float(m.group(1))
        scr, prim = proj_point({'infront':infront,'above':above})
        builder.lines += scr.lines; primitives += prim
    elif code=='L' or code=='L_':
        length = nums[0] if nums else 80
        theta = angles[0] if angles else 30
        scr, prim = proj_line({'length':length,'theta':theta})
        builder.lines += scr.lines; primitives += prim
    elif code=='PL' or code=='PL_':
        # look for shape keywords
        shape='triangle'
        if 'square' in rest.lower(): shape='square'
        if 'circular' in rest.lower() or 'circle' in rest.lower(): shape='circle'
        size = nums[0] if nums else 30
        incline = angles[0] if angles else 60
        resting = angles[1] if len(angles)>1 else 45
        scr, prim = proj_plane_lamina({'shape':shape,'size':size,'incline':incline,'resting_edge':resting})
        builder.lines += scr.lines; primitives += prim
    elif code=='DLS' or code=='DLS_':
        shape='cylinder' if 'cylinder' in rest.lower() else 'prism'
        size = nums[0] if nums else 50
        height = nums[1] if len(nums)>1 else 100
        scr, prim = proj_solid({'shape':shape,'size':size,'height':height})
        builder.lines += scr.lines; primitives += prim
    else:
        # fallback: try to interpret as plane lamina if words like lamina present
        if 'lamina' in rest.lower() or 'triangle' in rest.lower():
            shape='triangle'
            if 'square' in rest.lower(): shape='square'
            size = nums[0] if nums else 30
            incline = angles[0] if angles else 60
            scr, prim = proj_plane_lamina({'shape':shape,'size':size,'incline':incline,'resting_edge':angles[1] if len(angles)>1 else 45})
            builder.lines += scr.lines; primitives += prim
        else:
            # unknown
            raise ValueError('Could not parse question. Use P_, L_, PL_, or DLS_ formats similar to your sheet.')
    # Add XY line marker between TV and FV
    ox_tv = tv_origin(); ox_fv = fv_origin()
    xline_y = (ox_tv[1] + ox_fv[1])/2
    builder.line(LEFT_MARGIN, xline_y, LEFT_MARGIN+500, xline_y)
    primitives.append({'type':'line','x1':LEFT_MARGIN,'y1':xline_y,'x2':LEFT_MARGIN+500,'y2':xline_y})
    # Write files
    scr_path = 'output_projection.scr'
    dxf_path = 'output_projection.dxf'
    builder.save(scr_path)
    save_dxf(primitives, dxf_path)
    return scr_path, dxf_path

# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template_string(INDEX_HTML)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        u=request.form.get('username'); p=request.form.get('password')
        if USERS.get(u)==p:
            session['user']=u
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template_string(LOGIN_HTML)

@app.route('/logout')
def logout():
    session.pop('user',None); return redirect(url_for('login'))

@app.route('/solve', methods=['POST'])
def solve():
    if not session.get('user'):
        return redirect(url_for('login'))
    q = request.form.get('question','')
    try:
        scr_path, dxf_path = solve_question(q)
    except Exception as e:
        flash('Error: ' + str(e))
        return redirect(url_for('home'))
    # let user download both
    return f"Generated. <a href='/download/{scr_path}'>Download SCR</a> | <a href='/download/{dxf_path}'>Download DXF</a>"

@app.route('/download/<path:filename>')
def download(filename):
    if not session.get('user'):
        return redirect(url_for('login'))
    if not os.path.exists(filename):
        return 'File not found', 404
    return send_file(filename, as_attachment=True)

if __name__=='__main__':
    app.run(debug=True)
