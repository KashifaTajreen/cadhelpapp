from flask import Flask, request, render_template_string, redirect, session, send_file
import os

app = Flask(__name__)
app.secret_key = "autocad_secret"

USERNAME = "student"
PASSWORD = "cad2026"

LOGIN_PAGE = """
<h2>Login</h2>
<form method="post">
Username: <input name="username"><br>
Password: <input type="password" name="password"><br>
<button type="submit">Login</button>
</form>
"""

HOME_PAGE = """
<h2>AutoCAD Question Input</h2>
<form action="/solve" method="post">
<textarea name="question" rows="6" cols="60"
placeholder="PL_1: A triangular lamina of 30 mm sides resting on HP..."></textarea><br>
<button type="submit">Generate AutoCAD Script</button>
</form>
<a href="/logout">Logout</a>
"""

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == USERNAME and request.form["password"] == PASSWORD:
            session["user"] = USERNAME
            return redirect("/home")
    return render_template_string(LOGIN_PAGE)

@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/")
    return render_template_string(HOME_PAGE)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

@app.route("/solve", methods=["POST"])
def solve():
    if "user" not in session:
        return redirect("/")

    question = request.form["question"].strip()

    if not (question.startswith("P_") or question.startswith("L_") or question.startswith("PL_") or question.startswith("DLS_")):
        return "❌ Invalid Question Format! Use P_, L_, PL_, or DLS_ only."

    # ✅ AutoCAD Script Output
    script = f"""
_LINE 0,0 100,0
_CIRCLE 50,30 20
_TEXT 10,40 2 0 {question}
"""

    with open("output.scr", "w") as f:
        f.write(script)

    return send_file("output.scr", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

