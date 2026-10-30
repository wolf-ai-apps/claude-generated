"""Build the course website from the level READMEs.

    python3 site/build.py                 # writes site/index.html
    python3 -m http.server -d site 8080   # then open http://localhost:8080

The page is one self-contained HTML file; it loads fonts and the Markdown
renderer from public CDNs, so the browser needs internet access.
"""
import json, os, re, glob, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "index.html")
GH = "https://github.com/wolf-ai-apps/claude-generated/blob/ai-inference-academy/"

TIERS = {"00":"Laptop","01":"Laptop","02":"GPU","03":"GPU","04":"Laptop","05":"Laptop","06":"Laptop",
         "07":"Laptop","08":"GPU","09":"Laptop","10":"Laptop + K8s","11":"AWS","12":"AWS GPU","13":"AWS GPU",
         "14":"Laptop","15":"AWS GPU"}
PARTS = [("Foundations",["00","01","02","03"]),("Model and engine",["04","05","06","07"]),
         ("Serving",["08","09","10"]),("Cloud and production",["11","12","13","14","15"])]

pages = []
def add(pid, title, path, num=None):
    md = open(os.path.join(ROOT, path)).read()
    pages.append({"id": pid, "title": title, "path": path, "num": num, "tier": TIERS.get(num), "md": md})

add("home", "Course home", "README.md")
add("curriculum", "Curriculum", "CURRICULUM.md")
for d in sorted(glob.glob(os.path.join(ROOT, "levels/*/README.md"))):
    rel = os.path.relpath(d, ROOT)
    num = rel.split("/")[1][:2]
    first = open(d).readline().strip().lstrip("# ")
    title = re.sub(r"^Level \d+:\s*", "", first)
    add("level-" + num, title, rel, num)

def fix_links(p):
    base = os.path.dirname(p["path"])
    def repl(m):
        text, target = m.group(1), m.group(2)
        if target.startswith(("http", "#", "mailto")):
            return m.group(0)
        t = os.path.normpath(os.path.join(base, target))
        if t == "README.md": return f"[{text}](#home)"
        if t == "CURRICULUM.md": return f"[{text}](#curriculum)"
        mm = re.match(r"levels/(\d\d)-[^/]+/README\.md$", t)
        if mm: return f"[{text}](#level-{mm.group(1)})"
        return f"[{text}]({GH}{t})"
    p["md"] = re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", repl, p["md"])
for p in pages: fix_links(p)

tpl = open(os.path.join(HERE, "template.html")).read()
data = json.dumps({"pages": pages, "parts": PARTS, "gh": GH}).replace("</", "<\\/")
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n')
body = tpl.replace("__DATA__", data)
# the <title>, meta and <style> block go in <head>; everything after </style> is the body
split = body.index("</style>") + len("</style>")
html = head + body[:split] + "\n</head>\n<body>\n" + body[split:] + "\n</body>\n</html>\n"
if "--fragment" in sys.argv:  # for publishing as a claude.ai artifact, which adds its own skeleton
    html = body
open(OUT, "w").write(html)
print(f"wrote {os.path.relpath(OUT)} ({os.path.getsize(OUT) // 1024} KB). Serve it with:")
print("  python3 -m http.server -d site 8080   ->  http://localhost:8080")
