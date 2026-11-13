"""Strip an HTML file to text, keeping section headings as '## <heading>' lines. Usage: html2text.py in.html out.txt"""
import sys, html, re
from html.parser import HTMLParser
class P(HTMLParser):
    def __init__(s):
        super().__init__(); s.out=[]; s.skip=0; s.h=None
    def handle_starttag(s, t, a):
        if t in ("script","style"): s.skip+=1
        if re.fullmatch(r"h[1-6]", t): s.out.append("\n## "); s.h=t
        if t in ("p","li","div","br","tr","pre"): s.out.append("\n")
        if t=="a":
            href=dict(a).get("href")
            if href: s.out.append("")
    def handle_endtag(s, t):
        if t in ("script","style"): s.skip-=1
        if s.h==t: s.out.append("\n"); s.h=None
    def handle_data(s, d):
        if not s.skip: s.out.append(d)
p=P(); p.feed(open(sys.argv[1], errors="replace").read())
txt=re.sub(r"[ \t]+"," ","".join(p.out)); txt=re.sub(r"\n\s*\n+","\n",txt)
open(sys.argv[2],"w").write(txt)
