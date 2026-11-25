"""Build tagged documentation from each release's own source checkout."""

import re
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


output = Path(sys.argv[1]).resolve()
tags = subprocess.check_output(["git", "tag", "--list", "v*"], text=True).splitlines()
for tag in tags:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        continue
    with tempfile.TemporaryDirectory(prefix="adcs-docs-") as temp:
        checkout = Path(temp) / "checkout"
        run("git", "worktree", "add", "--detach", str(checkout), tag)
        try:
            source = checkout / "docs" / "source"
            if not (source / "conf.py").is_file():
                continue
            # Match docs/Sphinx.md in the checked-out release. Keep hand-written
            # root pages; only regenerate files produced by sphinx-apidoc.
            run("sphinx-apidoc", "-f", "-e", "--maxdepth", "1",
                "-o", str(source), str(checkout / "ADCS"))
            run("sphinx-build", "-b", "html", str(source), str(output / tag),
                cwd=checkout / "docs")
        finally:
            run("git", "worktree", "remove", "--force", str(checkout))

# The archived conf.py files predate sphinx-multiversion. Add the same picker
# to every generated page, including those older releases.
versions = ["main"] + [tag for tag in tags if (output / tag / "index.html").exists()]
for page in output.rglob("*.html"):
    if page == output / "index.html":
        continue
    relative_root = os.path.relpath(output, page.parent).replace(os.sep, "/")
    options = "".join(f'<option value="{relative_root}/{version}/index.html">{version}</option>'
                      for version in versions)
    picker = ("<script>document.addEventListener('DOMContentLoaded',function(){"
              "var s=document.createElement('select');s.setAttribute('aria-label','Documentation version');"
              f"s.innerHTML='{options}';"
              "s.value=Array.from(s.options).find(o=>location.pathname.includes('/'+o.text+'/'))?.value||s.options[0].value;"
              "s.onchange=function(){location.href=s.value};"
              "var host=document.querySelector('.sidebar-sticky')||document.body;host.prepend(s);"
              "});</script>")
    html = page.read_text(encoding="utf-8")
    page.write_text(html.replace("</body>", picker + "</body>"), encoding="utf-8")
