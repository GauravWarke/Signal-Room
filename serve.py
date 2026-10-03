"""Local preview of the website at http://localhost:8123
Usage:  python serve.py            (default port 8123)
        python serve.py 8080       (pick another port)
Run it from anywhere; it always serves this project's docs/ folder. Stop with Ctrl+C."""
import functools, http.server, os, socketserver, sys, threading, webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves docs/. Also accepts /docs/... links and sends unknown pages to the overview instead of a 404."""
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")      # never show an old cached version
        super().end_headers()
    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/docs/"): self.path = self.path[5:]
        elif path in ("/docs", "/docs/"): self.path = "/"
        target = os.path.join(ROOT, self.path.split("?")[0].lstrip("/"))
        if not os.path.exists(target):
            self.send_response(302); self.send_header("Location", "/index.html"); self.end_headers(); return
        super().do_GET()

class Server(socketserver.TCPServer):
    allow_reuse_address = True

if not os.path.isfile(os.path.join(ROOT, "index.html")):
    sys.exit(f"Can't find {ROOT}\\index.html. Run this file from inside the decision-analytics-final folder.")

with Server(("127.0.0.1", PORT), Handler) as httpd:
    url = f"http://localhost:{PORT}/index.html"
    print(f"Serving {ROOT}\nOpen {url}  (Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
