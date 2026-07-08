import os
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

DIRECTORY = r"C:\Projects\Sight\runs\sd_godot"
Handler = partial(SimpleHTTPRequestHandler, directory=DIRECTORY)

class Server(ThreadingHTTPServer):
    daemon_threads = True

httpd = Server(("127.0.0.1", 8791), Handler)
print("SERVING", DIRECTORY, "on http://127.0.0.1:8791/", flush=True)
httpd.serve_forever()
