"""Convenience entrypoint: `python run_server.py`.

Equivalent to `uvicorn neurograph.server:app`, just without needing to
remember the module path. See neurograph/server.py's docstring for the
service's actual design (stateless-but-resumable, JSON in/out).
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("neurograph.server:app", host="127.0.0.1", port=8000, reload=False)
