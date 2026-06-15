import time
from threading import Lock
from functools import wraps
from flask import jsonify

class RateLimiter:
    def __init__(self, max_requests: int = 1, window_seconds: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}
        self._lock = Lock()
    
    def limit(self, key: str = "global"):
        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                with self._lock:
                    now = time.time()
                    if key not in self._requests:
                        self._requests[key] = []
                    
                    self._requests[key] = [
                        req_time for req_time in self._requests[key]
                        if now - req_time < self.window_seconds
                    ]
                    
                    if len(self._requests[key]) >= self.max_requests:
                        return jsonify({
                            "success": False,
                            "error": "Rate limit exceeded. Training is already in progress."
                        }), 429
                    
                    self._requests[key].append(now)
                
                return f(*args, **kwargs)
            return decorated
        return decorator

training_limiter = RateLimiter(max_requests=1, window_seconds=60)
