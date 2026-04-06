from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from .api.routes import router
import time
from collections import defaultdict

app = FastAPI(
    title="AIM-Env API",
    description="API for AIM-Env email triage simulation",
    version="1.0.0"
)

# Rate limiting
request_counts = defaultdict(list)

def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old requests (keep last 60 seconds)
    request_counts[client_ip] = [t for t in request_counts[client_ip] if current_time - t < 60]
    
    # Check rate limit (100 requests per minute)
    if len(request_counts[client_ip]) >= 100:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_counts[client_ip].append(current_time)
    return call_next(request)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "AIM-Env API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}