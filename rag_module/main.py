from fastapi import FastAPI
from rag_module.api.routes import router as rag_router
from fastapi.responses import RedirectResponse

app = FastAPI(
    title="RAG Module API",
    description="A modular Retrieval-Augmented Generation backend",
    version="0.1.0"
)

app.include_router(rag_router)

# For quick local testing:
# uvicorn rag_module.main:app --reload
# in rag_module/main.py

@app.get("/")
def root():
    return RedirectResponse(url="/docs")