from fastapi import FastAPI

app = FastAPI(title="Fraud Detection App")


@app.get("/")
def root():
	return {"service": "fraud-detection-api", "docs": "/docs"}

@app.get("/health")
def health():
	return {"status" : "ok"}
