from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def demo_check():
    return "the demo check is successfull"
