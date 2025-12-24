from fastapi import FastAPI

app = FastAPI()

@pp.get("/")
async def demo_check():
    return "the demo check is successfull"