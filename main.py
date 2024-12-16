from fastapi import FastAPI

# Create an instance of FastAPI
app = FastAPI()

# Define a route
@app.get("/")
def read_root():
    return "My changeg the Code."
