from fastapi import FastAPI

# Create an instance of FastAPI
app = FastAPI()

# Define a route
@app.get("/")
def read_root():
    return "I changed the Code."


@app.get("/sultan/cgpa")
def read_root():
    return "Sultan CGPA : 3.58"


@app.get("/ali/cgpa")
def read_root():
    return "Ali CGPA : 3.44"
