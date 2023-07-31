from fastapi import FastAPI, UploadFile, APIRouter, Request, Form
from tracker.football_track_bytetrack import Football_bytetrack
# from team_matching.team_match import TeamMatch
import uvicorn
import json
from tracker.helper import imgByteToNumpy

temp = APIRouter()
app = FastAPI()
# app.include_router(temp, prefix='/api')


tracker = Football_bytetrack()
# team_matching = TeamMatch()


@app.get("/")
async def root():
    return {"message": "Hello rld"}


@app.post("/players-detection")
async def playersDetect(file: UploadFile):
    content = file.file.read()
    result = tracker.detect_players(content)

    return {
        "data": result
    }


@app.post("/team-matching")
async def teamMatching(file: UploadFile, team=Form(...)):
    img = file.file.read()
    img = imgByteToNumpy(img)
    team = json.loads(team)
    # team_matching.match(img, team=team)

    return {
        "success": True
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
