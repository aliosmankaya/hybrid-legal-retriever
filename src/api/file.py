import os
import shutil

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..parser.file import Delete, Update, Upload

router = APIRouter()

path = os.getcwd() + "/data/"


@router.post("/upload")
def upload_service(params: Upload = Depends()):
    file_name = params.file.filename
    folder_path = path + params.law_name
    os.makedirs(folder_path + "/upload", exist_ok=True)

    with open(f"{folder_path}/upload/{file_name}", "wb") as buffer:
        buffer.write(params.file.file.read())
    return JSONResponse(content="Uploaded successfully", status_code=200)


@router.get("/list")
def list_service():
    files = os.listdir(path=path)
    return JSONResponse(content=files, status_code=200)


@router.put("/update")
def update_service(params: Update):
    os.rename(path + params.current_name, path + params.new_name)
    return JSONResponse(content="Updated successfully", status_code=200)


@router.delete("/delete")
def delete_service(params: Delete):
    shutil.rmtree(path=path + params.current_name)
    return JSONResponse(content="Deleted successfully", status_code=200)
