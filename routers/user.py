from fastapi import APIRouter, HTTPException,Depends,Body
from src.schemas.schemas import RegisterRequest, LoginRequest,UpdateUserRequest
from src.controllers.userController import userRegister, userLogin,userUpdate
from src.middleware.findUser import get_current_user


router = APIRouter()

@router.post("/register")
def register_user(req: RegisterRequest):
    try:
        result = userRegister(
            firstName=req.firstName,
            lastName=req.lastName,
            email=req.email,
            password=req.password
        )
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail={"status": "error", "message": result["message"]})

        return {
            "status": "success",
            "message": result["message"],
            "token": result["token"]
        }
    except HTTPException as he:
        raise he
    
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "error", "message": f"Internal Server Error: {str(e)}"})


@router.post("/login")
def login_user(req: LoginRequest):
    try:
        result = userLogin(
            email=req.email,
            password=req.password
        )
        if result["status"] == "error":
            raise HTTPException(status_code=401, detail={"status": "error", "message": result["message"]})

        return {
            "status": "success",
            "message": result["message"],
            "token": result["token"],
            "user":result["user"]
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Internal Server Error: {str(e)}"}
        )
    

@router.patch("/update")
def update_user(
    user_id: int = Depends(get_current_user),  req: UpdateUserRequest = Body(...)):
    try:
        update_data = req.dict(exclude_unset=True)
        if not update_data:
            raise HTTPException(status_code=400, detail={"status": "error", "message": "No fields provided for update"})
        # Call your controller function to update user
        result = userUpdate(user_id=user_id, **update_data)

        if result["status"] == "error":
            raise HTTPException(status_code=400, detail={"status": "error", "message": result["message"]})

        return {
            "status": "success",
            "message": result["message"],
            "user": result.get("user")  # optional: updated user data
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Internal Server Error: {str(e)}"}
        )