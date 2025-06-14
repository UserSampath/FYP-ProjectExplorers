from fastapi import APIRouter, HTTPException,Depends,Body
from src.schemas.schemas import RegisterRequest, LoginRequest,UpdateUserRequest,APIResponse
from src.controllers.userController import userRegister, userLogin,userUpdate,getUserDetails
from src.middleware.findUser import get_current_user
from src.exception import raise_custom_error
router = APIRouter()

@router.post("/register",response_model=APIResponse)
def register_user(req: RegisterRequest):
    try:
        result = userRegister(
            firstName=req.firstName,
            lastName=req.lastName,
            email=req.email,
            password=req.password
        )
        if result["status"] == "error":
            raise_custom_error(400, result["message"])

        return {
            "status": "success",
            "success": True,
            "message": result["message"],
            "data": {"token": result["token"]}
            
        }
    except HTTPException as he:
        raise he
    
    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")


@router.post("/login",response_model=APIResponse)
def login_user(req: LoginRequest):
    try:
        result = userLogin(
            email=req.email,
            password=req.password
        )
        if result["status"] == "error":
            raise_custom_error(401, result["message"])

        return {
            "status": "success",
            "success": True,
            "message": result["message"],
            "data":{
                "token": result["token"],
                "user": result["user"]
            }
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise_custom_error(
            500, f"Internal Server Error: {str(e)}"
        )
        
    

@router.patch("/update",response_model=APIResponse)
def update_user(
    user_id: int = Depends(get_current_user),  req: UpdateUserRequest = Body(...)):
    try:
        update_data = req.dict(exclude_unset=True)
        if not update_data:
            raise_custom_error(400, "No fields provided for update")
        # Call your controller function to update user
        result = userUpdate(user_id=user_id, **update_data)

        if result["status"] == "error":
            raise_custom_error(400, result["message"])

        return {
            "status": "success",
            "message": result["message"],
            "success": True,
            "data":{
            "user": result.get("user")  
            }
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")

@router.post("/verify",response_model=APIResponse)
def verify_user(user_id: str = Depends(get_current_user)):
    try:
        result = getUserDetails(user_id)

        if result["status"] != "success":
            raise_custom_error(404, result["message"])

        return {
            "success": True,
            "status": "success",
            "message": "User fetched successfully",
            "data": result["user"]
            }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")