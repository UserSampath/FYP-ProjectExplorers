import uuid
import bcrypt
from sqlalchemy import Table, MetaData, insert,select,update
from jose import jwt
from datetime import datetime, timedelta
from src.utils import get_engine
import os
from dotenv import load_dotenv


load_dotenv() 
JWT_SECRET = os.getenv("JWT_SECRET")

# JWT configuration
JWT_ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def generate_jwt_token(user_data: dict) -> str:
    payload = {
        "exp": datetime.utcnow() + timedelta(hours=1),
        "userId": user_data["user_id"],  # UUID used as subject
        "fullName": user_data["fullName"]
        
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def userRegister(fullName: str, email: str, password: str):
    try:
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)
        users_table = metadata.tables.get("processed_users")

        if users_table is None:
            raise Exception("User table does not exist")

        with engine.connect() as conn:
            # Check if email already exists
            select_stmt = select(users_table).where(users_table.c.email == email)
            result = conn.execute(select_stmt).first()
            if result is not None:
                return {
                    "status": "error",
                    "message": "Email already registered"
                }

            # Email not found, proceed with insert
            user_id = str(uuid.uuid4())
            hashed_password = hash_password(password)

            insert_stmt = insert(users_table).values(
                user_id=user_id,
                fullName=fullName,
                email=email,
                password=hashed_password,
                train=False
            )
            conn.execute(insert_stmt)
            conn.commit()

            # Fetch the inserted user
            new_user_stmt = select(users_table).where(users_table.c.user_id == user_id)
            new_user = conn.execute(new_user_stmt).first()

            token = generate_jwt_token({
                "user_id": user_id,
                "fullName": fullName,
            })

            # Convert row to dictionary and select only specific fields
            user_data = dict(new_user._mapping) if new_user else {}
            filtered_user = {
                "fullName": user_data.get("fullName"),
                "email": user_data.get("email"),
                "expertise_level": user_data.get("expertise_level"),
                "years_of_experience": user_data.get("years_of_experience"),
                "familiar_technologies": user_data.get("familiar_technologies"),
            }

        return {
            "status": "success",
            "message": "User registered successfully",
            "token": token,
            "user": filtered_user
        }

    except Exception as e:
        print("Error in register:", e)
        return {
            "status": "error",
            "message": str(e)
        }

def userLogin(email: str, password: str):
    try:
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)
        users_table = metadata.tables.get("processed_users")

        if users_table is None:
            raise Exception("User table does not exist")

        with engine.connect() as conn:
            # Use .mappings() to get dict-like access
            select_stmt = select(users_table).where(users_table.c.email == email)
            user_row = conn.execute(select_stmt).mappings().first()

            if user_row is None:
                return {
                    "status": "error",
                    "message": "Invalid email or password"
                }

            # Check password
            if not bcrypt.checkpw(password.encode('utf-8'), user_row['password'].encode('utf-8')):
                return {
                    "status": "error",
                    "message": "Invalid email or password"
                }
            
            user_data = dict(user_row)
            # Generate JWT token
            token = generate_jwt_token({
                "user_id": user_data['user_id'],
                "fullName": user_data['fullName'],
            })

            # Convert to dict and remove password
           
            user_data.pop('password', None)
            user_data.pop('user_id', None) 
            user_data.pop('train', None)  
      

            

            return {
                "status": "success",
                "message": "Login successful",
                "token": token,
                "user": user_data
            }

    except Exception as e:
        print("Error in login:", e)
        return {
            "status": "error",
            "message": str(e)
        }


def userUpdate(user_id: str, **fields):
    try:
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)
        users_table = metadata.tables.get("processed_users")

        if users_table is None:
            return {
                "status": "error",
                "message": "User table does not exist"
            }

        # Hash password if provided
        if "password" in fields:
            fields["password"] = bcrypt.hashpw(fields["password"].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        with engine.connect() as conn:
            # Check if user exists
            select_stmt = select(users_table).where(users_table.c.user_id == user_id)
            user = conn.execute(select_stmt).first()
            if user is None:
                return {
                    "status": "error",
                    "message": "User not found"
                }

            # Update user with provided fields
            update_stmt = (
                update(users_table)
                .where(users_table.c.user_id == user_id)
                .values(**fields)
            )
            conn.execute(update_stmt)
            conn.commit()

            # Fetch updated user (excluding password)
            updated_user_row = conn.execute(select(users_table).where(users_table.c.user_id == user_id)).mappings().first()
            updated_user = dict(updated_user_row)
            updated_user.pop("password", None)

            return {
                "status": "success",
                "message": "User updated successfully",
                "user": updated_user
            }

    except Exception as e:
        print("Error in userUpdate:", e)
        return {
            "status": "error",
            "message": str(e)
        }
    

def getUserDetails(user_id: str):
    try:
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)
        users_table = metadata.tables.get("processed_users")

        if users_table is None:
            return {
                "status": "error",
                "message": "User table does not exist"
            }

        with engine.connect() as conn:
            # Query the user by user_id
            select_stmt = select(users_table).where(users_table.c.user_id == user_id)
            user_row = conn.execute(select_stmt).mappings().first()

            if user_row is None:
                return {
                    "status": "error",
                    "message": "User not found"
                }

            user = dict(user_row)
            filtered_user = {
                "fullName": user.get("fullName"),
                "email": user.get("email"),
                "expertise_level":user.get("expertise_level"),
                "years_of_experience":user.get("years_of_experience"),
                "familiar_technologies":user.get("familiar_technologies"),
            }

            return {
                "status": "success",
                "message": "User fetched successfully",
                "user": filtered_user
            }

    except Exception as e:
        print("Error in Get user:", e)
        return {
            "status": "error",
            "message": str(e)
        }
