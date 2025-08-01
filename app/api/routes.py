from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import sqlite3
from typing import List, Optional
from datetime import datetime

router = APIRouter()

# Pydantic models
class ItemCreate(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

class ItemUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    price: float
    created_at: str

# Database helper functions
def get_db_connection():
    """Get SQLite database connection"""
    return sqlite3.connect("app.db")

def init_items_table():
    """Initialize items table if it doesn't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Initialize items table
init_items_table()

# API routes
@router.get("/items", response_model=List[ItemResponse])
async def get_items():
    """Get all items"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description, price, created_at FROM items")
        items = cursor.fetchall()
        conn.close()
        
        return [
            ItemResponse(
                id=item[0],
                name=item[1],
                description=item[2],
                price=item[3],
                created_at=item[4]
            )
            for item in items
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch items: {str(e)}")

@router.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """Get a specific item by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description, price, created_at FROM items WHERE id = ?", (item_id,))
        item = cursor.fetchone()
        conn.close()
        
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        
        return ItemResponse(
            id=item[0],
            name=item[1],
            description=item[2],
            price=item[3],
            created_at=item[4]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch item: {str(e)}")

@router.post("/items", response_model=ItemResponse)
async def create_item(item: ItemCreate):
    """Create a new item"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO items (name, description, price) VALUES (?, ?, ?)",
            (item.name, item.description, item.price)
        )
        conn.commit()
        
        # Get the created item
        cursor.execute("SELECT id, name, description, price, created_at FROM items WHERE id = ?", 
                      (cursor.lastrowid,))
        item_data = cursor.fetchone()
        conn.close()
        
        return ItemResponse(
            id=item_data[0],
            name=item_data[1],
            description=item_data[2],
            price=item_data[3],
            created_at=item_data[4]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create item: {str(e)}")

@router.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: int, item: ItemUpdate):
    """Update an existing item"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if item exists
        cursor.execute("SELECT id FROM items WHERE id = ?", (item_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Build update query dynamically
        update_fields = []
        values = []
        if item.name is not None:
            update_fields.append("name = ?")
            values.append(item.name)
        if item.description is not None:
            update_fields.append("description = ?")
            values.append(item.description)
        if item.price is not None:
            update_fields.append("price = ?")
            values.append(item.price)
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        values.append(item_id)
        query = f"UPDATE items SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, values)
        conn.commit()
        
        # Get the updated item
        cursor.execute("SELECT id, name, description, price, created_at FROM items WHERE id = ?", (item_id,))
        item_data = cursor.fetchone()
        conn.close()
        
        return ItemResponse(
            id=item_data[0],
            name=item_data[1],
            description=item_data[2],
            price=item_data[3],
            created_at=item_data[4]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update item: {str(e)}")

@router.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """Delete an item"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if item exists
        cursor.execute("SELECT id FROM items WHERE id = ?", (item_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Item not found")
        
        cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))
        conn.commit()
        conn.close()
        
        return {"message": "Item deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete item: {str(e)}")

@router.get("/stats")
async def get_stats():
    """Get basic API statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Count items
        cursor.execute("SELECT COUNT(*) FROM items")
        item_count = cursor.fetchone()[0]
        
        # Count users
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_items": item_count,
            "total_users": user_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
