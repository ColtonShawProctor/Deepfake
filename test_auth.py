#!/usr/bin/env python3
"""
Test script to debug authentication issues
"""
import sqlite3
from passlib.context import CryptContext

# Create password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def test_password_verification():
    """Test password verification with existing database"""
    conn = sqlite3.connect('deepfake.db')
    cursor = conn.cursor()
    
    # Get user data
    cursor.execute("SELECT id, email, password_hash FROM users LIMIT 1")
    user = cursor.fetchone()
    
    if user:
        user_id, email, password_hash = user
        print(f"User: {email}")
        print(f"Password hash: {password_hash}")
        print(f"Hash length: {len(password_hash)}")
        
        # Try to verify with test password
        test_password = "test123"
        try:
            is_valid = pwd_context.verify(test_password, password_hash)
            print(f"Password verification result: {is_valid}")
        except Exception as e:
            print(f"Password verification error: {e}")
            
            # Try to hash a new password
            try:
                new_hash = pwd_context.hash(test_password)
                print(f"New hash for 'test123': {new_hash}")
                print(f"New hash length: {len(new_hash)}")
            except Exception as e2:
                print(f"Hash creation error: {e2}")
    
    conn.close()

if __name__ == "__main__":
    test_password_verification()

