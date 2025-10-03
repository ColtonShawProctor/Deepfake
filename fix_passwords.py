#!/usr/bin/env python3
"""
Script to migrate existing SHA-256 password hashes to bcrypt
"""
import sqlite3
import hashlib
from passlib.context import CryptContext

# Create password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def migrate_passwords():
    """Migrate existing SHA-256 passwords to bcrypt"""
    conn = sqlite3.connect('deepfake.db')
    cursor = conn.cursor()
    
    # Get all users
    cursor.execute("SELECT id, email, password_hash FROM users")
    users = cursor.fetchall()
    
    print(f"Found {len(users)} users to migrate")
    
    for user_id, email, password_hash in users:
        print(f"\nProcessing user: {email}")
        print(f"Current hash: {password_hash}")
        print(f"Hash length: {len(password_hash)}")
        
        # Check if this is already a bcrypt hash
        if password_hash.startswith('$2b$'):
            print("Already bcrypt hash, skipping...")
            continue
            
        # This is likely a SHA-256 hash, we need to know the original password
        # For now, let's set a default password and let users reset it
        default_password = "changeme123"
        
        try:
            # Create new bcrypt hash
            new_hash = pwd_context.hash(default_password)
            print(f"New bcrypt hash: {new_hash}")
            
            # Update the database
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_hash, user_id)
            )
            print(f"Updated user {email} with new hash")
            
        except Exception as e:
            print(f"Error updating user {email}: {e}")
    
    # Commit changes
    conn.commit()
    print(f"\nMigration completed. All users now have password: {default_password}")
    print("Users should change their passwords after first login!")
    
    conn.close()

if __name__ == "__main__":
    migrate_passwords()

