#!/usr/bin/env python3
"""
Chat History Management Script
Use this to view, manage, and backup your chat history
"""

import json
import os
from pathlib import Path
from datetime import datetime

CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    """Load chat history from file"""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_chat_history(data, filename=None):
    """Save chat history to file"""
    if filename is None:
        filename = CHAT_HISTORY_FILE
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved to {filename}")

def backup_chat_history():
    """Create a backup of chat history"""
    if not os.path.exists(CHAT_HISTORY_FILE):
        print("❌ No chat history file found to backup")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"chat_history_backup_{timestamp}.json"
    
    data = load_chat_history()
    save_chat_history(data, backup_file)
    print(f"💾 Backup created: {backup_file}")

def view_chat_history():
    """View all chat sessions"""
    data = load_chat_history()
    
    if not data:
        print("📝 No chat history found")
        return
    
    print(f"\n📊 Chat History Summary")
    print(f"{'='*50}")
    print(f"Total Sessions: {len(data)}")
    print(f"File: {CHAT_HISTORY_FILE}")
    print(f"Size: {os.path.getsize(CHAT_HISTORY_FILE) / 1024:.1f} KB")
    
    for i, (session_id, messages) in enumerate(data.items(), 1):
        print(f"\n🔹 Session {i}: {session_id[:8]}...")
        print(f"   Messages: {len(messages)}")
        if messages:
            last_msg = messages[-1]["content"][:80]
            print(f"   Last: {last_msg}...")

def view_session(session_id):
    """View a specific session"""
    data = load_chat_history()
    
    if session_id not in data:
        print(f"❌ Session {session_id} not found")
        return
    
    messages = data[session_id]
    print(f"\n💬 Session: {session_id}")
    print(f"{'='*50}")
    
    for i, msg in enumerate(messages, 1):
        role = "👤 You" if msg["role"] == "user" else "🤖 AI"
        content = msg["content"]
        print(f"\n{role} (Message {i}):")
        print(f"   {content}")

def delete_session(session_id):
    """Delete a specific session"""
    data = load_chat_history()
    
    if session_id not in data:
        print(f"❌ Session {session_id} not found")
        return
    
    del data[session_id]
    save_chat_history(data)
    print(f"✅ Session {session_id} deleted")

def clear_all_history():
    """Clear all chat history"""
    confirm = input("⚠️  Are you sure you want to delete ALL chat history? (yes/no): ")
    if confirm.lower() == 'yes':
        save_chat_history({})
        print("✅ All chat history cleared")
    else:
        print("❌ Operation cancelled")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*50)
        print("🗂️  Chat History Manager")
        print("="*50)
        print("1. View all sessions")
        print("2. View specific session")
        print("3. Delete specific session")
        print("4. Clear all history")
        print("5. Create backup")
        print("6. Exit")
        print("-"*50)
        
        choice = input("Choose an option (1-6): ").strip()
        
        if choice == "1":
            view_chat_history()
        elif choice == "2":
            session_id = input("Enter session ID: ").strip()
            view_session(session_id)
        elif choice == "3":
            session_id = input("Enter session ID to delete: ").strip()
            delete_session(session_id)
        elif choice == "4":
            clear_all_history()
        elif choice == "5":
            backup_chat_history()
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
