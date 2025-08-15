#!/usr/bin/env python3
"""
Simple Chat History Manager - Non-interactive version
Use this to view and manage your chat history
"""

import json
import os
import sys
from datetime import datetime

CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    """Load chat history from file"""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

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

def backup_chat_history():
    """Create a backup of chat history"""
    if not os.path.exists(CHAT_HISTORY_FILE):
        print("❌ No chat history file found to backup")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"chat_history_backup_{timestamp}.json"
    
    data = load_chat_history()
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Backup created: {backup_file}")

def main():
    """Main function with command line arguments"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python simple_chat_manager.py view                    # View all sessions")
        print("  python simple_chat_manager.py session <session_id>    # View specific session")
        print("  python simple_chat_manager.py backup                  # Create backup")
        print("  python simple_chat_manager.py status                 # Show status")
        return
    
    command = sys.argv[1].lower()
    
    if command == "view":
        view_chat_history()
    elif command == "session" and len(sys.argv) > 2:
        session_id = sys.argv[2]
        view_session(session_id)
    elif command == "backup":
        backup_chat_history()
    elif command == "status":
        data = load_chat_history()
        print(f"📊 Chat History Status")
        print(f"{'='*30}")
        print(f"File: {CHAT_HISTORY_FILE}")
        print(f"Exists: {os.path.exists(CHAT_HISTORY_FILE)}")
        if os.path.exists(CHAT_HISTORY_FILE):
            print(f"Size: {os.path.getsize(CHAT_HISTORY_FILE) / 1024:.1f} KB")
            print(f"Sessions: {len(data)}")
        else:
            print("Size: N/A")
            print("Sessions: 0")
    else:
        print("❌ Invalid command. Use 'view', 'session <id>', 'backup', or 'status'")

if __name__ == "__main__":
    main()
