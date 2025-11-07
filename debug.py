#!/usr/bin/env python3
"""
Quick Transcript Debug Tool
快速检查transcript问题
"""

import sqlite3
import json
import sys

def quick_check(db_path="negotiations.db"):
    """快速检查最近的session"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 获取最近的一个有多轮对话的session
    c.execute("""
        SELECT session_id, student_name, current_round, transcript, ai_memory, ai_plan
        FROM negotiation_sessions 
        WHERE current_round > 3
        ORDER BY updated_at DESC 
        LIMIT 1
    """)
    
    session = c.fetchone()
    if not session:
        print("❌ No sessions with more than 3 rounds found")
        return
    
    print(f"🔍 Checking Session: {session['session_id']}")
    print(f"   Student: {session['student_name']}, Round: {session['current_round']}")
    print()
    
    # 检查transcript
    try:
        transcript = json.loads(session['transcript']) if session['transcript'] else []
        print(f"📋 TRANSCRIPT ({len(transcript)} messages):")
        print("-" * 50)
        
        if transcript:
            for i, msg in enumerate(transcript):
                # 显示前50个字符
                preview = msg[:80] + "..." if len(msg) > 80 else msg
                print(f"[{i+1:2d}] {preview}")
        else:
            print("❌ EMPTY TRANSCRIPT!")
            
        print()
        
        # 检查history构建
        history = "\n\n".join(transcript)
        print(f"📝 CONSTRUCTED HISTORY:")
        print(f"   Length: {len(history)} characters")
        print(f"   Preview: {history[:200]}...")
        print()
        
        # 检查memory
        print(f"🧠 MEMORY STATUS:")
        if session['ai_memory']:
            print(f"   ✅ Memory exists ({len(session['ai_memory'])} chars)")
        else:
            print(f"   ❌ No memory")
            
        # 检查plan
        print(f"📋 PLAN STATUS:")
        if session['ai_plan']:
            print(f"   ✅ Plan exists ({len(session['ai_plan'])} chars)")
        else:
            print(f"   ❌ No plan")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    conn.close()

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "negotiations.db"
    quick_check(db_path)