#!/usr/bin/env python3
"""
Negotiation Debug Tool
用于诊断transcript和memory问题的调试工具

使用方法：
1. 将你的数据库文件下载并放在同一目录
2. 运行: python debug_negotiation.py [session_id]
"""

import sqlite3
import json
import sys
from datetime import datetime

def analyze_session(db_path, session_id):
    """分析特定的negotiation session"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 获取session数据
    c.execute("""
        SELECT * FROM negotiation_sessions WHERE session_id = ?
    """, (session_id,))
    
    session = c.fetchone()
    if not session:
        print(f"❌ Session {session_id} not found")
        return
        
    print("="*80)
    print(f"🔍 ANALYZING SESSION: {session_id}")
    print("="*80)
    
    # 基本信息
    print(f"📋 Basic Info:")
    print(f"   Student: {session['student_name']} ({session['student_id']})")
    print(f"   Scenario: {session['scenario_name']}")
    print(f"   Student Role: {session['student_role']}")
    print(f"   AI Role: {session['ai_role']}")
    print(f"   AI Model: {session['ai_model']}")
    print(f"   Student Goes First: {session['student_goes_first']}")
    print(f"   Use Memory: {session['use_memory']}")
    print(f"   Use Plan: {session['use_plan']}")
    print(f"   Current Round: {session['current_round']}")
    print(f"   Total Rounds: {session['total_rounds']}")
    print(f"   Status: {session['status']}")
    print(f"   Created: {session['created_at']}")
    print(f"   Updated: {session['updated_at']}")
    print()
    
    # Transcript分析
    print("💬 TRANSCRIPT ANALYSIS:")
    print("-" * 40)
    
    try:
        transcript = json.loads(session['transcript']) if session['transcript'] else []
        print(f"   Total Messages: {len(transcript)}")
        print(f"   Raw Transcript JSON Length: {len(session['transcript']) if session['transcript'] else 0} chars")
        print()
        
        if transcript:
            print("   Messages:")
            for i, msg in enumerate(transcript):
                # 截断长消息
                display_msg = msg[:100] + "..." if len(msg) > 100 else msg
                print(f"   [{i+1:2d}] {display_msg}")
            print()
            
            # 分析消息模式
            student_msgs = [msg for msg in transcript if msg.startswith(session['student_role']) or "Student:" in msg or "👤" in msg]
            ai_msgs = [msg for msg in transcript if msg.startswith(session['ai_role']) or "AI:" in msg or "🤖" in msg]
            
            print(f"   Student Messages: {len(student_msgs)}")
            print(f"   AI Messages: {len(ai_msgs)}")
            print()
            
        else:
            print("   ❌ No transcript found!")
            
    except Exception as e:
        print(f"   ❌ Error parsing transcript: {e}")
        print(f"   Raw transcript: {session['transcript'][:200]}...")
    
    print()
    
    # Memory分析
    print("🧠 MEMORY ANALYSIS:")
    print("-" * 40)
    
    if session['ai_memory']:
        memory_length = len(session['ai_memory'])
        print(f"   Memory Length: {memory_length} chars")
        print(f"   Memory Content Preview:")
        print(f"   {session['ai_memory'][:300]}...")
        if memory_length > 300:
            print(f"   ... (truncated, full length: {memory_length} chars)")
    else:
        print("   ❌ No AI memory found")
    print()
    
    # Plan分析
    print("📋 PLAN ANALYSIS:")
    print("-" * 40)
    
    if session['ai_plan']:
        plan_length = len(session['ai_plan'])
        print(f"   Plan Length: {plan_length} chars")
        print(f"   Plan Content Preview:")
        print(f"   {session['ai_plan'][:300]}...")
        if plan_length > 300:
            print(f"   ... (truncated, full length: {plan_length} chars)")
    else:
        print("   ❌ No AI plan found")
    print()
    
    # Deal分析
    print("🤝 DEAL ANALYSIS:")
    print("-" * 40)
    print(f"   Deal Reached: {session['deal_reached']}")
    print(f"   Deal Failed: {session['deal_failed']}")
    
    if session['student_deal_json']:
        try:
            student_deal = json.loads(session['student_deal_json'])
            print(f"   Student Deal: {json.dumps(student_deal, indent=2)}")
        except:
            print(f"   Student Deal (raw): {session['student_deal_json']}")
    
    if session['ai_deal_json']:
        try:
            ai_deal = json.loads(session['ai_deal_json'])
            print(f"   AI Deal: {json.dumps(ai_deal, indent=2)}")
        except:
            print(f"   AI Deal (raw): {session['ai_deal_json']}")
    
    print()
    
    # 问题诊断
    print("🔧 PROBLEM DIAGNOSIS:")
    print("-" * 40)
    
    issues = []
    
    # 检查transcript问题
    if not transcript:
        issues.append("❌ CRITICAL: No transcript found - messages are not being saved!")
    elif len(transcript) < 2:
        issues.append("⚠️  WARNING: Very short transcript - only 1 message")
    
    # 检查memory问题
    if session['use_memory'] and not session['ai_memory']:
        issues.append("❌ MEMORY: Memory is enabled but empty")
    elif session['use_memory'] and session['ai_memory']:
        issues.append("✅ MEMORY: Memory is working")
        
    # 检查plan问题
    if session['use_plan'] and not session['ai_plan']:
        issues.append("❌ PLAN: Plan is enabled but empty")
    elif session['use_plan'] and session['ai_plan']:
        issues.append("✅ PLAN: Plan is working")
    
    # 检查轮次问题
    expected_messages = session['current_round'] * 2 if session['student_goes_first'] else (session['current_round'] * 2) - 1
    actual_messages = len(transcript) if transcript else 0
    
    if actual_messages < expected_messages - 1:  # 允许1条消息的误差
        issues.append(f"❌ ROUND SYNC: Expected ~{expected_messages} messages for round {session['current_round']}, got {actual_messages}")
    
    # 分析AI重复性
    if transcript:
        ai_messages = [msg for msg in transcript if "AI:" in msg or session['ai_role'] in msg]
        if len(ai_messages) >= 2:
            # 简单检查是否有重复内容
            if len(set(ai_messages)) < len(ai_messages) * 0.8:
                issues.append("⚠️  WARNING: AI responses seem repetitive")
    
    if not issues:
        issues.append("✅ No obvious issues found")
    
    for issue in issues:
        print(f"   {issue}")
    
    print()
    conn.close()

def list_recent_sessions(db_path, limit=10):
    """列出最近的sessions"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("""
        SELECT session_id, student_name, scenario_name, current_round, status, created_at
        FROM negotiation_sessions 
        ORDER BY created_at DESC 
        LIMIT ?
    """, (limit,))
    
    sessions = c.fetchall()
    
    print("📋 RECENT SESSIONS:")
    print("-" * 60)
    print(f"{'Session ID':<36} {'Student':<15} {'Scenario':<15} {'Round':<5} {'Status':<10}")
    print("-" * 60)
    
    for session in sessions:
        session_id_short = session['session_id'][:8] + "..."
        print(f"{session_id_short:<36} {session['student_name'][:15]:<15} {session['scenario_name'][:15]:<15} {session['current_round']:<5} {session['status']:<10}")
    
    print()
    conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_negotiation.py <command> [args]")
        print("Commands:")
        print("  list [db_path]           - List recent sessions")
        print("  analyze <session_id> [db_path] - Analyze specific session")
        print("  db_path defaults to 'negotiations.db'")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        db_path = sys.argv[2] if len(sys.argv) > 2 else "negotiations.db"
        list_recent_sessions(db_path)
        
    elif command == "analyze":
        if len(sys.argv) < 3:
            print("Error: session_id required for analyze command")
            return
        session_id = sys.argv[2]
        db_path = sys.argv[3] if len(sys.argv) > 3 else "negotiations.db"
        analyze_session(db_path, session_id)
        
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()