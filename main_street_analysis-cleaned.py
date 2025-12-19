# 🏘️ Main Street Negotiation - Comprehensive Analysis Dashboard

Complete analysis of Main Street real estate negotiations (Fred Starr vs Rosalind Cain)

**Analysis Sections:**
1. 📋 All Negotiation Records Overview
2. 👥 Student Demographics & Characteristics Analysis
3. 💰 Configuration Comparison: Base vs M vs M+P (Price Analysis)
4. 🧠 Memory (M) Evolution Tracking
5. 🎯 Planning (P) Strategy Tracking
6. 💬 Full Transcript Viewer
7. 📊 Performance Metrics Dashboard

---

## Setup

import sqlite3
import pandas as pd
import json
import numpy as np
from datetime import datetime
from IPython.display import display, HTML, Markdown
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DB_PATH = "negotiations_mainst.db"  #
SCENARIO_FILTER = "Main_Street"  # Focus on Main Street scenario

conn = sqlite3.connect(DB_PATH)
print(f"✅ Connected to: {DB_PATH}")
print(f"🎯 Analyzing scenario: {SCENARIO_FILTER}")

# 在 df_all = pd.read_sql_query(query_all, conn) 之前添加：

# ==================== PANDAS DISPLAY SETTINGS ====================
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整宽度
pd.set_option('display.max_colwidth', 50)  # 列内容最大宽度

# Helper Functions
def safe_json_load(json_str):
    if not json_str or json_str == 'null':
        return None
    try:
        return json.loads(json_str)
    except:
        return None

def calculate_duration(start, end):
    try:
        return (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 60
    except:
        return None

def get_config_label(use_memory, use_plan):
    if use_memory and use_plan:
        return 'M+P'
    elif use_memory:
        return 'M'
    elif use_plan:
        return 'P'
    else:
        return 'Base'

def print_header(title, char='=', width=100):
    print("\n" + char * width)
    print(f"{title.center(width)}")
    print(char * width + "\n")

def print_subheader(title, char='-', width=80):
    print("\n" + char * width)
    print(title)
    print(char * width)

print("✅ Helper functions loaded")

---
## 1. 📋 All Negotiation Records Overview

Complete list of all Main Street negotiations

# Load all Main Street sessions
query_all = f"""
SELECT 
    session_id,
    student_role,
    ai_model,
    student_goes_first,
    use_memory,
    use_plan,
    total_rounds,
    deal_reached,
    deal_failed,
    status,
    created_at,
    updated_at,
    major,
    gender,
    negotiation_experience,
    transcript,
    student_deal_json,
    ai_deal_json
FROM negotiation_sessions
WHERE scenario_name = '{SCENARIO_FILTER}'
ORDER BY created_at DESC
"""

df_all = pd.read_sql_query(query_all, conn)

# Add derived columns
df_all['Config'] = df_all.apply(lambda r: get_config_label(r['use_memory'], r['use_plan']), axis=1)
df_all['Duration (min)'] = df_all.apply(lambda r: calculate_duration(r['created_at'], r['updated_at']), axis=1)
df_all['Outcome'] = df_all.apply(lambda r: 
    '✅ Deal' if r['deal_reached'] 
    else '❌ Failed' if r['deal_failed'] 
    else '⏸️ Incomplete', axis=1)

df_all['Role'] = df_all['student_role'].map({
    'side1': '🏢 Buyer (Fred)',
    'side2': '🏘️ Seller (Rosalind)'
})

print_header("📋 ALL MAIN STREET NEGOTIATIONS")
print(f"Total Sessions: {len(df_all)}")
print(f"Successful Deals: {df_all['deal_reached'].sum()}")
print(f"Failed Negotiations: {df_all['deal_failed'].sum()}")
print(f"Incomplete: {(~df_all['deal_reached'] & ~df_all['deal_failed']).sum()}")

print(f"\n📊 Configuration Breakdown:")
print(df_all['Config'].value_counts().to_string())

print(f"\n🎭 Role Distribution:")
print(df_all['Role'].value_counts().to_string())

# Display table
display_df = df_all[[
    'session_id', 'Role', 'Config', 'Outcome',
    'major', 'gender', 'negotiation_experience',  
    'Duration (min)', 'created_at'
]].copy()

display_df['session_id'] = display_df['session_id'].str[:8]
display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
display_df['Duration (min)'] = display_df['Duration (min)'].round(1)


display_df.columns = ['Session', 'Role', 'Config', 'Outcome', 'Major', 'Gender', 'Exp', 'Duration', 'Created']

print("\n" + "="*150) 
display(display_df)

# ==================== DATA FILTERING ====================
# 1. Filter by time: Only keep sessions after 2025-12-10 14:36
cutoff_time = pd.to_datetime('2025-12-10 14:36')
df_all = df_all[pd.to_datetime(df_all['created_at']) >= cutoff_time].copy()

# 2. Filter by major: Exclude SDS/sds
df_all = df_all[
    ~df_all['major'].str.lower().isin(['sds']) | df_all['major'].isna()
].copy()

# Recreate derived columns after filtering
df_all['Config'] = df_all.apply(lambda r: get_config_label(r['use_memory'], r['use_plan']), axis=1)
df_all['Duration (min)'] = df_all.apply(lambda r: calculate_duration(r['created_at'], r['updated_at']), axis=1)
df_all['Outcome'] = df_all.apply(lambda r: 
    '✅ Deal' if r['deal_reached'] 
    else '❌ Failed' if r['deal_failed'] 
    else '⏸️ Incomplete', axis=1)
df_all['Role'] = df_all['student_role'].map({
    'side1': '🏢 Buyer (Fred)',
    'side2': '🏘️ Seller (Rosalind)'
})

# Update display dataframe with filtered data
display_df = df_all[[
    'session_id', 'Role', 'Config', 'Outcome',
    'major', 'gender', 'negotiation_experience',  
    'Duration (min)', 'created_at'
]].copy()

display_df['session_id'] = display_df['session_id'].str[:8]
display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
display_df['Duration (min)'] = display_df['Duration (min)'].round(1)
display_df.columns = ['Session', 'Role', 'Config', 'Outcome', 'Major', 'Gender', 'Exp', 'Duration', 'Created']

# Print filtered statistics
print("\n" + "="*150)
print("🔍 FILTERED DATA (After 2025-12-10 14:36, Excluding SDS/sds majors)")
print("="*150)
print(f"Total Sessions: {len(df_all)}")
print(f"Successful Deals: {df_all['deal_reached'].sum()}")
print(f"Failed Negotiations: {df_all['deal_failed'].sum()}")
print(f"Incomplete: {(~df_all['deal_reached'] & ~df_all['deal_failed']).sum()}")

print(f"\n📊 Configuration Breakdown:")
print(df_all['Config'].value_counts().to_string())

print(f"\n🎭 Role Distribution:")
print(df_all['Role'].value_counts().to_string())

print("\n" + "="*150) 
display(display_df)

---
## 2. 👥 Student Demographics & Characteristics Analysis

print_header("👥 STUDENT DEMOGRAPHICS & CHARACTERISTICS")

# Filter out rows with no demographic data
df_demo = df_all.copy()

# Major Analysis
print_subheader("📚 Academic Major Distribution")
if df_demo['major'].notna().sum() > 0:
    major_counts = df_demo['major'].value_counts()
    print(f"Total with major data: {df_demo['major'].notna().sum()}")
    print("\nMajor breakdown:")
    for major, count in major_counts.items():
        pct = count / len(df_demo) * 100
        print(f"  {major}: {count} ({pct:.1f}%)")
else:
    print("No major data available")

# Gender Analysis
print_subheader("⚧ Gender Distribution")
if df_demo['gender'].notna().sum() > 0:
    gender_counts = df_demo['gender'].value_counts()
    print(f"Total with gender data: {df_demo['gender'].notna().sum()}")
    print("\nGender breakdown:")
    for gender, count in gender_counts.items():
        pct = count / len(df_demo) * 100
        print(f"  {gender}: {count} ({pct:.1f}%)")
else:
    print("No gender data available")

# Experience Analysis
print_subheader("💼 Negotiation Experience Distribution")
if df_demo['negotiation_experience'].notna().sum() > 0:
    exp_counts = df_demo['negotiation_experience'].value_counts()
    print(f"Total with experience data: {df_demo['negotiation_experience'].notna().sum()}")
    print("\nExperience breakdown:")
    for exp, count in exp_counts.items():
        pct = count / len(df_demo) * 100
        print(f"  {exp}: {count} ({pct:.1f}%)")
else:
    print("No experience data available")

# Success rate by demographics
print_subheader("📈 Success Rates by Demographics")

if df_demo['major'].notna().sum() > 0:
    print("\nBy Major:")
    major_success = df_demo.groupby('major')['deal_reached'].agg(['sum', 'count', 'mean'])
    major_success.columns = ['Deals', 'Total', 'Success Rate']
    major_success['Success Rate'] = (major_success['Success Rate'] * 100).round(1)
    print(major_success.to_string())

if df_demo['gender'].notna().sum() > 0:
    print("\nBy Gender:")
    gender_success = df_demo.groupby('gender')['deal_reached'].agg(['sum', 'count', 'mean'])
    gender_success.columns = ['Deals', 'Total', 'Success Rate']
    gender_success['Success Rate'] = (gender_success['Success Rate'] * 100).round(1)
    print(gender_success.to_string())

if df_demo['negotiation_experience'].notna().sum() > 0:
    print("\nBy Negotiation Experience:")
    exp_success = df_demo.groupby('negotiation_experience')['deal_reached'].agg(['sum', 'count', 'mean'])
    exp_success.columns = ['Deals', 'Total', 'Success Rate']
    exp_success['Success Rate'] = (exp_success['Success Rate'] * 100).round(1)
    print(exp_success.to_string())

---
## 3. 💰 Configuration Comparison: Base vs M vs M+P

Analysis of deal prices across different AI configurations

# Load successful deals with price data
query_deals = f"""
SELECT 
    session_id,
    student_role,
    ai_role,
    use_memory,
    use_plan,
    student_deal_json,
    ai_deal_json,
    transcript,
    created_at,
    updated_at
FROM negotiation_sessions
WHERE scenario_name = '{SCENARIO_FILTER}' AND deal_reached = 1
ORDER BY created_at DESC
"""

df_deals = pd.read_sql_query(query_deals, conn)

# Parse deal JSONs and extract prices
def extract_price(deal_json_str):
    deal = safe_json_load(deal_json_str)
    if deal and 'final_price' in deal:
        return deal['final_price']
    return None

df_deals['student_price'] = df_deals['student_deal_json'].apply(extract_price)
df_deals['ai_price'] = df_deals['ai_deal_json'].apply(extract_price)
df_deals['config'] = df_deals.apply(lambda r: get_config_label(r['use_memory'], r['use_plan']), axis=1)
df_deals['duration'] = df_deals.apply(lambda r: calculate_duration(r['created_at'], r['updated_at']), axis=1)

# Count rounds
def count_rounds(transcript_json):
    transcript = safe_json_load(transcript_json)
    return len(transcript) // 2 if transcript else 0

df_deals['rounds'] = df_deals['transcript'].apply(count_rounds)

# Map roles
df_deals['student_role_label'] = df_deals['student_role'].map({
    'side1': 'Buyer (Fred)',
    'side2': 'Seller (Rosalind)'
})

df_deals['ai_role_label'] = df_deals['ai_role'].map({
    'side1': 'Buyer (Fred)',
    'side2': 'Seller (Rosalind)'
})

print_header("💰 DEAL PRICE ANALYSIS BY CONFIGURATION")

if len(df_deals) == 0:
    print("❌ No successful deals found")
else:
    print(f"Total Successful Deals: {len(df_deals)}\n")
    
    # Overall statistics
    print_subheader("📊 Overall Price Statistics")
    print(f"Average Final Price: ${df_deals['student_price'].mean():,.0f}")
    print(f"Median Final Price: ${df_deals['student_price'].median():,.0f}")
    print(f"Min Price: ${df_deals['student_price'].min():,.0f}")
    print(f"Max Price: ${df_deals['student_price'].max():,.0f}")
    print(f"Std Dev: ${df_deals['student_price'].std():,.0f}")
    
    # Price by configuration
    print_subheader("🔧 Average Price by Configuration")
    
    config_stats = df_deals.groupby('config').agg({
        'student_price': ['count', 'mean', 'median', 'std', 'min', 'max']
    }).round(0)
    
    print("\nConfiguration Summary:")
    print("="*80)
    for config in ['Base', 'M', 'P', 'M+P']:
        if config in config_stats.index:
            stats = config_stats.loc[config, 'student_price']
            print(f"\n{config}:")
            print(f"  Count: {int(stats['count'])} deals")
            print(f"  Average: ${stats['mean']:,.0f}")
            print(f"  Median:  ${stats['median']:,.0f}")
            print(f"  Range:   ${stats['min']:,.0f} - ${stats['max']:,.0f}")
            if stats['count'] > 1:
                print(f"  Std Dev: ${stats['std']:,.0f}")
    
    # Price by role and configuration
    print_subheader("🎭 Average Price by Role & Configuration")
    
    role_config_stats = df_deals.groupby(['student_role_label', 'config'])['student_price'].agg(['count', 'mean', 'median']).round(0)
    
    print("\nWhen Student is BUYER (Fred Starr):")
    print("  Goal: Minimize price (lower is better)")
    print("  BATNA: $675,000 (will walk away if price > $675k)")
    print("-"*80)
    if 'Buyer (Fred)' in role_config_stats.index.get_level_values(0):
        buyer_stats = role_config_stats.loc['Buyer (Fred)']
        for config in buyer_stats.index:
            stats = buyer_stats.loc[config]
            print(f"  {config}: ${stats['mean']:,.0f} avg (n={int(stats['count'])})")
    
    print("\nWhen Student is SELLER (Rosalind Cain):")
    print("  Goal: Maximize price (higher is better)")
    print("  BATNA: $475,000 (will walk away if price < $475k)")
    print("-"*80)
    if 'Seller (Rosalind)' in role_config_stats.index.get_level_values(0):
        seller_stats = role_config_stats.loc['Seller (Rosalind)']
        for config in seller_stats.index:
            stats = seller_stats.loc[config]
            print(f"  {config}: ${stats['mean']:,.0f} avg (n={int(stats['count'])})")

    # Detailed deal list
    print_subheader("📋 All Deals - Detailed List")

    deal_display = df_deals[[
        'session_id', 'student_role_label', 'config', 
        'student_price', 'rounds', 'duration', 'created_at'  # ← 添加 created_at
    ]].copy()

    deal_display['session_id'] = deal_display['session_id'].str[:8]
    deal_display['student_price'] = deal_display['student_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    deal_display['duration'] = deal_display['duration'].round(1)
    deal_display['created_at'] = pd.to_datetime(deal_display['created_at']).dt.strftime('%m-%d %H:%M')  # ← 格式化时间

    deal_display.columns = ['Session', 'Student Role', 'Config', 'Final Price', 'Rounds', 'Duration (min)', 'Created']  # ← 添加列名

    display(deal_display)
    


### 3.1 Price Comparison Summary Table

# Create comprehensive comparison table
if len(df_deals) > 0:
    print("\n" + "="*100)
    print("COMPREHENSIVE PRICE COMPARISON: BASE vs M vs M+P".center(100))
    print("="*100 + "\n")
    
    comparison_data = []
    
    for config in ['Base', 'M', 'P', 'M+P']:
        config_data = df_deals[df_deals['config'] == config]
        
        if len(config_data) > 0:
            comparison_data.append({
                'Configuration': config,
                'N Deals': len(config_data),
                'Avg Price': f"${config_data['student_price'].mean():,.0f}",
                'Median Price': f"${config_data['student_price'].median():,.0f}",
                'Min Price': f"${config_data['student_price'].min():,.0f}",
                'Max Price': f"${config_data['student_price'].max():,.0f}",
                'Avg Rounds': f"{config_data['rounds'].mean():.1f}",
                'Avg Duration': f"{config_data['duration'].mean():.1f} min"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        display(comparison_df)
        
        print("\n💡 Key Insights:")
        print("-"*80)
        
        # Calculate buyer vs seller average for each config
        for config in ['Base', 'M', 'P', 'M+P']:
            config_data = df_deals[df_deals['config'] == config]
            if len(config_data) > 0:
                buyer_avg = config_data[config_data['student_role'] == 'side1']['student_price'].mean()
                seller_avg = config_data[config_data['student_role'] == 'side2']['student_price'].mean()
                
                print(f"\n{config}:")
                if not pd.isna(buyer_avg):
                    print(f"  When student is buyer: ${buyer_avg:,.0f} (student wants lower)")
                if not pd.isna(seller_avg):
                    print(f"  When student is seller: ${seller_avg:,.0f} (student wants higher)")
else:
    print("No deal data available for comparison")

# ==================== DETAILED ANALYSIS: WHO GOES FIRST ====================
if len(df_deals) > 0:
    print("\n" + "="*100)
    print("DETAILED ANALYSIS: IMPACT OF WHO GOES FIRST".center(100))
    print("="*100 + "\n")
    
    # Add student_goes_first column to df_deals if not already there
    if 'student_goes_first' not in df_deals.columns:
        # Need to re-query with student_goes_first
        query_deals_detailed = f"""
        SELECT 
            session_id,
            student_role,
            ai_role,
            use_memory,
            use_plan,
            student_goes_first,
            student_deal_json,
            ai_deal_json,
            transcript,
            created_at,
            updated_at
        FROM negotiation_sessions
        WHERE scenario_name = '{SCENARIO_FILTER}' AND deal_reached = 1
        ORDER BY created_at DESC
        """
        df_deals = pd.read_sql_query(query_deals_detailed, conn)
        
        # Re-apply all transformations
        df_deals['student_price'] = df_deals['student_deal_json'].apply(extract_price)
        df_deals['ai_price'] = df_deals['ai_deal_json'].apply(extract_price)
        df_deals['config'] = df_deals.apply(lambda r: get_config_label(r['use_memory'], r['use_plan']), axis=1)
        df_deals['duration'] = df_deals.apply(lambda r: calculate_duration(r['created_at'], r['updated_at']), axis=1)
        df_deals['rounds'] = df_deals['transcript'].apply(count_rounds)
        df_deals['student_role_label'] = df_deals['student_role'].map({
            'side1': 'Buyer (Fred)',
            'side2': 'Seller (Rosalind)'
        })
        df_deals['ai_role_label'] = df_deals['ai_role'].map({
            'side1': 'Buyer (Fred)',
            'side2': 'Seller (Rosalind)'
        })
    
    # Create detailed breakdown by config and who goes first
    for config in ['Base', 'M', 'P', 'M+P']:
        config_data = df_deals[df_deals['config'] == config]
        
        if len(config_data) > 0:
            print(f"\n{'─'*100}")
            print(f"📊 {config} Configuration")
            print(f"{'─'*100}")
            
            # Student goes first
            student_first = config_data[config_data['student_goes_first'] == True]
            ai_first = config_data[config_data['student_goes_first'] == False]
            
            print(f"\n🎯 When STUDENT goes first (n={len(student_first)}):")
            if len(student_first) > 0:
                print(f"   Average Price: ${student_first['student_price'].mean():,.0f}")
                print(f"   Median Price:  ${student_first['student_price'].median():,.0f}")
                print(f"   Range: ${student_first['student_price'].min():,.0f} - ${student_first['student_price'].max():,.0f}")
                
                # Breakdown by role
                buyer_student_first = student_first[student_first['student_role'] == 'side1']
                seller_student_first = student_first[student_first['student_role'] == 'side2']
                
                if len(buyer_student_first) > 0:
                    print(f"     → As Buyer (n={len(buyer_student_first)}): ${buyer_student_first['student_price'].mean():,.0f} avg")
                if len(seller_student_first) > 0:
                    print(f"     → As Seller (n={len(seller_student_first)}): ${seller_student_first['student_price'].mean():,.0f} avg")
            else:
                print("   No data")
            
            print(f"\n🤖 When AI goes first (n={len(ai_first)}):")
            if len(ai_first) > 0:
                print(f"   Average Price: ${ai_first['student_price'].mean():,.0f}")
                print(f"   Median Price:  ${ai_first['student_price'].median():,.0f}")
                print(f"   Range: ${ai_first['student_price'].min():,.0f} - ${ai_first['student_price'].max():,.0f}")
                
                # Breakdown by role
                buyer_ai_first = ai_first[ai_first['student_role'] == 'side1']
                seller_ai_first = ai_first[ai_first['student_role'] == 'side2']
                
                if len(buyer_ai_first) > 0:
                    print(f"     → As Buyer (n={len(buyer_ai_first)}): ${buyer_ai_first['student_price'].mean():,.0f} avg")
                if len(seller_ai_first) > 0:
                    print(f"     → As Seller (n={len(seller_ai_first)}): ${seller_ai_first['student_price'].mean():,.0f} avg")
            else:
                print("   No data")
            
            # Calculate impact
            if len(student_first) > 0 and len(ai_first) > 0:
                diff = student_first['student_price'].mean() - ai_first['student_price'].mean()
                print(f"\n💡 Impact: Student going first vs AI going first = ${abs(diff):,.0f} difference")
                if diff > 0:
                    print(f"   → Prices are HIGHER when student goes first")
                elif diff < 0:
                    print(f"   → Prices are LOWER when student goes first")
                else:
                    print(f"   → No difference")
    
    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE: WHO GOES FIRST IMPACT".center(100))
    print("="*100 + "\n")
    
    summary_data = []
    for config in ['Base', 'M', 'P', 'M+P']:
        config_data = df_deals[df_deals['config'] == config]
        if len(config_data) > 0:
            student_first = config_data[config_data['student_goes_first'] == True]
            ai_first = config_data[config_data['student_goes_first'] == False]
            
            summary_data.append({
                'Config': config,
                'Student First (n)': len(student_first),
                'Student First Avg': f"${student_first['student_price'].mean():,.0f}" if len(student_first) > 0 else "N/A",
                'AI First (n)': len(ai_first),
                'AI First Avg': f"${ai_first['student_price'].mean():,.0f}" if len(ai_first) > 0 else "N/A",
                'Difference': f"${abs(student_first['student_price'].mean() - ai_first['student_price'].mean()):,.0f}" 
                              if len(student_first) > 0 and len(ai_first) > 0 else "N/A"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        display(summary_df)

---
## 4. 🧠 Memory (M) Evolution Tracking

Track how AI memory evolves round-by-round

# ============================================================================
# Section 4: 🧠 Memory (M) Evolution Tracking
# ============================================================================

print_header("🧠 AI MEMORY EVOLUTION TRACKING")

# Use filtered df_all and query additional fields
memory_sessions = df_all[df_all['use_memory'] == True]['session_id'].tolist()

if len(memory_sessions) == 0:
    print("❌ No sessions with memory in filtered data")
else:
    # Query full details for memory sessions
    session_ids_str = "','".join(memory_sessions)
    query_memory = f"""
    SELECT 
        session_id,
        student_role,
        use_memory,
        use_plan,
        ai_memory,
        ai_memory_history,
        transcript,
        deal_reached,
        student_deal_json,
        ai_deal_json,
        created_at
    FROM negotiation_sessions
    WHERE session_id IN ('{session_ids_str}')
      AND ai_memory_history IS NOT NULL
      AND ai_memory_history != '[]'
    ORDER BY created_at DESC
    """
    
    df_memory = pd.read_sql_query(query_memory, conn)
    
    if len(df_memory) == 0:
        print("❌ No sessions with memory history data")
    else:
        print(f"Found {len(df_memory)} sessions with Memory tracking (from filtered data)\n")
        
        # ==================== STEP 1: LIST ALL MEMORY SESSIONS ====================
        print("="*150)
        print("📋 All Sessions with Memory (M) - Choose one to analyze")
        print("="*150)
        print(f"{'IDX':>4} | {'SESSION':>10} | {'CFG':>4} | {'ROLE':>6} | {'STATUS':>10} | {'PRICE':>10} | {'TIME':>12} | {'UPDATES':>7}")
        print("-"*150)
        
        for idx, row in df_memory.iterrows():
            config = get_config_label(row['use_memory'], row['use_plan'])
            status = "✅ Deal" if row['deal_reached'] else "❌ No Deal"
            role = "Buyer" if row['student_role'] == 'side1' else "Seller"
            created = pd.to_datetime(row['created_at']).strftime('%m-%d %H:%M')
            
            # Count memory updates
            memory_history = safe_json_load(row['ai_memory_history'])
            updates = len(memory_history) if memory_history else 0
            
            # Extract final price
            if row['deal_reached']:
                deal_json = safe_json_load(row['student_deal_json'])
                price = deal_json.get('final_price') if deal_json else None
                price_str = f"${price:,}" if price else "N/A"
            else:
                price_str = "---"
            
            print(f"{idx:>4} | {row['session_id'][:10]:>10} | {config:>4} | {role:>6} | {status:>10} | {price_str:>10} | {created:>12} | {updates:>7}")
        
        print("="*150)
        print("\n⚠️  Set session_idx below to analyze a specific session\n")
        
        # ==================== STEP 2: ANALYZE SELECTED SESSION ====================
        session_idx = 0  # ⚠️ CHANGE THIS to analyze different sessions
        
        if session_idx >= len(df_memory):
            print(f"❌ Invalid session_idx: {session_idx}. Max index is {len(df_memory)-1}")
        else:
            session = df_memory.iloc[session_idx]
            
            print("\n" + "="*120)
            print(f"📌 Analyzing Session: {session['session_id'][:8]}...")
            print("="*120)
            print(f"   Student Role: {'Buyer (Fred)' if session['student_role'] == 'side1' else 'Seller (Rosalind)'}")
            print(f"   Configuration: {get_config_label(session['use_memory'], session['use_plan'])}")
            print(f"   Deal Reached: {'✅ Yes' if session['deal_reached'] else '❌ No'}")
            
            # Show final price if deal reached
            if session['deal_reached']:
                deal_json = safe_json_load(session['student_deal_json'])
                if deal_json and 'final_price' in deal_json:
                    print(f"   Final Price: ${deal_json['final_price']:,}")
            
            memory_history = safe_json_load(session['ai_memory_history'])
            
            if memory_history and len(memory_history) > 0:
                print(f"   Total Memory Updates: {len(memory_history)}\n")
                
                # Show each memory state
                print("="*120)
                print("MEMORY EVOLUTION (Round-by-Round)")
                print("="*120)
                
                for i, memory_state in enumerate(memory_history):
                    round_info = memory_state.get('round', f'Update {i+1}')
                    content = memory_state.get('content', '')
                    
                    print(f"\n{'─'*120}")
                    print(f"🧠 MEMORY STATE #{i+1} | Round {round_info}")
                    print(f"{'─'*120}")
                    print(content)
                
                # Show final memory state
                print(f"\n\n{'='*120}")
                print(f"🎯 FINAL MEMORY STATE")
                print(f"{'='*120}")
                current_memory = session['ai_memory']
                if current_memory:
                    print(current_memory)
                else:
                    print("(empty)")
                print(f"{'='*120}")
                
                # Analysis statistics
                transcript = safe_json_load(session['transcript'])
                if transcript and len(transcript) > 0:
                    total_rounds = len(transcript) // 2
                    print(f"\n📊 Memory Evolution Statistics:")
                    print(f"{'─'*80}")
                    print(f"   Total Rounds Completed: {total_rounds}")
                    print(f"   Memory Updates: {len(memory_history)}")
                    if total_rounds > 0:
                        print(f"   Updates per Round: {len(memory_history)/total_rounds:.2f}")
                    print(f"{'─'*80}")
                else:
                    print(f"\n⚠️  No transcript data available for analysis")
            else:
                print("❌ No memory history data found for this session")

---
## 5. 🎯 Planning (P) Strategy Tracking

Track how AI planning evolves round-by-round

# ============================================================================
# Section 5: 🎯 Planning (P) Strategy Tracking
# ============================================================================

print_header("🎯 AI PLANNING STRATEGY TRACKING")

# Use filtered df_all and query additional fields
plan_sessions = df_all[df_all['use_plan'] == True]['session_id'].tolist()

if len(plan_sessions) == 0:
    print("❌ No sessions with planning in filtered data")
else:
    # Query full details for planning sessions
    session_ids_str = "','".join(plan_sessions)
    query_plan = f"""
    SELECT 
        session_id,
        student_role,
        use_memory,
        use_plan,
        ai_plan,
        ai_plan_history,
        transcript,
        deal_reached,
        student_deal_json,
        ai_deal_json,
        created_at
    FROM negotiation_sessions
    WHERE session_id IN ('{session_ids_str}')
      AND ai_plan_history IS NOT NULL
      AND ai_plan_history != '[]'
    ORDER BY created_at DESC
    """
    
    df_plan = pd.read_sql_query(query_plan, conn)
    
    if len(df_plan) == 0:
        print("❌ No sessions with plan history data")
    else:
        print(f"Found {len(df_plan)} sessions with Planning tracking (from filtered data)\n")
        
        # ==================== STEP 1: LIST ALL PLANNING SESSIONS ====================
        print("="*150)
        print("📋 All Sessions with Planning (P) - Choose one to analyze")
        print("="*150)
        print(f"{'IDX':>4} | {'SESSION':>10} | {'CFG':>4} | {'ROLE':>6} | {'STATUS':>10} | {'PRICE':>10} | {'TIME':>12} | {'UPDATES':>7}")
        print("-"*150)
        
        for idx, row in df_plan.iterrows():
            config = get_config_label(row['use_memory'], row['use_plan'])
            status = "✅ Deal" if row['deal_reached'] else "❌ No Deal"
            role = "Buyer" if row['student_role'] == 'side1' else "Seller"
            created = pd.to_datetime(row['created_at']).strftime('%m-%d %H:%M')
            
            # Count plan updates
            plan_history = safe_json_load(row['ai_plan_history'])
            updates = len(plan_history) if plan_history else 0
            
            # Extract final price
            if row['deal_reached']:
                deal_json = safe_json_load(row['student_deal_json'])
                price = deal_json.get('final_price') if deal_json else None
                price_str = f"${price:,}" if price else "N/A"
            else:
                price_str = "---"
            
            print(f"{idx:>4} | {row['session_id'][:10]:>10} | {config:>4} | {role:>6} | {status:>10} | {price_str:>10} | {created:>12} | {updates:>7}")
        
        print("="*150)
        print("\n⚠️  Set session_idx below to analyze a specific session\n")
        
        # ==================== STEP 2: ANALYZE SELECTED SESSION ====================
        session_idx = 0  # ⚠️ CHANGE THIS to analyze different sessions
        
        if session_idx >= len(df_plan):
            print(f"❌ Invalid session_idx: {session_idx}. Max index is {len(df_plan)-1}")
        else:
            session = df_plan.iloc[session_idx]
            
            print("\n" + "="*120)
            print(f"📌 Analyzing Session: {session['session_id'][:8]}...")
            print("="*120)
            print(f"   Student Role: {'Buyer (Fred)' if session['student_role'] == 'side1' else 'Seller (Rosalind)'}")
            print(f"   Configuration: {get_config_label(session['use_memory'], session['use_plan'])}")
            print(f"   Deal Reached: {'✅ Yes' if session['deal_reached'] else '❌ No'}")
            
            # Show final price if deal reached
            if session['deal_reached']:
                deal_json = safe_json_load(session['student_deal_json'])
                if deal_json and 'final_price' in deal_json:
                    print(f"   Final Price: ${deal_json['final_price']:,}")
            
            plan_history = safe_json_load(session['ai_plan_history'])
            
            if plan_history and len(plan_history) > 0:
                print(f"   Total Plan Updates: {len(plan_history)}\n")
                
                # Show each plan state
                print("="*120)
                print("PLANNING EVOLUTION (Round-by-Round)")
                print("="*120)
                
                for i, plan_state in enumerate(plan_history):
                    round_info = plan_state.get('round', f'Update {i+1}')
                    content = plan_state.get('content', '')
                    
                    print(f"\n{'─'*120}")
                    print(f"🎯 PLAN STATE #{i+1} | Round {round_info}")
                    print(f"{'─'*120}")
                    print(content)
                
                # Show final plan state
                print(f"\n\n{'='*120}")
                print(f"🏁 FINAL PLAN STATE")
                print(f"{'='*120}")
                current_plan = session['ai_plan']
                if current_plan:
                    print(current_plan)
                else:
                    print("(empty)")
                print(f"{'='*120}")
                
                # Analysis statistics
                transcript = safe_json_load(session['transcript'])
                if transcript and len(transcript) > 0:
                    total_rounds = len(transcript) // 2
                    print(f"\n📊 Planning Evolution Statistics:")
                    print(f"{'─'*80}")
                    print(f"   Total Rounds Completed: {total_rounds}")
                    print(f"   Plan Updates: {len(plan_history)}")
                    if total_rounds > 0:
                        print(f"   Updates per Round: {len(plan_history)/total_rounds:.2f}")
                    print(f"{'─'*80}")
                else:
                    print(f"\n⚠️  No transcript data available for analysis")
            else:
                print("❌ No plan history data found for this session")


---
## 6. 💬 Full Transcript Viewer

Display complete conversation transcripts

# ============================================================================
# Section 6: 💬 Full Transcript Viewer
# ============================================================================

print_header("💬 FULL CONVERSATION TRANSCRIPT")

# df_deals should already be filtered from Section 3
# But let's make sure it's using the filtered sessions
if 'df_deals' not in locals() or len(df_deals) == 0:
    print("❌ No deal data available. Please run Section 3 first.")
else:
    print(f"Found {len(df_deals)} successful deals (from filtered data)\n")
    
    # ==================== STEP 1: LIST ALL SUCCESSFUL DEALS ====================
    print("="*120)
    print("📋 All Successful Deals - Choose one to view transcript")
    print("="*120)
    
    for idx, row in df_deals.iterrows():
        price = row['student_price']
        config = row['config']
        role = row['student_role_label']
        created = pd.to_datetime(row['created_at']).strftime('%m-%d %H:%M')
        rounds = row['rounds']
        duration = row['duration']
        
        print(f"{idx:3d}. {row['session_id'][:8]}... | {role:17s} | {config:4s} | ${price:>7,.0f} | {rounds} rounds | {duration:.1f} min | {created}")
    
    print("="*120)
    print("\n⚠️  Set session_to_view below to view a specific transcript\n")
    
    # ==================== STEP 2: VIEW SELECTED TRANSCRIPT ====================
    session_to_view = 10  # ⚠️ CHANGE THIS to view different sessions
    
    if session_to_view >= len(df_deals):
        print(f"❌ Invalid session_to_view: {session_to_view}. Max index is {len(df_deals)-1}")
    else:
        session = df_deals.iloc[session_to_view]
        
        print("\n" + "="*120)
        print(f"📌 Viewing Transcript: {session['session_id'][:8]}...")
        print("="*120)
        print(f"   Student Role: {session['student_role_label']}")
        print(f"   AI Role: {session['ai_role_label']}")
        print(f"   Configuration: {session['config']}")
        print(f"   Final Price: ${session['student_price']:,.0f}")
        print(f"   Rounds: {session['rounds']}")
        print(f"   Duration: {session['duration']:.1f} minutes\n")
        
        transcript = safe_json_load(session['transcript'])
        
        if transcript and len(transcript) > 0:
            print("="*120)
            print("CONVERSATION TRANSCRIPT")
            print("="*120)
            
            for i, message in enumerate(transcript):
                # Parse message format: "Round X.Y - Label: content"
                if " - " in message:
                    parts = message.split(" - ", 1)
                    round_info = parts[0]
                    
                    if len(parts) > 1 and ": " in parts[1]:
                        label, content = parts[1].split(": ", 1)
                        
                        # Determine icon based on role
                        if "Fred" in label or "Buyer" in label:
                            icon = "🏢"
                        elif "Rosalind" in label or "Seller" in label:
                            icon = "🏘️"
                        else:
                            icon = "💬"
                        
                        print(f"\n{icon} {round_info} - {label}")
                        print("─"*120)
                        print(content)
                    else:
                        print(f"\n💬 {round_info}")
                        print("─"*120)
                        print(parts[1] if len(parts) > 1 else message)
                else:
                    print(f"\n💬 Message {i+1}")
                    print("─"*120)
                    print(message)
            
            print("\n" + "="*120)
            print(f"✅ Deal reached at price: ${session['student_price']:,.0f}")
            print("="*120)
        else:
            print("❌ No transcript data available for this session")

---
## 7. 📊 Performance Metrics Dashboard

Comprehensive performance metrics across all sessions

print_header("📊 PERFORMANCE METRICS DASHBOARD")

# Overall success rates
print_subheader("✅ Success Rates by Configuration")
success_by_config = df_all.groupby('Config').agg({
    'deal_reached': ['sum', 'count', 'mean']
})
success_by_config.columns = ['Successful', 'Total', 'Success Rate']
success_by_config['Success Rate'] = (success_by_config['Success Rate'] * 100).round(1)
print(success_by_config.to_string())

# Duration analysis
if len(df_deals) > 0:
    print_subheader("⏱️ Duration Analysis (Successful Deals Only)")
    duration_by_config = df_deals.groupby('config')['duration'].agg(['count', 'mean', 'median', 'min', 'max'])
    duration_by_config = duration_by_config.round(1)
    duration_by_config.columns = ['N', 'Mean (min)', 'Median (min)', 'Min (min)', 'Max (min)']
    print(duration_by_config.to_string())
    
    # Rounds analysis
    print_subheader("🔄 Rounds Analysis (Successful Deals Only)")
    rounds_by_config = df_deals.groupby('config')['rounds'].agg(['count', 'mean', 'median', 'min', 'max'])
    rounds_by_config = rounds_by_config.round(1)
    rounds_by_config.columns = ['N', 'Mean', 'Median', 'Min', 'Max']
    print(rounds_by_config.to_string())

# Efficiency metrics
print_subheader("⚡ Efficiency: Minutes per Round")
if len(df_deals) > 0:
    df_deals['min_per_round'] = df_deals['duration'] / df_deals['rounds']
    efficiency = df_deals.groupby('config')['min_per_round'].agg(['mean', 'median']).round(2)
    efficiency.columns = ['Mean (min/round)', 'Median (min/round)']
    print(efficiency.to_string())

# Value distribution by role
if len(df_deals) > 0:
    print_subheader("💰 Price Distribution by Student Role")
    
    print("\nWhen Student is BUYER (wants lower price):")
    buyer_deals = df_deals[df_deals['student_role'] == 'side1']
    if len(buyer_deals) > 0:
        buyer_stats = buyer_deals.groupby('config')['student_price'].agg(['count', 'mean', 'median'])
        buyer_stats['mean'] = buyer_stats['mean'].apply(lambda x: f"${x:,.0f}")
        buyer_stats['median'] = buyer_stats['median'].apply(lambda x: f"${x:,.0f}")
        buyer_stats.columns = ['N', 'Mean Price', 'Median Price']
        print(buyer_stats.to_string())
    else:
        print("  No data")
    
    print("\nWhen Student is SELLER (wants higher price):")
    seller_deals = df_deals[df_deals['student_role'] == 'side2']
    if len(seller_deals) > 0:
        seller_stats = seller_deals.groupby('config')['student_price'].agg(['count', 'mean', 'median'])
        seller_stats['mean'] = seller_stats['mean'].apply(lambda x: f"${x:,.0f}")
        seller_stats['median'] = seller_stats['median'].apply(lambda x: f"${x:,.0f}")
        seller_stats.columns = ['N', 'Mean Price', 'Median Price']
        print(seller_stats.to_string())
    else:
        print("  No data")

print("\n" + "="*100)

# ==================== DATA FILTERING ====================
# 1. Filter by major: Exclude SDS/sds first
df_all = df_all[
    ~df_all['major'].str.lower().isin(['sds']) | df_all['major'].isna()
].copy()

# 2. Define time windows
morning_start = pd.to_datetime('2025-12-10 14:36')
morning_end = pd.to_datetime('2025-12-10 15:48')
afternoon_start = pd.to_datetime('2025-12-10 19:21')
afternoon_end = pd.to_datetime('2025-12-10 20:01')

# 3. Create time period column
def get_time_period(created_at):
    dt = pd.to_datetime(created_at)
    if morning_start <= dt <= morning_end:
        return 'Morning'
    elif afternoon_start <= dt <= afternoon_end:
        return 'Afternoon'
    else:
        return 'Other'

df_all['time_period'] = df_all['created_at'].apply(get_time_period)

# 4. Filter to only include Morning and Afternoon sessions
df_all = df_all[df_all['time_period'].isin(['Morning', 'Afternoon'])].copy()

# ==================== ADD ADDITIONAL COLUMNS ====================
# Recreate derived columns after filtering
df_all['Config'] = df_all.apply(lambda r: get_config_label(r['use_memory'], r['use_plan']), axis=1)
df_all['Duration (min)'] = df_all.apply(lambda r: calculate_duration(r['created_at'], r['updated_at']), axis=1)
df_all['Outcome'] = df_all.apply(lambda r: 
    '✅ Deal' if r['deal_reached'] 
    else '❌ Failed' if r['deal_failed'] 
    else '⏸️ Incomplete', axis=1)
df_all['Role'] = df_all['student_role'].map({
    'side1': '🏢 Buyer (Fred)',
    'side2': '🏘️ Seller (Rosalind)'
})

# Add rounds column
def count_rounds(transcript_json):
    transcript = safe_json_load(transcript_json)
    return len(transcript) // 2 if transcript else 0

df_all['rounds'] = df_all['transcript'].apply(count_rounds)

# Add final price column (with fallback to AI JSON)
def extract_final_price(student_json_str, ai_json_str):
    # Try student JSON first
    student_json = safe_json_load(student_json_str)
    if student_json and 'final_price' in student_json:
        return student_json['final_price']
    
    # Fallback to AI JSON
    ai_json = safe_json_load(ai_json_str)
    if ai_json and 'final_price' in ai_json:
        return ai_json['final_price']
    
    return None

df_all['final_price'] = df_all.apply(
    lambda r: extract_final_price(r['student_deal_json'], r['ai_deal_json']) if r['deal_reached'] else None, 
    axis=1
)

# ==================== OVERALL STATISTICS ====================
print_header("📊 OVERALL STATISTICS (Both Sessions)")

print(f"Total Sessions: {len(df_all)}")
print(f"  Morning (14:36-15:48): {(df_all['time_period'] == 'Morning').sum()}")
print(f"  Afternoon (19:21-20:01): {(df_all['time_period'] == 'Afternoon').sum()}")

print(f"\nSuccessful Deals: {df_all['deal_reached'].sum()}")
print(f"Failed Negotiations: {df_all['deal_failed'].sum()}")
print(f"Incomplete: {(~df_all['deal_reached'] & ~df_all['deal_failed']).sum()}")

print(f"\n📊 Configuration Breakdown:")
print(df_all['Config'].value_counts().to_string())

print(f"\n🎭 Role Distribution:")
print(df_all['Role'].value_counts().to_string())

# ==================== DISPLAY ALL DATA ====================
print("\n" + "="*170)
print("ALL FILTERED DATA")
print("="*170)

display_df = df_all[[
    'session_id', 'time_period', 'Role', 'Config', 'Outcome',
    'major', 'gender', 'negotiation_experience',
    'rounds', 'final_price', 'Duration (min)', 'created_at'
]].copy()

display_df['session_id'] = display_df['session_id'].str[:8]
display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%m-%d %H:%M')
display_df['Duration (min)'] = display_df['Duration (min)'].round(1)
display_df['final_price'] = display_df['final_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
display_df['rounds'] = display_df['rounds'].fillna(0).astype(int)

display_df.columns = ['Session', 'Period', 'Role', 'Config', 'Outcome', 'Major', 'Gender', 'Exp', 'Rounds', 'Final Price', 'Duration', 'Created']

pd.set_option('display.max_rows', None)
display(display_df)


# ==================== FUNCTION: ANALYZE ONE TIME PERIOD ====================
def analyze_time_period(df, period_name, period_label):
    """
    Analyze a specific time period's data
    
    Args:
        df: DataFrame filtered to one time period
        period_name: 'Morning' or 'Afternoon'
        period_label: Display label like '🌅 MORNING SESSION (14:36-15:48)'
    """
    print_header(period_label)
    
    if len(df) == 0:
        print(f"❌ No data for {period_name}")
        return
    
    # Basic statistics
    print(f"Total Sessions: {len(df)}")
    print(f"Successful Deals: {df['deal_reached'].sum()} ({df['deal_reached'].sum()/len(df)*100:.1f}%)")
    print(f"Failed Negotiations: {df['deal_failed'].sum()}")
    print(f"Incomplete: {(~df['deal_reached'] & ~df['deal_failed']).sum()}")
    
    print(f"\n📊 Configuration Breakdown:")
    print(df['Config'].value_counts().to_string())
    
    print(f"\n🎭 Role Distribution:")
    print(df['Role'].value_counts().to_string())
    
    # Demographics
    print_subheader("👥 Demographics")
    
    if df['major'].notna().sum() > 0:
        print("\n📚 Major:")
        major_counts = df['major'].value_counts()
        for major, count in major_counts.items():
            print(f"  {major}: {count}")
    
    if df['gender'].notna().sum() > 0:
        print("\n⚧ Gender:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            print(f"  {gender}: {count}")
    
    if df['negotiation_experience'].notna().sum() > 0:
        print("\n💼 Experience:")
        exp_counts = df['negotiation_experience'].value_counts()
        for exp, count in exp_counts.items():
            print(f"  {exp}: {count}")
    
    # Deal analysis
    df_deals = df[df['deal_reached'] == True].copy()
    
    if len(df_deals) > 0:
        print_subheader("💰 Deal Price Analysis")
        
        # Extract numeric prices for statistics
        df_deals['price_numeric'] = df_deals['final_price']
        valid_prices = df_deals['price_numeric'].dropna()
        
        if len(valid_prices) > 0:
            print(f"\n📊 Overall Price Statistics:")
            print(f"  Average: ${valid_prices.mean():,.0f}")
            print(f"  Median:  ${valid_prices.median():,.0f}")
            print(f"  Min:     ${valid_prices.min():,.0f}")
            print(f"  Max:     ${valid_prices.max():,.0f}")
            print(f"  Std Dev: ${valid_prices.std():,.0f}")
            
            # By configuration
            print(f"\n🔧 By Configuration:")
            for config in ['Base', 'M', 'P', 'M+P']:
                config_deals = df_deals[df_deals['Config'] == config]
                config_prices = config_deals['price_numeric'].dropna()
                if len(config_prices) > 0:
                    print(f"\n  {config}:")
                    print(f"    Count: {len(config_prices)}")
                    print(f"    Average: ${config_prices.mean():,.0f}")
                    print(f"    Median:  ${config_prices.median():,.0f}")
                    print(f"    Range: ${config_prices.min():,.0f} - ${config_prices.max():,.0f}")
            
            # By role
            print(f"\n🎭 By Student Role:")
            
            buyer_deals = df_deals[df_deals['student_role'] == 'side1']
            buyer_prices = buyer_deals['price_numeric'].dropna()
            if len(buyer_prices) > 0:
                print(f"\n  When Student is BUYER (Fred):")
                print(f"    Goal: Minimize price (lower is better)")
                print(f"    Count: {len(buyer_prices)}")
                print(f"    Average: ${buyer_prices.mean():,.0f}")
                print(f"    Range: ${buyer_prices.min():,.0f} - ${buyer_prices.max():,.0f}")
            
            seller_deals = df_deals[df_deals['student_role'] == 'side2']
            seller_prices = seller_deals['price_numeric'].dropna()
            if len(seller_prices) > 0:
                print(f"\n  When Student is SELLER (Rosalind):")
                print(f"    Goal: Maximize price (higher is better)")
                print(f"    Count: {len(seller_prices)}")
                print(f"    Average: ${seller_prices.mean():,.0f}")
                print(f"    Range: ${seller_prices.min():,.0f} - ${seller_prices.max():,.0f}")
        
        # Duration and rounds
        print_subheader("⏱️ Duration & Rounds Analysis")
        
        valid_durations = df_deals['Duration (min)'].dropna()
        if len(valid_durations) > 0:
            print(f"\nDuration Statistics:")
            print(f"  Average: {valid_durations.mean():.1f} minutes")
            print(f"  Median:  {valid_durations.median():.1f} minutes")
            print(f"  Range: {valid_durations.min():.1f} - {valid_durations.max():.1f} minutes")
        
        valid_rounds = df_deals['rounds']
        if len(valid_rounds) > 0:
            print(f"\nRounds Statistics:")
            print(f"  Average: {valid_rounds.mean():.1f} rounds")
            print(f"  Median:  {valid_rounds.median():.0f} rounds")
            print(f"  Range: {valid_rounds.min():.0f} - {valid_rounds.max():.0f} rounds")
    
    # Detailed table
    print_subheader("📋 Detailed Session List")
    
    detail_df = df[[
        'session_id', 'Role', 'Config', 'Outcome', 'major',
        'rounds', 'final_price', 'Duration (min)', 'created_at'
    ]].copy()
    
    detail_df['session_id'] = detail_df['session_id'].str[:8]
    detail_df['created_at'] = pd.to_datetime(detail_df['created_at']).dt.strftime('%H:%M')
    detail_df['Duration (min)'] = detail_df['Duration (min)'].round(1)
    detail_df['final_price'] = detail_df['final_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    
    detail_df.columns = ['Session', 'Role', 'Config', 'Outcome', 'Major', 'Rounds', 'Price', 'Duration', 'Time']
    
    display(detail_df)


# ==================== MORNING SESSION ANALYSIS ====================
df_morning = df_all[df_all['time_period'] == 'Morning'].copy()
analyze_time_period(df_morning, 'Morning', '🌅 MORNING SESSION (14:36-15:48)')

# ==================== AFTERNOON SESSION ANALYSIS ====================
df_afternoon = df_all[df_all['time_period'] == 'Afternoon'].copy()
analyze_time_period(df_afternoon, 'Afternoon', '🌆 AFTERNOON SESSION (19:21-20:01)')

# ==================== COMPARISON BETWEEN SESSIONS ====================
print_header("⚖️ MORNING vs AFTERNOON COMPARISON")

comparison_data = []

for period_name, df_period in [('Morning', df_morning), ('Afternoon', df_afternoon)]:
    deals = df_period[df_period['deal_reached'] == True]
    prices = deals['final_price'].dropna()
    
    comparison_data.append({
        'Period': period_name,
        'Total Sessions': len(df_period),
        'Deals': len(deals),
        'Success Rate': f"{len(deals)/len(df_period)*100:.1f}%" if len(df_period) > 0 else "N/A",
        'Avg Price': f"${prices.mean():,.0f}" if len(prices) > 0 else "N/A",
        'Avg Rounds': f"{deals['rounds'].mean():.1f}" if len(deals) > 0 else "N/A",
        'Avg Duration': f"{deals['Duration (min)'].mean():.1f} min" if len(deals) > 0 else "N/A"
    })

comparison_df = pd.DataFrame(comparison_data)
display(comparison_df)

# ============================================================================
# Section 1: 📋 Seller-Only Analysis (Student as Seller)
# ============================================================================

import sqlite3
import pandas as pd
import json
import numpy as np
from datetime import datetime
from IPython.display import display, HTML, Markdown
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DB_PATH = "negotiations_mainst.db"  # ⚠️ CHANGE THIS TO YOUR DATABASE PATH
SCENARIO_FILTER = "Main_Street"  # Focus on Main Street scenario

conn = sqlite3.connect(DB_PATH)
print(f"✅ Connected to: {DB_PATH}")
print(f"🎯 Analyzing scenario: {SCENARIO_FILTER}")
print(f"🎯 Focus: Student as SELLER only\n")

# ==================== HELPER FUNCTIONS ====================
def safe_json_load(json_str):
    if not json_str or json_str == 'null':
        return None
    try:
        return json.loads(json_str)
    except:
        return None

def calculate_duration(start, end):
    try:
        return (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 60
    except:
        return None

def get_config_label(use_memory, use_plan):
    if use_memory and use_plan:
        return 'M+P'
    elif use_memory:
        return 'M'
    elif use_plan:
        return 'P'
    else:
        return 'Base'

def print_header(title, char='=', width=100):
    print("\n" + char * width)
    print(f"{title.center(width)}")
    print(char * width + "\n")

def print_subheader(title, char='-', width=80):
    print("\n" + char * width)
    print(title)
    print(char * width)

print("✅ Helper functions loaded")

# ==================== LOAD DATA ====================
query_all = f"""
SELECT 
    session_id,
    student_role,
    ai_model,
    student_goes_first,
    use_memory,
    use_plan,
    total_rounds,
    deal_reached,
    deal_failed,
    status,
    created_at,
    updated_at,
    major,
    gender,
    negotiation_experience,
    transcript,
    student_deal_json,
    ai_deal_json
FROM negotiation_sessions
WHERE scenario_name = '{SCENARIO_FILTER}'
ORDER BY created_at DESC
"""

df_all = pd.read_sql_query(query_all, conn)
print(f"✅ Loaded {len(df_all)} sessions from database")

# ==================== DATA FILTERING ====================
print("\n" + "="*100)
print("🔍 APPLYING FILTERS")
print("="*100)

# 1. Filter by major: Exclude SDS/sds
original_count = len(df_all)
df_all = df_all[
    ~df_all['major'].str.lower().isin(['sds']) | df_all['major'].isna()
].copy()
print(f"1. Major filter: {original_count} → {len(df_all)} sessions (excluded SDS/sds)")

# 2. Filter by role: Only keep student as seller (side2)
before_role_filter = len(df_all)
df_all = df_all[df_all['student_role'] == 'side2'].copy()
print(f"2. Role filter: {before_role_filter} → {len(df_all)} sessions (Student = Seller only)")

# 3. Define time windows
morning_start = pd.to_datetime('2025-12-10 14:36')
morning_end = pd.to_datetime('2025-12-10 15:48')
afternoon_start = pd.to_datetime('2025-12-10 19:21')
afternoon_end = pd.to_datetime('2025-12-10 20:01')

# 4. Create time period column
def get_time_period(created_at):
    dt = pd.to_datetime(created_at)
    if morning_start <= dt <= morning_end:
        return 'Morning'
    elif afternoon_start <= dt <= afternoon_end:
        return 'Afternoon'
    else:
        return 'Other'

df_all['time_period'] = df_all['created_at'].apply(get_time_period)

# 5. Filter to only include Morning and Afternoon sessions
before_time_filter = len(df_all)
df_all = df_all[df_all['time_period'].isin(['Morning', 'Afternoon'])].copy()
print(f"3. Time filter: {before_time_filter} → {len(df_all)} sessions (Morning: 14:36-15:48, Afternoon: 19:21-20:01)")

# ==================== ADD ADDITIONAL COLUMNS ====================
print("\n✅ Creating derived columns...")

# Config
df_all['Config'] = df_all.apply(lambda r: get_config_label(r['use_memory'], r['use_plan']), axis=1)

# Duration
df_all['Duration (min)'] = df_all.apply(lambda r: calculate_duration(r['created_at'], r['updated_at']), axis=1)

# Outcome
df_all['Outcome'] = df_all.apply(lambda r: 
    '✅ Deal' if r['deal_reached'] 
    else '❌ Failed' if r['deal_failed'] 
    else '⏸️ Incomplete', axis=1)

# Role (all will be Seller now)
df_all['Role'] = '🏘️ Seller (Rosalind)'

# Rounds
def count_rounds(transcript_json):
    transcript = safe_json_load(transcript_json)
    return len(transcript) // 2 if transcript else 0

df_all['rounds'] = df_all['transcript'].apply(count_rounds)

# Final price (with fallback to AI JSON)
def extract_final_price(student_json_str, ai_json_str):
    # Try student JSON first
    student_json = safe_json_load(student_json_str)
    if student_json and 'final_price' in student_json:
        return student_json['final_price']
    
    # Fallback to AI JSON
    ai_json = safe_json_load(ai_json_str)
    if ai_json and 'final_price' in ai_json:
        return ai_json['final_price']
    
    return None

df_all['final_price'] = df_all.apply(
    lambda r: extract_final_price(r['student_deal_json'], r['ai_deal_json']) if r['deal_reached'] else None, 
    axis=1
)

print("✅ All columns created")

# ==================== OVERALL STATISTICS ====================
print_header("📊 OVERALL STATISTICS - SELLER ONLY (Both Sessions)")

print(f"Total Seller Sessions: {len(df_all)}")
print(f"  Morning (14:36-15:48): {(df_all['time_period'] == 'Morning').sum()}")
print(f"  Afternoon (19:21-20:01): {(df_all['time_period'] == 'Afternoon').sum()}")

print(f"\nSuccessful Deals: {df_all['deal_reached'].sum()} ({df_all['deal_reached'].sum()/len(df_all)*100:.1f}%)")
print(f"Failed Negotiations: {df_all['deal_failed'].sum()}")
print(f"Incomplete: {(~df_all['deal_reached'] & ~df_all['deal_failed']).sum()}")

print(f"\n📊 Configuration Breakdown:")
print(df_all['Config'].value_counts().to_string())

print(f"\n🎯 Who Goes First:")
print(df_all['student_goes_first'].map({True: 'Student First', False: 'AI First'}).value_counts().to_string())

# ==================== DISPLAY ALL DATA ====================
print("\n" + "="*170)
print("ALL FILTERED DATA - SELLER ONLY")
print("="*170)

display_df = df_all[[
    'session_id', 'time_period', 'Config', 'Outcome',
    'major', 'gender', 'negotiation_experience',
    'student_goes_first', 'rounds', 'final_price', 'Duration (min)', 'created_at'
]].copy()

display_df['session_id'] = display_df['session_id'].str[:8]
display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%m-%d %H:%M')
display_df['Duration (min)'] = display_df['Duration (min)'].round(1)
display_df['final_price'] = display_df['final_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
display_df['rounds'] = display_df['rounds'].fillna(0).astype(int)
display_df['student_goes_first'] = display_df['student_goes_first'].map({True: '👤', False: '🤖'})

display_df.columns = ['Session', 'Period', 'Config', 'Outcome', 'Major', 'Gender', 'Exp', 'First', 'Rounds', 'Final Price', 'Duration', 'Created']

pd.set_option('display.max_rows', None)
display(display_df)


# ==================== ANALYSIS FUNCTION ====================
def analyze_time_period(df, period_name, period_label):
    """Analyze a specific time period's data - SELLER ONLY"""
    
    print_header(period_label)
    
    if len(df) == 0:
        print(f"❌ No data for {period_name}")
        return
    
    # Basic statistics
    print(f"Total Seller Sessions: {len(df)}")
    print(f"Successful Deals: {df['deal_reached'].sum()} ({df['deal_reached'].sum()/len(df)*100:.1f}%)")
    print(f"Failed Negotiations: {df['deal_failed'].sum()}")
    print(f"Incomplete: {(~df['deal_reached'] & ~df['deal_failed']).sum()}")
    
    print(f"\n📊 Configuration Breakdown:")
    print(df['Config'].value_counts().to_string())
    
    print(f"\n🎯 Who Goes First:")
    print(df['student_goes_first'].map({True: 'Student First', False: 'AI First'}).value_counts().to_string())
    
    # Demographics
    print_subheader("👥 Demographics")
    
    if df['major'].notna().sum() > 0:
        print("\n📚 Major:")
        major_counts = df['major'].value_counts()
        for major, count in major_counts.items():
            print(f"  {major}: {count}")
    
    if df['gender'].notna().sum() > 0:
        print("\n⚧ Gender:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            print(f"  {gender}: {count}")
    
    if df['negotiation_experience'].notna().sum() > 0:
        print("\n💼 Experience:")
        exp_counts = df['negotiation_experience'].value_counts()
        for exp, count in exp_counts.items():
            print(f"  {exp}: {count}")
    
    # Deal analysis
    df_deals = df[df['deal_reached'] == True].copy()
    
    if len(df_deals) == 0:
        print("\n❌ No successful deals in this period")
        return
    
    print_subheader("💰 Deal Price Analysis - SELLER PERSPECTIVE")
    print("📈 Remember: As SELLER, HIGHER prices are BETTER (Goal: Maximize price)")
    
    # Extract numeric prices
    df_deals['price_numeric'] = df_deals['final_price']
    valid_prices = df_deals['price_numeric'].dropna()
    
    if len(valid_prices) == 0:
        print("\n❌ No valid price data")
        return
    
    print(f"\n📊 Overall Price Statistics:")
    print(f"  Average: ${valid_prices.mean():,.0f}")
    print(f"  Median:  ${valid_prices.median():,.0f}")
    print(f"  Min:     ${valid_prices.min():,.0f}")
    print(f"  Max:     ${valid_prices.max():,.0f}")
    print(f"  Std Dev: ${valid_prices.std():,.0f}")
    
    # ============================================================
    # DETAILED ANALYSIS: BASE vs M+P
    # ============================================================
    print_subheader("🔧 Configuration Deep Dive: BASE vs M+P")
    
    for config in ['Base', 'M+P']:
        config_deals = df_deals[df_deals['Config'] == config].copy()
        config_prices = config_deals['price_numeric'].dropna()
        
        if len(config_prices) == 0:
            print(f"\n{config}: No deals found")
            continue
        
        print(f"\n{'='*80}")
        print(f"📊 {config} Configuration - SELLER Results")
        print(f"{'='*80}")
        print(f"Total Deals: {len(config_prices)}")
        print(f"Average Price: ${config_prices.mean():,.0f} 📈 (Higher is better for seller)")
        print(f"Median Price:  ${config_prices.median():,.0f}")
        print(f"Price Range: ${config_prices.min():,.0f} - ${config_prices.max():,.0f}")
        if len(config_prices) > 1:
            print(f"Std Dev: ${config_prices.std():,.0f}")
        
        # ============================================================
        # WHO GOES FIRST ANALYSIS
        # ============================================================
        print(f"\n🎯 Who Goes First Analysis:")
        
        student_first = config_deals[config_deals['student_goes_first'] == True]
        ai_first = config_deals[config_deals['student_goes_first'] == False]
        
        student_first_prices = student_first['price_numeric'].dropna()
        ai_first_prices = ai_first['price_numeric'].dropna()
        
        print(f"\n  👤 Student First (n={len(student_first_prices)}):")
        if len(student_first_prices) > 0:
            print(f"    Average: ${student_first_prices.mean():,.0f}")
            print(f"    Median:  ${student_first_prices.median():,.0f}")
            print(f"    Range: ${student_first_prices.min():,.0f} - ${student_first_prices.max():,.0f}")
        else:
            print(f"    No data")
        
        print(f"\n  🤖 AI First (n={len(ai_first_prices)}):")
        if len(ai_first_prices) > 0:
            print(f"    Average: ${ai_first_prices.mean():,.0f}")
            print(f"    Median:  ${ai_first_prices.median():,.0f}")
            print(f"    Range: ${ai_first_prices.min():,.0f} - ${ai_first_prices.max():,.0f}")
        else:
            print(f"    No data")
        
        # Calculate impact if both exist
        if len(student_first_prices) > 0 and len(ai_first_prices) > 0:
            diff = student_first_prices.mean() - ai_first_prices.mean()
            print(f"\n  💡 Impact for SELLER:")
            print(f"    Price difference: ${abs(diff):,.0f}")
            if diff > 0:
                print(f"    ✅ BETTER for seller when student goes first (+${diff:,.0f})")
            elif diff < 0:
                print(f"    ❌ WORSE for seller when student goes first (${diff:,.0f})")
            else:
                print(f"    → No difference")
    
    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print_subheader("📊 Summary Comparison Table - SELLER PERSPECTIVE")
    
    comparison_data = []
    for config in ['Base', 'M+P']:
        config_deals_sub = df_deals[df_deals['Config'] == config]
        config_prices_sub = config_deals_sub['price_numeric'].dropna()
        
        if len(config_prices_sub) == 0:
            continue
        
        # Overall
        comparison_data.append({
            'Config': config,
            'Category': 'Overall',
            'N': len(config_prices_sub),
            'Avg Price': f"${config_prices_sub.mean():,.0f}",
            'Median': f"${config_prices_sub.median():,.0f}",
            'Range': f"${config_prices_sub.min():,.0f}-${config_prices_sub.max():,.0f}"
        })
        
        # Student goes first
        sf = config_deals_sub[config_deals_sub['student_goes_first'] == True]['price_numeric'].dropna()
        if len(sf) > 0:
            comparison_data.append({
                'Config': config,
                'Category': 'Student First',
                'N': len(sf),
                'Avg Price': f"${sf.mean():,.0f}",
                'Median': f"${sf.median():,.0f}",
                'Range': f"${sf.min():,.0f}-${sf.max():,.0f}"
            })
        
        # AI goes first
        af = config_deals_sub[config_deals_sub['student_goes_first'] == False]['price_numeric'].dropna()
        if len(af) > 0:
            comparison_data.append({
                'Config': config,
                'Category': 'AI First',
                'N': len(af),
                'Avg Price': f"${af.mean():,.0f}",
                'Median': f"${af.median():,.0f}",
                'Range': f"${af.min():,.0f}-${af.max():,.0f}"
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        display(comp_df)
    
    # Duration and rounds
    print_subheader("⏱️ Duration & Rounds Analysis")
    
    valid_durations = df_deals['Duration (min)'].dropna()
    if len(valid_durations) > 0:
        print(f"\nDuration Statistics:")
        print(f"  Average: {valid_durations.mean():.1f} minutes")
        print(f"  Median:  {valid_durations.median():.1f} minutes")
        print(f"  Range: {valid_durations.min():.1f} - {valid_durations.max():.1f} minutes")
    
    valid_rounds = df_deals['rounds']
    if len(valid_rounds) > 0:
        print(f"\nRounds Statistics:")
        print(f"  Average: {valid_rounds.mean():.1f} rounds")
        print(f"  Median:  {valid_rounds.median():.0f} rounds")
        print(f"  Range: {valid_rounds.min():.0f} - {valid_rounds.max():.0f} rounds")
    
    # Detailed table
    print_subheader("📋 Detailed Session List - SELLER ONLY")
    
    detail_df = df[[
        'session_id', 'Config', 'Outcome', 'major',
        'student_goes_first', 'rounds', 'final_price', 'Duration (min)', 'created_at'
    ]].copy()
    
    detail_df['session_id'] = detail_df['session_id'].str[:8]
    detail_df['created_at'] = pd.to_datetime(detail_df['created_at']).dt.strftime('%H:%M')
    detail_df['Duration (min)'] = detail_df['Duration (min)'].round(1)
    detail_df['final_price'] = detail_df['final_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    detail_df['student_goes_first'] = detail_df['student_goes_first'].map({True: '👤 Student', False: '🤖 AI'})
    
    detail_df.columns = ['Session', 'Config', 'Outcome', 'Major', 'First', 'Rounds', 'Price', 'Duration', 'Time']
    
    display(detail_df)


# ==================== MORNING SESSION ANALYSIS ====================
df_morning = df_all[df_all['time_period'] == 'Morning'].copy()
analyze_time_period(df_morning, 'Morning', '🌅 MORNING SESSION - SELLER ONLY (14:36-15:48)')

# ==================== AFTERNOON SESSION ANALYSIS ====================
df_afternoon = df_all[df_all['time_period'] == 'Afternoon'].copy()
analyze_time_period(df_afternoon, 'Afternoon', '🌆 AFTERNOON SESSION - SELLER ONLY (19:21-20:01)')

# ==================== COMPARISON BETWEEN SESSIONS ====================
print_header("⚖️ MORNING vs AFTERNOON COMPARISON - SELLER ONLY")

comparison_data = []

for period_name, df_period in [('Morning', df_morning), ('Afternoon', df_afternoon)]:
    deals = df_period[df_period['deal_reached'] == True]
    prices = deals['final_price'].dropna()
    
    comparison_data.append({
        'Period': period_name,
        'Total Sessions': len(df_period),
        'Deals': len(deals),
        'Success Rate': f"{len(deals)/len(df_period)*100:.1f}%" if len(df_period) > 0 else "N/A",
        'Avg Price': f"${prices.mean():,.0f}" if len(prices) > 0 else "N/A",
        'Avg Rounds': f"{deals['rounds'].mean():.1f}" if len(deals) > 0 else "N/A",
        'Avg Duration': f"{deals['Duration (min)'].mean():.1f} min" if len(deals) > 0 else "N/A"
    })

comparison_df = pd.DataFrame(comparison_data)
display(comparison_df)

print("\n✅ Seller-only analysis complete!")
print("📈 Remember: As SELLER, higher prices indicate better negotiation outcomes")

---
## 8. Close Database Connection

conn.close()
print("✅ Database connection closed")