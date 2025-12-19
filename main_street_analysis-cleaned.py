# 🏘️ Main Street Negotiation - Comprehensive Analysis Dashboard
"""
Complete analysis of Main Street real estate negotiations (Fred Starr vs Rosalind Cain)

Analysis Sections:
1. 📋 All Negotiation Records Overview
2. 👥 Student Demographics & Characteristics Analysis
3. 💰 Configuration Comparison: Base vs M vs M+P (Price Analysis)
4. 🧠 Memory (M) Evolution Tracking
5. 🎯 Planning (P) Strategy Tracking
6. 💬 Full Transcript Viewer
7. 📊 Performance Metrics Dashboard
8. ⏰ Time-Window Comparisons
9. 🏘️ Seller-Only Analysis
"""

# ==================== SETUP ====================
import sqlite3
import pandas as pd
import json
import numpy as np
from datetime import datetime
from IPython.display import display
import warnings

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
DB_PATH = "negotiations_mainst.db"
SCENARIO_FILTER = "Main_Street"
CUTOFF_TIME = "2025-12-10 14:36"
EXCLUDED_MAJORS = {"sds"}

TIME_WINDOWS = {
    "Morning": ("2025-12-10 14:36", "2025-12-10 15:48"),
    "Afternoon": ("2025-12-10 19:21", "2025-12-10 20:01"),
}

SESSION_IDX_MEMORY = 0
SESSION_IDX_PLAN = 0
SESSION_IDX_TRANSCRIPT = 10


# ==================== HELPERS ====================


def configure_display():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)


def safe_json_load(json_str):
    if not json_str or json_str == "null":
        return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def calculate_duration(start, end):
    try:
        return (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 60
    except (TypeError, ValueError):
        return None


def get_config_label(use_memory, use_plan):
    if use_memory and use_plan:
        return "M+P"
    if use_memory:
        return "M"
    if use_plan:
        return "P"
    return "Base"


def print_header(title, char="=", width=100):
    print("\n" + char * width)
    print(f"{title.center(width)}")
    print(char * width + "\n")


def print_subheader(title, char="-", width=80):
    print("\n" + char * width)
    print(title)
    print(char * width)


def count_rounds(transcript_json):
    transcript = safe_json_load(transcript_json)
    return len(transcript) // 2 if transcript else 0


def extract_final_price(student_json_str, ai_json_str):
    student_json = safe_json_load(student_json_str)
    if student_json and "final_price" in student_json:
        return student_json["final_price"]

    ai_json = safe_json_load(ai_json_str)
    if ai_json and "final_price" in ai_json:
        return ai_json["final_price"]

    return None


def add_derived_columns(df):
    df = df.copy()
    df["Config"] = df.apply(lambda r: get_config_label(r["use_memory"], r["use_plan"]), axis=1)
    df["Duration (min)"] = df.apply(
        lambda r: calculate_duration(r["created_at"], r["updated_at"]), axis=1
    )
    df["Outcome"] = df.apply(
        lambda r: "✅ Deal"
        if r["deal_reached"]
        else "❌ Failed"
        if r["deal_failed"]
        else "⏸️ Incomplete",
        axis=1,
    )
    df["Role"] = df["student_role"].map(
        {"side1": "🏢 Buyer (Fred)", "side2": "🏘️ Seller (Rosalind)"}
    )
    df["rounds"] = df["transcript"].apply(count_rounds)
    df["final_price"] = df.apply(
        lambda r: extract_final_price(r["student_deal_json"], r["ai_deal_json"])
        if r["deal_reached"]
        else None,
        axis=1,
    )
    return df


def add_time_periods(df, time_windows):
    df = df.copy()

    def get_time_period(created_at):
        dt = pd.to_datetime(created_at)
        for label, (start, end) in time_windows.items():
            if pd.to_datetime(start) <= dt <= pd.to_datetime(end):
                return label
        return "Other"

    df["time_period"] = df["created_at"].apply(get_time_period)
    return df


def load_sessions(conn, scenario_name):
    query_all = f"""
    SELECT
        session_id,
        student_role,
        ai_role,
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
    WHERE scenario_name = '{scenario_name}'
    ORDER BY created_at DESC
    """
    return pd.read_sql_query(query_all, conn)


def apply_filters(df, cutoff_time=None, excluded_majors=None, role=None, time_windows=None):
    df = df.copy()

    if cutoff_time:
        df = df[pd.to_datetime(df["created_at"]) >= pd.to_datetime(cutoff_time)].copy()

    if excluded_majors:
        df = df[
            ~df["major"].str.lower().isin(excluded_majors) | df["major"].isna()
        ].copy()

    if role:
        df = df[df["student_role"] == role].copy()

    if time_windows:
        df = add_time_periods(df, time_windows)
        df = df[df["time_period"].isin(time_windows.keys())].copy()

    return df


def to_display_df(df, columns, column_map, created_format="%Y-%m-%d %H:%M"):
    display_df = df[columns].copy()
    display_df["session_id"] = display_df["session_id"].str[:8]
    display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime(created_format)
    display_df["Duration (min)"] = display_df["Duration (min)"].round(1)
    display_df["final_price"] = display_df["final_price"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
    display_df["rounds"] = display_df["rounds"].fillna(0).astype(int)
    display_df.columns = column_map
    return display_df


def get_deals(df):
    return df[df["deal_reached"] == True].copy()


# ==================== SECTION 1: OVERVIEW ====================


def show_overview(df, title):
    print_header(title)
    print(f"Total Sessions: {len(df)}")
    print(f"Successful Deals: {df['deal_reached'].sum()}")
    print(f"Failed Negotiations: {df['deal_failed'].sum()}")
    print(f"Incomplete: {(~df['deal_reached'] & ~df['deal_failed']).sum()}")

    print("\n📊 Configuration Breakdown:")
    print(df["Config"].value_counts().to_string())

    print("\n🎭 Role Distribution:")
    print(df["Role"].value_counts().to_string())


def show_overview_table(df):
    display_df = to_display_df(
        df,
        columns=[
            "session_id",
            "Role",
            "Config",
            "Outcome",
            "major",
            "gender",
            "negotiation_experience",
            "Duration (min)",
            "created_at",
        ],
        column_map=[
            "Session",
            "Role",
            "Config",
            "Outcome",
            "Major",
            "Gender",
            "Exp",
            "Duration",
            "Created",
        ],
    )
    print("\n" + "=" * 150)
    display(display_df)


# ==================== SECTION 2: DEMOGRAPHICS ====================


def analyze_demographics(df):
    print_header("👥 STUDENT DEMOGRAPHICS & CHARACTERISTICS")

    print_subheader("📚 Academic Major Distribution")
    if df["major"].notna().sum() > 0:
        major_counts = df["major"].value_counts()
        print(f"Total with major data: {df['major'].notna().sum()}")
        print("\nMajor breakdown:")
        for major, count in major_counts.items():
            pct = count / len(df) * 100
            print(f"  {major}: {count} ({pct:.1f}%)")
    else:
        print("No major data available")

    print_subheader("⚧ Gender Distribution")
    if df["gender"].notna().sum() > 0:
        gender_counts = df["gender"].value_counts()
        print(f"Total with gender data: {df['gender'].notna().sum()}")
        print("\nGender breakdown:")
        for gender, count in gender_counts.items():
            pct = count / len(df) * 100
            print(f"  {gender}: {count} ({pct:.1f}%)")
    else:
        print("No gender data available")

    print_subheader("💼 Negotiation Experience Distribution")
    if df["negotiation_experience"].notna().sum() > 0:
        exp_counts = df["negotiation_experience"].value_counts()
        print(f"Total with experience data: {df['negotiation_experience'].notna().sum()}")
        print("\nExperience breakdown:")
        for exp, count in exp_counts.items():
            pct = count / len(df) * 100
            print(f"  {exp}: {count} ({pct:.1f}%)")
    else:
        print("No experience data available")

    print_subheader("📈 Success Rates by Demographics")
    if df["major"].notna().sum() > 0:
        print("\nBy Major:")
        major_success = df.groupby("major")["deal_reached"].agg(["sum", "count", "mean"])
        major_success.columns = ["Deals", "Total", "Success Rate"]
        major_success["Success Rate"] = (major_success["Success Rate"] * 100).round(1)
        print(major_success.to_string())

    if df["gender"].notna().sum() > 0:
        print("\nBy Gender:")
        gender_success = df.groupby("gender")["deal_reached"].agg(["sum", "count", "mean"])
        gender_success.columns = ["Deals", "Total", "Success Rate"]
        gender_success["Success Rate"] = (gender_success["Success Rate"] * 100).round(1)
        print(gender_success.to_string())

    if df["negotiation_experience"].notna().sum() > 0:
        print("\nBy Negotiation Experience:")
        exp_success = df.groupby("negotiation_experience")["deal_reached"].agg([
            "sum",
            "count",
            "mean",
        ])
        exp_success.columns = ["Deals", "Total", "Success Rate"]
        exp_success["Success Rate"] = (exp_success["Success Rate"] * 100).round(1)
        print(exp_success.to_string())


# ==================== SECTION 3: DEAL ANALYSIS ====================


def analyze_deal_prices(df_deals):
    print_header("💰 DEAL PRICE ANALYSIS BY CONFIGURATION")

    if len(df_deals) == 0:
        print("❌ No successful deals found")
        return

    print(f"Total Successful Deals: {len(df_deals)}\n")
    print_subheader("📊 Overall Price Statistics")
    print(f"Average Final Price: ${df_deals['final_price'].mean():,.0f}")
    print(f"Median Final Price: ${df_deals['final_price'].median():,.0f}")
    print(f"Min Price: ${df_deals['final_price'].min():,.0f}")
    print(f"Max Price: ${df_deals['final_price'].max():,.0f}")
    print(f"Std Dev: ${df_deals['final_price'].std():,.0f}")

    print_subheader("🔧 Average Price by Configuration")
    config_stats = df_deals.groupby("Config").agg(
        {"final_price": ["count", "mean", "median", "std", "min", "max"]}
    ).round(0)

    print("\nConfiguration Summary:")
    print("=" * 80)
    for config in ["Base", "M", "P", "M+P"]:
        if config in config_stats.index:
            stats = config_stats.loc[config, "final_price"]
            print(f"\n{config}:")
            print(f"  Count: {int(stats['count'])} deals")
            print(f"  Average: ${stats['mean']:,.0f}")
            print(f"  Median:  ${stats['median']:,.0f}")
            print(f"  Range:   ${stats['min']:,.0f} - ${stats['max']:,.0f}")
            if stats["count"] > 1:
                print(f"  Std Dev: ${stats['std']:,.0f}")

    print_subheader("🎭 Average Price by Role & Configuration")
    role_config_stats = df_deals.groupby(["Role", "Config"])["final_price"].agg(
        ["count", "mean", "median"]
    ).round(0)

    print("\nWhen Student is BUYER (Fred Starr):")
    print("  Goal: Minimize price (lower is better)")
    print("  BATNA: $675,000 (will walk away if price > $675k)")
    print("-" * 80)
    if "🏢 Buyer (Fred)" in role_config_stats.index.get_level_values(0):
        buyer_stats = role_config_stats.loc["🏢 Buyer (Fred)"]
        for config in buyer_stats.index:
            stats = buyer_stats.loc[config]
            print(f"  {config}: ${stats['mean']:,.0f} avg (n={int(stats['count'])})")

    print("\nWhen Student is SELLER (Rosalind Cain):")
    print("  Goal: Maximize price (higher is better)")
    print("  BATNA: $475,000 (will walk away if price < $475k)")
    print("-" * 80)
    if "🏘️ Seller (Rosalind)" in role_config_stats.index.get_level_values(0):
        seller_stats = role_config_stats.loc["🏘️ Seller (Rosalind)"]
        for config in seller_stats.index:
            stats = seller_stats.loc[config]
            print(f"  {config}: ${stats['mean']:,.0f} avg (n={int(stats['count'])})")

    print_subheader("📋 All Deals - Detailed List")
    deal_display = df_deals[[
        "session_id",
        "Role",
        "Config",
        "final_price",
        "rounds",
        "Duration (min)",
        "created_at",
    ]].copy()

    deal_display["session_id"] = deal_display["session_id"].str[:8]
    deal_display["final_price"] = deal_display["final_price"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
    deal_display["Duration (min)"] = deal_display["Duration (min)"].round(1)
    deal_display["created_at"] = pd.to_datetime(deal_display["created_at"]).dt.strftime(
        "%m-%d %H:%M"
    )
    deal_display.columns = [
        "Session",
        "Student Role",
        "Config",
        "Final Price",
        "Rounds",
        "Duration (min)",
        "Created",
    ]
    display(deal_display)


def show_price_comparison(df_deals):
    if len(df_deals) == 0:
        print("No deal data available for comparison")
        return

    print("\n" + "=" * 100)
    print("COMPREHENSIVE PRICE COMPARISON: BASE vs M vs M+P".center(100))
    print("=" * 100 + "\n")

    comparison_data = []
    for config in ["Base", "M", "P", "M+P"]:
        config_data = df_deals[df_deals["Config"] == config]
        if len(config_data) > 0:
            comparison_data.append(
                {
                    "Configuration": config,
                    "N Deals": len(config_data),
                    "Avg Price": f"${config_data['final_price'].mean():,.0f}",
                    "Median Price": f"${config_data['final_price'].median():,.0f}",
                    "Min Price": f"${config_data['final_price'].min():,.0f}",
                    "Max Price": f"${config_data['final_price'].max():,.0f}",
                    "Avg Rounds": f"{config_data['rounds'].mean():.1f}",
                    "Avg Duration": f"{config_data['Duration (min)'].mean():.1f} min",
                }
            )

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        display(comparison_df)

        print("\n💡 Key Insights:")
        print("-" * 80)
        for config in ["Base", "M", "P", "M+P"]:
            config_data = df_deals[df_deals["Config"] == config]
            if len(config_data) > 0:
                buyer_avg = config_data[config_data["student_role"] == "side1"][
                    "final_price"
                ].mean()
                seller_avg = config_data[config_data["student_role"] == "side2"][
                    "final_price"
                ].mean()

                print(f"\n{config}:")
                if not pd.isna(buyer_avg):
                    print(f"  When student is buyer: ${buyer_avg:,.0f} (student wants lower)")
                if not pd.isna(seller_avg):
                    print(f"  When student is seller: ${seller_avg:,.0f} (student wants higher)")


# ==================== SECTION 3.2: WHO GOES FIRST ====================


def analyze_who_goes_first(df_deals):
    if len(df_deals) == 0:
        return

    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS: IMPACT OF WHO GOES FIRST".center(100))
    print("=" * 100 + "\n")

    for config in ["Base", "M", "P", "M+P"]:
        config_data = df_deals[df_deals["Config"] == config]

        if len(config_data) == 0:
            continue

        print(f"\n{'─'*100}")
        print(f"📊 {config} Configuration")
        print(f"{'─'*100}")

        student_first = config_data[config_data["student_goes_first"] == True]
        ai_first = config_data[config_data["student_goes_first"] == False]

        print(f"\n🎯 When STUDENT goes first (n={len(student_first)}):")
        if len(student_first) > 0:
            print(f"   Average Price: ${student_first['final_price'].mean():,.0f}")
            print(f"   Median Price:  ${student_first['final_price'].median():,.0f}")
            print(
                f"   Range: ${student_first['final_price'].min():,.0f} - ${student_first['final_price'].max():,.0f}"
            )

            buyer_student_first = student_first[student_first["student_role"] == "side1"]
            seller_student_first = student_first[student_first["student_role"] == "side2"]

            if len(buyer_student_first) > 0:
                print(
                    f"     → As Buyer (n={len(buyer_student_first)}): ${buyer_student_first['final_price'].mean():,.0f} avg"
                )
            if len(seller_student_first) > 0:
                print(
                    f"     → As Seller (n={len(seller_student_first)}): ${seller_student_first['final_price'].mean():,.0f} avg"
                )
        else:
            print("   No data")

        print(f"\n🤖 When AI goes first (n={len(ai_first)}):")
        if len(ai_first) > 0:
            print(f"   Average Price: ${ai_first['final_price'].mean():,.0f}")
            print(f"   Median Price:  ${ai_first['final_price'].median():,.0f}")
            print(
                f"   Range: ${ai_first['final_price'].min():,.0f} - ${ai_first['final_price'].max():,.0f}"
            )

            buyer_ai_first = ai_first[ai_first["student_role"] == "side1"]
            seller_ai_first = ai_first[ai_first["student_role"] == "side2"]

            if len(buyer_ai_first) > 0:
                print(
                    f"     → As Buyer (n={len(buyer_ai_first)}): ${buyer_ai_first['final_price'].mean():,.0f} avg"
                )
            if len(seller_ai_first) > 0:
                print(
                    f"     → As Seller (n={len(seller_ai_first)}): ${seller_ai_first['final_price'].mean():,.0f} avg"
                )
        else:
            print("   No data")

        if len(student_first) > 0 and len(ai_first) > 0:
            diff = student_first["final_price"].mean() - ai_first["final_price"].mean()
            print(
                f"\n💡 Impact: Student going first vs AI going first = ${abs(diff):,.0f} difference"
            )
            if diff > 0:
                print("   → Prices are HIGHER when student goes first")
            elif diff < 0:
                print("   → Prices are LOWER when student goes first")
            else:
                print("   → No difference")

    summary_data = []
    for config in ["Base", "M", "P", "M+P"]:
        config_data = df_deals[df_deals["Config"] == config]
        if len(config_data) == 0:
            continue

        student_first = config_data[config_data["student_goes_first"] == True]
        ai_first = config_data[config_data["student_goes_first"] == False]

        summary_data.append(
            {
                "Config": config,
                "Student First (n)": len(student_first),
                "Student First Avg": f"${student_first['final_price'].mean():,.0f}"
                if len(student_first) > 0
                else "N/A",
                "AI First (n)": len(ai_first),
                "AI First Avg": f"${ai_first['final_price'].mean():,.0f}"
                if len(ai_first) > 0
                else "N/A",
                "Difference": f"${abs(student_first['final_price'].mean() - ai_first['final_price'].mean()):,.0f}"
                if len(student_first) > 0 and len(ai_first) > 0
                else "N/A",
            }
        )

    if summary_data:
        print("\n" + "=" * 100)
        print("SUMMARY TABLE: WHO GOES FIRST IMPACT".center(100))
        print("=" * 100 + "\n")
        summary_df = pd.DataFrame(summary_data)
        display(summary_df)


# ==================== SECTION 4: MEMORY EVOLUTION ====================


def analyze_memory(conn, df_filtered):
    print_header("🧠 AI MEMORY EVOLUTION TRACKING")

    memory_sessions = df_filtered[df_filtered["use_memory"] == True]["session_id"].tolist()
    if len(memory_sessions) == 0:
        print("❌ No sessions with memory in filtered data")
        return

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
      AND deal_reached = 1
    ORDER BY created_at DESC
    """

    df_memory = pd.read_sql_query(query_memory, conn)
    if len(df_memory) == 0:
        print("❌ No sessions with memory history data")
        return

    print(f"Found {len(df_memory)} sessions with Memory tracking (deals only)\n")

    print("=" * 150)
    print("📋 All Sessions with Memory (M) - Choose one to analyze")
    print("=" * 150)
    print(
        f"{'IDX':>4} | {'SESSION':>10} | {'CFG':>4} | {'ROLE':>6} | {'STATUS':>10} | {'PRICE':>10} | {'TIME':>12} | {'UPDATES':>7}"
    )
    print("-" * 150)

    for idx, row in df_memory.iterrows():
        config = get_config_label(row["use_memory"], row["use_plan"])
        status = "✅ Deal" if row["deal_reached"] else "❌ No Deal"
        role = "Buyer" if row["student_role"] == "side1" else "Seller"
        created = pd.to_datetime(row["created_at"]).strftime("%m-%d %H:%M")

        memory_history = safe_json_load(row["ai_memory_history"])
        updates = len(memory_history) if memory_history else 0

        deal_json = safe_json_load(row["student_deal_json"])
        price = deal_json.get("final_price") if deal_json else None
        price_str = f"${price:,}" if price else "N/A"

        print(
            f"{idx:>4} | {row['session_id'][:10]:>10} | {config:>4} | {role:>6} | {status:>10} | {price_str:>10} | {created:>12} | {updates:>7}"
        )

    print("=" * 150)
    print("\n⚠️  Set session_idx below to analyze a specific session\n")

    session_idx = SESSION_IDX_MEMORY
    if session_idx >= len(df_memory):
        print(f"❌ Invalid session_idx: {session_idx}. Max index is {len(df_memory)-1}")
        return

    session = df_memory.iloc[session_idx]
    print("\n" + "=" * 120)
    print(f"📌 Analyzing Session: {session['session_id'][:8]}...")
    print("=" * 120)
    print(
        f"   Student Role: {'Buyer (Fred)' if session['student_role'] == 'side1' else 'Seller (Rosalind)'}"
    )
    print(f"   Configuration: {get_config_label(session['use_memory'], session['use_plan'])}")
    print(f"   Deal Reached: {'✅ Yes' if session['deal_reached'] else '❌ No'}")

    deal_json = safe_json_load(session["student_deal_json"])
    if deal_json and "final_price" in deal_json:
        print(f"   Final Price: ${deal_json['final_price']:,}")

    memory_history = safe_json_load(session["ai_memory_history"])
    if not memory_history:
        print("❌ No memory history data found for this session")
        return

    print(f"   Total Memory Updates: {len(memory_history)}\n")
    print("=" * 120)
    print("MEMORY EVOLUTION (Round-by-Round)")
    print("=" * 120)

    for i, memory_state in enumerate(memory_history):
        round_info = memory_state.get("round", f"Update {i+1}")
        content = memory_state.get("content", "")

        print(f"\n{'─'*120}")
        print(f"🧠 MEMORY STATE #{i+1} | Round {round_info}")
        print(f"{'─'*120}")
        print(content)

    print(f"\n\n{'='*120}")
    print("🎯 FINAL MEMORY STATE")
    print(f"{'='*120}")
    current_memory = session["ai_memory"]
    print(current_memory if current_memory else "(empty)")
    print(f"{'='*120}")

    transcript = safe_json_load(session["transcript"])
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


# ==================== SECTION 5: PLAN EVOLUTION ====================


def analyze_plan(conn, df_filtered):
    print_header("🎯 AI PLANNING STRATEGY TRACKING")

    plan_sessions = df_filtered[df_filtered["use_plan"] == True]["session_id"].tolist()
    if len(plan_sessions) == 0:
        print("❌ No sessions with planning in filtered data")
        return

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
      AND deal_reached = 1
    ORDER BY created_at DESC
    """

    df_plan = pd.read_sql_query(query_plan, conn)
    if len(df_plan) == 0:
        print("❌ No sessions with plan history data")
        return

    print(f"Found {len(df_plan)} sessions with Planning tracking (deals only)\n")

    print("=" * 150)
    print("📋 All Sessions with Planning (P) - Choose one to analyze")
    print("=" * 150)
    print(
        f"{'IDX':>4} | {'SESSION':>10} | {'CFG':>4} | {'ROLE':>6} | {'STATUS':>10} | {'PRICE':>10} | {'TIME':>12} | {'UPDATES':>7}"
    )
    print("-" * 150)

    for idx, row in df_plan.iterrows():
        config = get_config_label(row["use_memory"], row["use_plan"])
        status = "✅ Deal" if row["deal_reached"] else "❌ No Deal"
        role = "Buyer" if row["student_role"] == "side1" else "Seller"
        created = pd.to_datetime(row["created_at"]).strftime("%m-%d %H:%M")

        plan_history = safe_json_load(row["ai_plan_history"])
        updates = len(plan_history) if plan_history else 0

        deal_json = safe_json_load(row["student_deal_json"])
        price = deal_json.get("final_price") if deal_json else None
        price_str = f"${price:,}" if price else "N/A"

        print(
            f"{idx:>4} | {row['session_id'][:10]:>10} | {config:>4} | {role:>6} | {status:>10} | {price_str:>10} | {created:>12} | {updates:>7}"
        )

    print("=" * 150)
    print("\n⚠️  Set session_idx below to analyze a specific session\n")

    session_idx = SESSION_IDX_PLAN
    if session_idx >= len(df_plan):
        print(f"❌ Invalid session_idx: {session_idx}. Max index is {len(df_plan)-1}")
        return

    session = df_plan.iloc[session_idx]
    print("\n" + "=" * 120)
    print(f"📌 Analyzing Session: {session['session_id'][:8]}...")
    print("=" * 120)
    print(
        f"   Student Role: {'Buyer (Fred)' if session['student_role'] == 'side1' else 'Seller (Rosalind)'}"
    )
    print(f"   Configuration: {get_config_label(session['use_memory'], session['use_plan'])}")
    print(f"   Deal Reached: {'✅ Yes' if session['deal_reached'] else '❌ No'}")

    deal_json = safe_json_load(session["student_deal_json"])
    if deal_json and "final_price" in deal_json:
        print(f"   Final Price: ${deal_json['final_price']:,}")

    plan_history = safe_json_load(session["ai_plan_history"])
    if not plan_history:
        print("❌ No plan history data found for this session")
        return

    print(f"   Total Plan Updates: {len(plan_history)}\n")
    print("=" * 120)
    print("PLANNING EVOLUTION (Round-by-Round)")
    print("=" * 120)

    for i, plan_state in enumerate(plan_history):
        round_info = plan_state.get("round", f"Update {i+1}")
        content = plan_state.get("content", "")

        print(f"\n{'─'*120}")
        print(f"🎯 PLAN STATE #{i+1} | Round {round_info}")
        print(f"{'─'*120}")
        print(content)

    print(f"\n\n{'='*120}")
    print("🏁 FINAL PLAN STATE")
    print(f"{'='*120}")
    current_plan = session["ai_plan"]
    print(current_plan if current_plan else "(empty)")
    print(f"{'='*120}")

    transcript = safe_json_load(session["transcript"])
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


# ==================== SECTION 6: TRANSCRIPT VIEWER ====================


def view_transcript(df_deals):
    print_header("💬 FULL CONVERSATION TRANSCRIPT")

    if len(df_deals) == 0:
        print("❌ No deal data available. Please run the deal analysis first.")
        return

    print(f"Found {len(df_deals)} successful deals\n")
    print("=" * 120)
    print("📋 All Successful Deals - Choose one to view transcript")
    print("=" * 120)

    for idx, row in df_deals.iterrows():
        price = row["final_price"]
        config = row["Config"]
        role = row["Role"]
        created = pd.to_datetime(row["created_at"]).strftime("%m-%d %H:%M")
        rounds = row["rounds"]
        duration = row["Duration (min)"]

        print(
            f"{idx:3d}. {row['session_id'][:8]}... | {role:17s} | {config:4s} | ${price:>7,.0f} | {rounds} rounds | {duration:.1f} min | {created}"
        )

    print("=" * 120)
    print("\n⚠️  Set session_to_view below to view a specific transcript\n")

    session_to_view = SESSION_IDX_TRANSCRIPT
    if session_to_view >= len(df_deals):
        print(
            f"❌ Invalid session_to_view: {session_to_view}. Max index is {len(df_deals)-1}"
        )
        return

    session = df_deals.iloc[session_to_view]

    print("\n" + "=" * 120)
    print(f"📌 Viewing Transcript: {session['session_id'][:8]}...")
    print("=" * 120)
    print(f"   Student Role: {session['Role']}")
    print(
        f"   AI Role: {'Buyer (Fred)' if session['ai_role'] == 'side1' else 'Seller (Rosalind)'}"
    )
    print(f"   Configuration: {session['Config']}")
    print(f"   Final Price: ${session['final_price']:,.0f}")
    print(f"   Rounds: {session['rounds']}")
    print(f"   Duration: {session['Duration (min)']:.1f} minutes\n")

    transcript = safe_json_load(session["transcript"])
    if not transcript:
        print("❌ No transcript data available for this session")
        return

    print("=" * 120)
    print("CONVERSATION TRANSCRIPT")
    print("=" * 120)

    for i, message in enumerate(transcript):
        if " - " in message:
            parts = message.split(" - ", 1)
            round_info = parts[0]

            if len(parts) > 1 and ": " in parts[1]:
                label, content = parts[1].split(": ", 1)

                if "Fred" in label or "Buyer" in label:
                    icon = "🏢"
                elif "Rosalind" in label or "Seller" in label:
                    icon = "🏘️"
                else:
                    icon = "💬"

                print(f"\n{icon} {round_info} - {label}")
                print("─" * 120)
                print(content)
            else:
                print(f"\n💬 {round_info}")
                print("─" * 120)
                print(parts[1] if len(parts) > 1 else message)
        else:
            print(f"\n💬 Message {i+1}")
            print("─" * 120)
            print(message)

    print("\n" + "=" * 120)
    print(f"✅ Deal reached at price: ${session['final_price']:,.0f}")
    print("=" * 120)


# ==================== SECTION 7: PERFORMANCE METRICS ====================


def performance_metrics(df_all, df_deals):
    print_header("📊 PERFORMANCE METRICS DASHBOARD")

    print_subheader("✅ Success Rates by Configuration")
    success_by_config = df_all.groupby("Config").agg({"deal_reached": ["sum", "count", "mean"]})
    success_by_config.columns = ["Successful", "Total", "Success Rate"]
    success_by_config["Success Rate"] = (success_by_config["Success Rate"] * 100).round(1)
    print(success_by_config.to_string())

    if len(df_deals) > 0:
        print_subheader("⏱️ Duration Analysis (Successful Deals Only)")
        duration_by_config = df_deals.groupby("Config")["Duration (min)"].agg(
            ["count", "mean", "median", "min", "max"]
        )
        duration_by_config = duration_by_config.round(1)
        duration_by_config.columns = ["N", "Mean (min)", "Median (min)", "Min (min)", "Max (min)"]
        print(duration_by_config.to_string())

        print_subheader("🔄 Rounds Analysis (Successful Deals Only)")
        rounds_by_config = df_deals.groupby("Config")["rounds"].agg(
            ["count", "mean", "median", "min", "max"]
        )
        rounds_by_config = rounds_by_config.round(1)
        rounds_by_config.columns = ["N", "Mean", "Median", "Min", "Max"]
        print(rounds_by_config.to_string())

        print_subheader("⚡ Efficiency: Minutes per Round")
        df_deals["min_per_round"] = df_deals["Duration (min)"] / df_deals["rounds"]
        efficiency = df_deals.groupby("Config")["min_per_round"].agg(["mean", "median"]).round(2)
        efficiency.columns = ["Mean (min/round)", "Median (min/round)"]
        print(efficiency.to_string())

        print_subheader("💰 Price Distribution by Student Role")
        print("\nWhen Student is BUYER (wants lower price):")
        buyer_deals = df_deals[df_deals["student_role"] == "side1"]
        if len(buyer_deals) > 0:
            buyer_stats = buyer_deals.groupby("Config")["final_price"].agg(
                ["count", "mean", "median"]
            )
            buyer_stats["mean"] = buyer_stats["mean"].apply(lambda x: f"${x:,.0f}")
            buyer_stats["median"] = buyer_stats["median"].apply(lambda x: f"${x:,.0f}")
            buyer_stats.columns = ["N", "Mean Price", "Median Price"]
            print(buyer_stats.to_string())
        else:
            print("  No data")

        print("\nWhen Student is SELLER (wants higher price):")
        seller_deals = df_deals[df_deals["student_role"] == "side2"]
        if len(seller_deals) > 0:
            seller_stats = seller_deals.groupby("Config")["final_price"].agg(
                ["count", "mean", "median"]
            )
            seller_stats["mean"] = seller_stats["mean"].apply(lambda x: f"${x:,.0f}")
            seller_stats["median"] = seller_stats["median"].apply(lambda x: f"${x:,.0f}")
            seller_stats.columns = ["N", "Mean Price", "Median Price"]
            print(seller_stats.to_string())
        else:
            print("  No data")

    print("\n" + "=" * 100)


# ==================== SECTION 8: TIME WINDOW ANALYSIS ====================


def analyze_time_period(df, period_name, period_label, seller_only=False):
    print_header(period_label)

    if len(df) == 0:
        print(f"❌ No data for {period_name}")
        return

    print(f"Total Sessions: {len(df)}")
    print(f"Successful Deals: {df['deal_reached'].sum()} ({df['deal_reached'].sum()/len(df)*100:.1f}%)")
    print(f"Failed Negotiations: {df['deal_failed'].sum()}")
    print(f"Incomplete: {(~df['deal_reached'] & ~df['deal_failed']).sum()}")

    print("\n📊 Configuration Breakdown:")
    print(df["Config"].value_counts().to_string())

    if seller_only:
        print("\n🎯 Who Goes First:")
        print(df["student_goes_first"].map({True: "Student First", False: "AI First"}).value_counts().to_string())
    else:
        print("\n🎭 Role Distribution:")
        print(df["Role"].value_counts().to_string())

    print_subheader("👥 Demographics")
    if df["major"].notna().sum() > 0:
        print("\n📚 Major:")
        for major, count in df["major"].value_counts().items():
            print(f"  {major}: {count}")

    if df["gender"].notna().sum() > 0:
        print("\n⚧ Gender:")
        for gender, count in df["gender"].value_counts().items():
            print(f"  {gender}: {count}")

    if df["negotiation_experience"].notna().sum() > 0:
        print("\n💼 Experience:")
        for exp, count in df["negotiation_experience"].value_counts().items():
            print(f"  {exp}: {count}")

    df_deals = get_deals(df)
    if len(df_deals) > 0:
        print_subheader("💰 Deal Price Analysis")

        valid_prices = df_deals["final_price"].dropna()
        if len(valid_prices) > 0:
            print("\n📊 Overall Price Statistics:")
            print(f"  Average: ${valid_prices.mean():,.0f}")
            print(f"  Median:  ${valid_prices.median():,.0f}")
            print(f"  Min:     ${valid_prices.min():,.0f}")
            print(f"  Max:     ${valid_prices.max():,.0f}")
            print(f"  Std Dev: ${valid_prices.std():,.0f}")

            print("\n🔧 By Configuration:")
            for config in ["Base", "M", "P", "M+P"]:
                config_deals = df_deals[df_deals["Config"] == config]
                config_prices = config_deals["final_price"].dropna()
                if len(config_prices) > 0:
                    print(f"\n  {config}:")
                    print(f"    Count: {len(config_prices)}")
                    print(f"    Average: ${config_prices.mean():,.0f}")
                    print(f"    Median:  ${config_prices.median():,.0f}")
                    print(
                        f"    Range: ${config_prices.min():,.0f} - ${config_prices.max():,.0f}"
                    )

            if not seller_only:
                print("\n🎭 By Student Role:")
                buyer_deals = df_deals[df_deals["student_role"] == "side1"]
                buyer_prices = buyer_deals["final_price"].dropna()
                if len(buyer_prices) > 0:
                    print("\n  When Student is BUYER (Fred):")
                    print("    Goal: Minimize price (lower is better)")
                    print(f"    Count: {len(buyer_prices)}")
                    print(f"    Average: ${buyer_prices.mean():,.0f}")
                    print(
                        f"    Range: ${buyer_prices.min():,.0f} - ${buyer_prices.max():,.0f}"
                    )

                seller_deals = df_deals[df_deals["student_role"] == "side2"]
                seller_prices = seller_deals["final_price"].dropna()
                if len(seller_prices) > 0:
                    print("\n  When Student is SELLER (Rosalind):")
                    print("    Goal: Maximize price (higher is better)")
                    print(f"    Count: {len(seller_prices)}")
                    print(f"    Average: ${seller_prices.mean():,.0f}")
                    print(
                        f"    Range: ${seller_prices.min():,.0f} - ${seller_prices.max():,.0f}"
                    )

        print_subheader("⏱️ Duration & Rounds Analysis")
        valid_durations = df_deals["Duration (min)"].dropna()
        if len(valid_durations) > 0:
            print("\nDuration Statistics:")
            print(f"  Average: {valid_durations.mean():.1f} minutes")
            print(f"  Median:  {valid_durations.median():.1f} minutes")
            print(f"  Range: {valid_durations.min():.1f} - {valid_durations.max():.1f} minutes")

        valid_rounds = df_deals["rounds"]
        if len(valid_rounds) > 0:
            print("\nRounds Statistics:")
            print(f"  Average: {valid_rounds.mean():.1f} rounds")
            print(f"  Median:  {valid_rounds.median():.0f} rounds")
            print(f"  Range: {valid_rounds.min():.0f} - {valid_rounds.max():.0f} rounds")

    print_subheader("📋 Detailed Session List")
    detail_columns = [
        "session_id",
        "Config",
        "Outcome",
        "major",
        "rounds",
        "final_price",
        "Duration (min)",
        "created_at",
    ]
    if not seller_only:
        detail_columns.insert(1, "Role")

    detail_df = df[detail_columns].copy()
    detail_df["session_id"] = detail_df["session_id"].str[:8]
    detail_df["created_at"] = pd.to_datetime(detail_df["created_at"]).dt.strftime("%H:%M")
    detail_df["Duration (min)"] = detail_df["Duration (min)"].round(1)
    detail_df["final_price"] = detail_df["final_price"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )

    if seller_only:
        detail_df.columns = [
            "Session",
            "Config",
            "Outcome",
            "Major",
            "Rounds",
            "Price",
            "Duration",
            "Time",
        ]
    else:
        detail_df.columns = [
            "Session",
            "Role",
            "Config",
            "Outcome",
            "Major",
            "Rounds",
            "Price",
            "Duration",
            "Time",
        ]

    display(detail_df)


def analyze_time_windows(df):
    df = add_time_periods(df, TIME_WINDOWS)
    df = df[df["time_period"].isin(TIME_WINDOWS.keys())].copy()

    print_header("📊 OVERALL STATISTICS (Time Windows)")
    print(f"Total Sessions: {len(df)}")
    print(f"  Morning: {(df['time_period'] == 'Morning').sum()}")
    print(f"  Afternoon: {(df['time_period'] == 'Afternoon').sum()}")

    print(f"\nSuccessful Deals: {df['deal_reached'].sum()}")
    print(f"Failed Negotiations: {df['deal_failed'].sum()}")
    print(f"Incomplete: {(~df['deal_reached'] & ~df['deal_failed']).sum()}")

    print("\n📊 Configuration Breakdown:")
    print(df["Config"].value_counts().to_string())

    print("\n🎭 Role Distribution:")
    print(df["Role"].value_counts().to_string())

    print("\n" + "=" * 170)
    print("ALL FILTERED DATA")
    print("=" * 170)

    display_df = to_display_df(
        df,
        columns=[
            "session_id",
            "time_period",
            "Role",
            "Config",
            "Outcome",
            "major",
            "gender",
            "negotiation_experience",
            "rounds",
            "final_price",
            "Duration (min)",
            "created_at",
        ],
        column_map=[
            "Session",
            "Period",
            "Role",
            "Config",
            "Outcome",
            "Major",
            "Gender",
            "Exp",
            "Rounds",
            "Final Price",
            "Duration",
            "Created",
        ],
        created_format="%m-%d %H:%M",
    )
    display(display_df)

    df_morning = df[df["time_period"] == "Morning"].copy()
    analyze_time_period(df_morning, "Morning", "🌅 MORNING SESSION (14:36-15:48)")

    df_afternoon = df[df["time_period"] == "Afternoon"].copy()
    analyze_time_period(df_afternoon, "Afternoon", "🌆 AFTERNOON SESSION (19:21-20:01)")

    print_header("⚖️ MORNING vs AFTERNOON COMPARISON")
    comparison_data = []

    for period_name, df_period in [("Morning", df_morning), ("Afternoon", df_afternoon)]:
        deals = get_deals(df_period)
        prices = deals["final_price"].dropna()
        comparison_data.append(
            {
                "Period": period_name,
                "Total Sessions": len(df_period),
                "Deals": len(deals),
                "Success Rate": f"{len(deals)/len(df_period)*100:.1f}%" if len(df_period) > 0 else "N/A",
                "Avg Price": f"${prices.mean():,.0f}" if len(prices) > 0 else "N/A",
                "Avg Rounds": f"{deals['rounds'].mean():.1f}" if len(deals) > 0 else "N/A",
                "Avg Duration": f"{deals['Duration (min)'].mean():.1f} min" if len(deals) > 0 else "N/A",
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    display(comparison_df)


# ==================== SECTION 9: SELLER-ONLY ANALYSIS ====================


def seller_only_analysis(df_all):
    print_header("📊 OVERALL STATISTICS - SELLER ONLY")

    df_seller = apply_filters(
        df_all,
        excluded_majors=EXCLUDED_MAJORS,
        role="side2",
        time_windows=TIME_WINDOWS,
    )
    df_seller = add_derived_columns(df_seller)

    print(f"Total Seller Sessions: {len(df_seller)}")
    print(f"  Morning: {(df_seller['time_period'] == 'Morning').sum()}")
    print(f"  Afternoon: {(df_seller['time_period'] == 'Afternoon').sum()}")
    print(f"\nSuccessful Deals: {df_seller['deal_reached'].sum()} ({df_seller['deal_reached'].sum()/len(df_seller)*100:.1f}%)")
    print(f"Failed Negotiations: {df_seller['deal_failed'].sum()}")
    print(f"Incomplete: {(~df_seller['deal_reached'] & ~df_seller['deal_failed']).sum()}")

    print("\n📊 Configuration Breakdown:")
    print(df_seller["Config"].value_counts().to_string())

    print("\n🎯 Who Goes First:")
    print(df_seller["student_goes_first"].map({True: "Student First", False: "AI First"}).value_counts().to_string())

    print("\n" + "=" * 170)
    print("ALL FILTERED DATA - SELLER ONLY")
    print("=" * 170)

    display_df = df_seller[[
        "session_id",
        "time_period",
        "Config",
        "Outcome",
        "major",
        "gender",
        "negotiation_experience",
        "student_goes_first",
        "rounds",
        "final_price",
        "Duration (min)",
        "created_at",
    ]].copy()

    display_df["session_id"] = display_df["session_id"].str[:8]
    display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime("%m-%d %H:%M")
    display_df["Duration (min)"] = display_df["Duration (min)"].round(1)
    display_df["final_price"] = display_df["final_price"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
    display_df["rounds"] = display_df["rounds"].fillna(0).astype(int)
    display_df["student_goes_first"] = display_df["student_goes_first"].map({True: "👤", False: "🤖"})

    display_df.columns = [
        "Session",
        "Period",
        "Config",
        "Outcome",
        "Major",
        "Gender",
        "Exp",
        "First",
        "Rounds",
        "Final Price",
        "Duration",
        "Created",
    ]

    display(display_df)

    df_morning = df_seller[df_seller["time_period"] == "Morning"].copy()
    analyze_time_period(
        df_morning,
        "Morning",
        "🌅 MORNING SESSION - SELLER ONLY (14:36-15:48)",
        seller_only=True,
    )

    df_afternoon = df_seller[df_seller["time_period"] == "Afternoon"].copy()
    analyze_time_period(
        df_afternoon,
        "Afternoon",
        "🌆 AFTERNOON SESSION - SELLER ONLY (19:21-20:01)",
        seller_only=True,
    )

    print_header("⚖️ MORNING vs AFTERNOON COMPARISON - SELLER ONLY")
    comparison_data = []

    for period_name, df_period in [("Morning", df_morning), ("Afternoon", df_afternoon)]:
        deals = get_deals(df_period)
        prices = deals["final_price"].dropna()

        comparison_data.append(
            {
                "Period": period_name,
                "Total Sessions": len(df_period),
                "Deals": len(deals),
                "Success Rate": f"{len(deals)/len(df_period)*100:.1f}%" if len(df_period) > 0 else "N/A",
                "Avg Price": f"${prices.mean():,.0f}" if len(prices) > 0 else "N/A",
                "Avg Rounds": f"{deals['rounds'].mean():.1f}" if len(deals) > 0 else "N/A",
                "Avg Duration": f"{deals['Duration (min)'].mean():.1f} min" if len(deals) > 0 else "N/A",
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    display(comparison_df)

    print("\n✅ Seller-only analysis complete!")
    print("📈 Remember: As SELLER, higher prices indicate better negotiation outcomes")


# ==================== MAIN EXECUTION ====================


def main():
    configure_display()

    conn = sqlite3.connect(DB_PATH)
    print(f"✅ Connected to: {DB_PATH}")
    print(f"🎯 Analyzing scenario: {SCENARIO_FILTER}")

    df_all = load_sessions(conn, SCENARIO_FILTER)
    df_all = apply_filters(df_all, cutoff_time=CUTOFF_TIME, excluded_majors=EXCLUDED_MAJORS)
    df_all = add_derived_columns(df_all)

    show_overview(df_all, "📋 ALL MAIN STREET NEGOTIATIONS")
    show_overview_table(df_all)

    analyze_demographics(df_all)

    df_deals = get_deals(df_all)
    analyze_deal_prices(df_deals)
    show_price_comparison(df_deals)
    analyze_who_goes_first(df_deals)

    analyze_memory(conn, df_all)
    analyze_plan(conn, df_all)
    view_transcript(df_deals)

    performance_metrics(df_all, df_deals)
    analyze_time_windows(df_all)
    seller_only_analysis(df_all)

    conn.close()
    print("✅ Database connection closed")


if __name__ == "__main__":
    main()
