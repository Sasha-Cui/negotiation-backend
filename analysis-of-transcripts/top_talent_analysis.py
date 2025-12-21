# 🎯 Top Talent Negotiation - Comprehensive Analysis Dashboard
"""
Complete analysis of Top Talent negotiations (Recruiter vs Candidate)

Analysis Sections:
1. 📋 All Negotiation Records Overview
2. 📈 Total Pie & Human Share Distributions
3. 🧠 AI vs Human Performance Tests
4. 👥 Student Demographics & Characteristics Analysis
5. 🎭 Treatment Variations (Role & First Mover)
"""

# ==================== SETUP ====================
import importlib.util
import json
import sqlite3
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from TTverifier_and_scorer import verify_and_score_top_talent

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "negotiations.db"
SCENARIO_FILTER = "Top_talent"
CUTOFF_TIME = "2025-12-10 14:36"
EXCLUDED_MAJORS = {"sds"}

OUTPUT_TOP_TALENT_CSV = BASE_DIR / "top_talent_sessions.csv"
OUTPUT_MAIN_STREET_CSV = BASE_DIR / "main_street_sessions.csv"

# ==================== OPTIONAL STATS ====================
scipy_stats = None
if importlib.util.find_spec("scipy") is not None:
    from scipy import stats as scipy_stats


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


def load_sessions(conn):
    query_all = """
    SELECT
        session_id,
        scenario_name,
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
    WHERE created_at >= ?
    ORDER BY created_at DESC
    """
    return pd.read_sql_query(query_all, conn, params=[CUTOFF_TIME])


def apply_filters(df, scenario_name=None, excluded_majors=None):
    df = df.copy()

    if scenario_name:
        df = df[df["scenario_name"].str.lower() == scenario_name.lower()].copy()

    if excluded_majors:
        df = df[
            ~df["major"].str.lower().isin(excluded_majors) | df["major"].isna()
        ].copy()

    return df


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
        {"side1": "🧑‍💼 Recruiter", "side2": "🧑‍🎓 Candidate"}
    )
    df["rounds"] = df["transcript"].apply(count_rounds)
    return df


def score_deal(row):
    if not row.get("deal_reached"):
        return {
            "recruiter_points": None,
            "candidate_points": None,
            "true_total_points": None,
            "verified_agreement": None,
        }

    student_deal = safe_json_load(row.get("student_deal_json")) or {}
    ai_deal = safe_json_load(row.get("ai_deal_json")) or {}

    if row.get("student_role") == "side1":
        side1_final = student_deal
        side2_final = ai_deal
    else:
        side1_final = ai_deal
        side2_final = student_deal

    try:
        result = verify_and_score_top_talent(
            side1_final=side1_final,
            side2_final=side2_final,
            deal_reached_token=row.get("deal_reached"),
        )
    except ValueError as exc:
        print(f"⚠️ Scoring failed for session {row.get('session_id')}: {exc}")
        return {
            "recruiter_points": None,
            "candidate_points": None,
            "true_total_points": None,
            "verified_agreement": None,
        }

    return {
        "recruiter_points": result.get("recruiter_points"),
        "candidate_points": result.get("candidate_points"),
        "true_total_points": result.get("true_total_points"),
        "verified_agreement": result.get("verified_agreement"),
    }


def add_scoring_columns(df):
    scored = df.apply(score_deal, axis=1, result_type="expand")
    df = pd.concat([df, scored], axis=1)

    df["student_points"] = np.where(
        df["student_role"] == "side1", df["recruiter_points"], df["candidate_points"]
    )
    df["ai_points"] = np.where(
        df["student_role"] == "side1", df["candidate_points"], df["recruiter_points"]
    )

    df["total_pie"] = df["true_total_points"]
    df["percent_pie_human"] = np.where(
        df["total_pie"] > 0, df["student_points"] / df["total_pie"], np.nan
    )
    df["percent_pie_ai"] = np.where(
        df["total_pie"] > 0, df["ai_points"] / df["total_pie"], np.nan
    )
    return df


def show_overview(df):
    print_header("📋 ALL TOP TALENT NEGOTIATIONS")
    print(f"Total Sessions: {len(df)}")
    print(f"Successful Deals: {df['deal_reached'].sum()}")
    print(f"Failed Negotiations: {df['deal_failed'].sum()}")
    print(f"Incomplete: {(~df['deal_reached'] & ~df['deal_failed']).sum()}")

    print("\n📊 Configuration Breakdown:")
    print(df["Config"].value_counts().to_string())

    print("\n🎭 Role Distribution:")
    print(df["Role"].value_counts().to_string())


def show_overview_table(df):
    display_df = df[[
        "session_id",
        "Role",
        "Config",
        "Outcome",
        "major",
        "gender",
        "negotiation_experience",
        "Duration (min)",
        "created_at",
        "total_pie",
        "percent_pie_human",
        "rounds",
    ]].copy()

    display_df["session_id"] = display_df["session_id"].str[:8]
    display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    display_df["Duration (min)"] = display_df["Duration (min)"].round(1)
    display_df["total_pie"] = display_df["total_pie"].apply(
        lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
    )
    display_df["percent_pie_human"] = display_df["percent_pie_human"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
    )
    display_df["rounds"] = display_df["rounds"].fillna(0).astype(int)

    display_df.columns = [
        "Session",
        "Role",
        "Config",
        "Outcome",
        "Major",
        "Gender",
        "Exp",
        "Duration",
        "Created",
        "Total Pie",
        "Human Share",
        "Rounds",
    ]

    print("\n" + "=" * 150)
    display(display_df)


def analyze_pie_distributions(df_deals):
    print_header("📈 TOTAL PIE & HUMAN SHARE DISTRIBUTIONS")

    if len(df_deals) == 0:
        print("❌ No successful deals found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df_deals["total_pie"].dropna(), bins=25, edgecolor="black", alpha=0.7)
    axes[0].set_title("Distribution of Total Pie Size")
    axes[0].set_xlabel("Total Points")
    axes[0].set_ylabel("Deals")

    axes[1].hist(df_deals["percent_pie_human"].dropna(), bins=20, edgecolor="black", alpha=0.7)
    axes[1].axvline(0.5, color="red", linestyle="--", linewidth=2, label="50-50 split")
    axes[1].set_title("Distribution of Human Share")
    axes[1].set_xlabel("Human % of Total Pie")
    axes[1].set_ylabel("Deals")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    summary = df_deals[["total_pie", "percent_pie_human"]].describe().T
    display(summary.round(2))


def analyze_ai_vs_human(df_deals):
    print_header("🧠 AI vs HUMAN PERFORMANCE TESTS")

    df_valid = df_deals.dropna(subset=["student_points", "ai_points", "total_pie"]).copy()
    if len(df_valid) == 0:
        print("❌ No scored deals available for comparison")
        return

    df_valid["ai_minus_human"] = df_valid["ai_points"] - df_valid["student_points"]

    print(f"Deals with scores: {len(df_valid)}")
    print(f"Average Human Points: {df_valid['student_points'].mean():,.1f}")
    print(f"Average AI Points: {df_valid['ai_points'].mean():,.1f}")
    print(f"Average (AI - Human): {df_valid['ai_minus_human'].mean():,.1f}\n")

    if scipy_stats is None:
        print("⚠️ scipy not available - skipping statistical tests.")
        return

    t_stat, p_two_sided = scipy_stats.ttest_1samp(df_valid["ai_minus_human"], 0, nan_policy="omit")
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)

    print("Paired t-test on (AI - Human) points")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  two-sided p-value: {p_two_sided:.4f}")
    print(f"  one-sided p-value (AI > Human): {p_one_sided:.4f}")


def analyze_demographics(df_deals):
    print_header("👥 STUDENT DEMOGRAPHICS & PERFORMANCE")

    def plot_distribution(series, title, xlabel):
        counts = series.dropna().value_counts()
        if counts.empty:
            print(f"No {title.lower()} data available")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    print_subheader("📚 Academic Major Distribution")
    plot_distribution(df_deals["major"], "Academic Major Distribution", "Major")

    print_subheader("⚧ Gender Distribution")
    plot_distribution(df_deals["gender"], "Gender Distribution", "Gender")

    print_subheader("💼 Negotiation Experience Distribution")
    plot_distribution(
        df_deals["negotiation_experience"],
        "Negotiation Experience Distribution",
        "Experience Level",
    )

    print_subheader("📊 Performance by Demographics")
    group_cols = ["major", "gender", "negotiation_experience"]
    for col in group_cols:
        if df_deals[col].dropna().empty:
            continue

        summary = (
            df_deals.groupby(col)
            .agg(
                n_deals=("session_id", "count"),
                avg_total_pie=("total_pie", "mean"),
                avg_human_share=("percent_pie_human", "mean"),
            )
            .sort_values("avg_total_pie", ascending=False)
        )
        print_subheader(f"Performance by {col}")
        display(summary.round(2))


def analyze_treatments(df_deals):
    print_header("🎭 TREATMENT VARIATIONS")

    if len(df_deals) == 0:
        print("❌ No successful deals found")
        return

    print_subheader("Role Breakdown")
    role_summary = df_deals.groupby("Role").agg(
        n_deals=("session_id", "count"),
        avg_total_pie=("total_pie", "mean"),
        avg_human_share=("percent_pie_human", "mean"),
    )
    display(role_summary.round(2))

    print_subheader("First Mover Breakdown")
    mover_summary = df_deals.groupby("student_goes_first").agg(
        n_deals=("session_id", "count"),
        avg_total_pie=("total_pie", "mean"),
        avg_human_share=("percent_pie_human", "mean"),
    )
    mover_summary.index = mover_summary.index.map({True: "Student First", False: "AI First"})
    display(mover_summary.round(2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    role_summary["avg_total_pie"].plot(kind="bar", ax=axes[0], title="Avg Total Pie by Role")
    axes[0].set_ylabel("Total Points")

    mover_summary["avg_total_pie"].plot(kind="bar", ax=axes[1], title="Avg Total Pie by First Mover")
    axes[1].set_ylabel("Total Points")

    plt.tight_layout()
    plt.show()


# ==================== MAIN EXECUTION ====================
configure_display()

conn = sqlite3.connect(DB_PATH)
print(f"✅ Connected to: {DB_PATH}")

raw_sessions = load_sessions(conn)
print(f"Loaded {len(raw_sessions)} total sessions after cutoff")

main_street_sessions = apply_filters(raw_sessions, scenario_name="Main_Street")
main_street_sessions.to_csv(OUTPUT_MAIN_STREET_CSV, index=False)
print(f"✅ Saved {len(main_street_sessions)} Main Street sessions to {OUTPUT_MAIN_STREET_CSV}")

top_talent_sessions = apply_filters(
    raw_sessions,
    scenario_name=SCENARIO_FILTER,
    excluded_majors=EXCLUDED_MAJORS,
)

top_talent_sessions.to_csv(OUTPUT_TOP_TALENT_CSV, index=False)
print(f"✅ Saved {len(top_talent_sessions)} Top Talent sessions to {OUTPUT_TOP_TALENT_CSV}")

conn.close()
print("✅ Database connection closed")

# ==================== TOP TALENT ANALYSIS ====================

df_all = add_derived_columns(top_talent_sessions)
df_all["deal_reached"] = df_all["deal_reached"].fillna(False).astype(bool)
df_all["deal_failed"] = df_all["deal_failed"].fillna(False).astype(bool)

df_all = add_scoring_columns(df_all)

df_deals = df_all[df_all["deal_reached"]].copy()

df_deals = df_deals[df_deals["total_pie"].notna()].copy()

show_overview(df_all)
show_overview_table(df_all)

analyze_pie_distributions(df_deals)
analyze_ai_vs_human(df_deals)

analyze_demographics(df_deals)

analyze_treatments(df_deals)
