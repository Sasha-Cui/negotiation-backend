"""TTverifier_and_scorer.py – Top Talent deal verification and scoring.

Deterministic verifier/scorer for the Top Talent scenario.
- Full agreement → always compute true points (even if constraint fails)
- verified_agreement = full_agreement AND all_valid
- 'invalid' sentinel → throwaway (no scoring)
- 'misunderstanding' is always False here (runner already handles it)
- Claimed points key: total_points_of_deal_to_me
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import sys

def _error(msg: str):
    print(f"[TTverifier_and_scorer] ERROR: {msg}", file=sys.stderr)
    raise ValueError(msg)

def _parse_remote_days(s: str | None) -> int:
    if not isinstance(s, str):
        _error("'remote_work' missing or not a string")
    try:
        return int(s.strip().split()[0])  # "0 day(s) / week" -> 0
    except (ValueError, IndexError):
        _error(f"Unable to parse remote_work value: {s!r}")

def _parse_vacation_weeks(s: str | None) -> float:
    if not isinstance(s, str):
        _error("'vacation_time' missing or not a string")
    try:
        return float(s.strip().split()[0])  # "3.5 weeks" -> 3.5
    except (ValueError, IndexError):
        _error(f"Unable to parse vacation_time value: {s!r}")

def _has_invalid_field(d: Dict[str, Any]) -> bool:
    return any(v == "invalid" for v in (d or {}).values())

def _compute_agreement_and_finals(side1_final: Dict[str, Any], side2_final: Dict[str, Any]) -> Dict[str, Any]:
    """Compare JSONs to determine per-issue agreement and final values."""
    result: Dict[str, Any] = {}

    issues = [
        "start_date", "work_location", "base_salary", "remote_work",
        "signing_bonus", "vacation_time", "blair_rotation"
    ]
    all_agree = []
    for issue in issues:
        k1 = f"final_{issue}"
        k2 = f"final_{issue}"
        v1 = side1_final.get(k1)
        v2 = side2_final.get(k2)

        agree = (v1 == v2) and (v1 is not None)
        result[f"{issue}_agreement"] = agree
        all_agree.append(agree)
        result[f"final_{issue}"] = v1 if agree else None

    result["full_agreement"] = all(all_agree)

    # Blair rotation requires August 1st or later
    br = result.get("final_blair_rotation")
    sd = result.get("final_start_date")
    if br is True and sd != "August 1st or later":
        result["blair_start_valid"] = False
    else:
        result["blair_start_valid"] = True

    return result

# ---------- True points (must match YAML exactly) ----------

def _recruiter_points_top_talent(deal: Dict[str, Any]) -> int:
    p = 0
    # 1) Start date
    sd = str(deal.get("start_date", ""))
    if sd == "July 5th":
        p += 4000
    elif sd == "July 15th":
        p += 2000
    # else August 1st or later: +0

    # 2) Work location
    loc = str(deal.get("work_location", ""))
    if loc == "New Haven, CT":
        p += 20000
    elif loc == "Seoul, Korea":
        p += 7500
    # Philadelphia: +0

    # 3) Base salary (linear around 135k): +1 per $ below 135k, -1 per $ above
    if "base_salary" not in deal or deal["base_salary"] is None:
        _error("'base_salary' missing in deal")
    salary = int(deal["base_salary"])
    p += 135_000 - salary

    # 4) Remote work
    remote_days = _parse_remote_days(deal.get("remote_work"))
    if remote_days == 0:
        p += 5000
    elif remote_days == 1:
        p += 2500
    # 2 days: +0

    # 5) Signing bonus: +0.2 per $1 below 40k
    if "signing_bonus" not in deal or deal["signing_bonus"] is None:
        _error("'signing_bonus' missing in deal")
    bonus = int(deal["signing_bonus"])
    p += int(0.2 * (40_000 - bonus))

    # 6) Vacation
    vac = _parse_vacation_weeks(deal.get("vacation_time"))
    vac_table = {2: 20000, 2.5: 15000, 3: 10000, 3.5: 5000, 4: 0}
    p += vac_table.get(vac, 0)

    # 7) Blair rotation
    if deal.get("blair_rotation", False):
        p -= 500

    return int(p)

def _candidate_points_top_talent(deal: Dict[str, Any]) -> int:
    p = 0
    # 1) Start date
    sd = str(deal.get("start_date", ""))
    if sd == "August 1st or later":
        p += 15000
    elif sd == "July 15th":
        p += 7500
    # July 5th: +0

    # 2) Work location
    loc = str(deal.get("work_location", ""))
    if loc == "Philadelphia, PA":
        p += 4000
    elif loc == "Seoul, Korea":
        p += 2000
    # New Haven: +0

    # 3) Base salary (linear around 165k): +1 per $ above 165k, -1 per $ below
    if "base_salary" not in deal or deal["base_salary"] is None:
        _error("'base_salary' missing in deal")
    salary = int(deal["base_salary"])
    p += salary - 165_000

    # 4) Remote work
    remote_days = _parse_remote_days(deal.get("remote_work"))
    if remote_days == 0:
        p += 5000
    elif remote_days == 1:
        p += 2500
    # 2 days: +0

    # 5) Signing bonus: +0.5 per $1 above 0
    if "signing_bonus" not in deal or deal["signing_bonus"] is None:
        _error("'signing_bonus' missing in deal")
    bonus = int(deal["signing_bonus"])
    p += int(0.5 * bonus)

    # 6) Vacation
    vac = _parse_vacation_weeks(deal.get("vacation_time"))
    vac_table = {2: 0, 2.5: 2000, 3: 4000, 3.5: 6000, 4: 8000}
    p += vac_table.get(vac, 0)

    # 7) Blair rotation
    if deal.get("blair_rotation", False):
        p += 5000

    return int(p)

# ---------- Main verifier/scorer ----------

def verify_and_score_top_talent(side1_final: Dict[str, Any],
                                side2_final: Dict[str, Any],
                                deal_reached_token: bool) -> Dict[str, Any]:
    """
    Returns a dict with:
      - deal_reached_token, misunderstanding(False), full_agreement
      - per-issue *_agreement and final_* values
      - blair_start_valid, all_valid, verified_agreement
      - recruiter_points, candidate_points (true points; set when full_agreement)
      - true_total_points
      - recruiter_points_correct, candidate_points_correct (bool | None)
      - recruiter_point_wrongness, candidate_point_wrongness (signed diff or None)
    """
    res: Dict[str, Any] = {
        "deal_reached_token": bool(deal_reached_token),
        "misunderstanding": False,  # runner 已处理确认阶段；这里固定 False
        "verified_agreement": False,
        "all_valid": None,
    }

    # Throwaway: any 'invalid'
    if _has_invalid_field(side1_final) or _has_invalid_field(side2_final):
        issues = ["start_date", "work_location", "base_salary", "remote_work",
                  "signing_bonus", "vacation_time", "blair_rotation"]
        for it in issues:
            res[f"{it}_agreement"] = None
            res[f"final_{it}"] = None
        res["full_agreement"] = None
        res["blair_start_valid"] = None
        res["all_valid"] = False
        # 不计分，也不做正确性比较（缺乏可信的 final_*）
        res["recruiter_points"] = None
        res["candidate_points"] = None
        res["true_total_points"] = None
        res["recruiter_points_correct"] = None
        res["candidate_points_correct"] = None
        res["recruiter_point_wrongness"] = None
        res["candidate_point_wrongness"] = None
        return res

    # Compare and get finals
    agree = _compute_agreement_and_finals(side1_final, side2_final)
    res.update(agree)  # *_agreement, final_*, full_agreement, blair_start_valid

    full = bool(res.get("full_agreement"))
    all_valid = bool(res.get("blair_start_valid"))
    res["all_valid"] = all_valid
    res["verified_agreement"] = bool(full and all_valid)

    # 当 full_agreement 成立就计算真实分（与 SnyderMed 对齐）
    if full:
        deal = {
            k.removeprefix("final_"): v
            for k, v in res.items()
            if k.startswith("final_") and v is not None
        }
        recruiter_pts = _recruiter_points_top_talent(deal)
        candidate_pts = _candidate_points_top_talent(deal)
        res["recruiter_points"] = recruiter_pts
        res["candidate_points"] = candidate_pts
        res["true_total_points"] = recruiter_pts + candidate_pts

        # 正确性/误差（有符号：claimed - true），无论是否 verified 都对比
        s1_claim = side1_final.get("total_points_of_deal_to_me")
        s2_claim = side2_final.get("total_points_of_deal_to_me")

        if isinstance(s1_claim, (int, float)):
            diff = s1_claim - recruiter_pts
            res["recruiter_points_correct"] = (diff == 0)
            res["recruiter_point_wrongness"] = diff
        else:
            res["recruiter_points_correct"] = None
            res["recruiter_point_wrongness"] = None

        if isinstance(s2_claim, (int, float)):
            diff = s2_claim - candidate_pts
            res["candidate_points_correct"] = (diff == 0)
            res["candidate_point_wrongness"] = diff
        else:
            res["candidate_points_correct"] = None
            res["candidate_point_wrongness"] = None
    else:
        # 条款不完全一致：无法唯一确定最终条款 → 不计分
        res["recruiter_points"] = None
        res["candidate_points"] = None
        res["true_total_points"] = None
        res["recruiter_points_correct"] = None
        res["candidate_points_correct"] = None
        res["recruiter_point_wrongness"] = None
        res["candidate_point_wrongness"] = None

    return res