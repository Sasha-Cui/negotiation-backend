"""
Optimized Negotiation Backend - Complete Fixed Version v3

Key Fixes in v3:
1. ✅ Filter private value fields (total_value_of_deal_to_me, etc.) from deal_terms
2. ✅ Fixed maximum rounds logic - allows overtime only when AI proposes deal at round 10
3. ✅ Removed redundant system_message when AI proposes deal (frontend handles UI)
4. ✅ Added total_rounds parameter to allow custom round limits
5. ✅ Consistent deal confirmation logic for all scenarios
6. ✅ Proper handling of final round scenarios

Previous fixes from v2:
- ✅ Fixed {{history}} double braces issue
- ✅ Added transcript in deal confirmation
- ✅ Relaxed $DEAL_REACHED$ detection
- ✅ Added are_deals_equivalent() for flexible JSON comparison
- ✅ Simplified AI confirmation prompt
- ✅ Strict handling when AI proposes deal
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import json
import yaml
import re
import os
import uuid
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# ============================================================================
# Import OpenAIWrapper
# ============================================================================
from openai_wrapper import OpenAIWrapper

# ============================================================================
# Configuration
# ============================================================================
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

DB_PATH = os.getenv("DB_PATH", "/data/negotiations.db")
SCENARIOS_DIR = os.getenv("SCENARIOS_DIR", "./scenarios")

# ============================================================================
# Constants
# ============================================================================
PRIVATE_VALUE_FIELDS = [
    'total_value_of_deal_to_me',
    'expected_value_of_deal_to_me_in_millions',
    'total_points_of_deal_to_me'
]

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(title="Negotiation Practice API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Database Initialization
# ============================================================================
def init_db():
    """Initialize database with feedback fields"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS negotiation_sessions (
            session_id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            student_name TEXT,
            scenario_name TEXT NOT NULL,
            student_role TEXT NOT NULL,
            ai_role TEXT NOT NULL,
            ai_model TEXT NOT NULL,
            student_goes_first BOOLEAN NOT NULL,
            use_memory BOOLEAN NOT NULL,
            use_plan BOOLEAN NOT NULL,
            current_round INTEGER NOT NULL,
            total_rounds INTEGER NOT NULL,
            transcript TEXT NOT NULL,
            ai_memory TEXT,
            ai_plan TEXT,
            student_deal_json TEXT,
            ai_deal_json TEXT,
            deal_reached BOOLEAN NOT NULL,
            deal_failed BOOLEAN NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            feedback_text TEXT,
            feedback_generated_at TEXT,
            feedback_model TEXT
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize on startup
init_db()

# ============================================================================
# Helper Functions
# ============================================================================

def _escape_braces(s: str) -> str:
    """Escape braces for .format()"""
    return s.replace("{", "{{").replace("}", "}}")

def _unescape_braces(s: str) -> str:
    """Unescape braces"""
    return s.replace("{{", "{").replace("}}", "}")

def filter_private_fields(deal_json: Dict) -> Dict:
    """
    ⭐ NEW v3: Remove AI's private value calculations from deal terms
    
    This ensures that when displaying deal terms to users, we don't expose
    the AI's internal value calculations, which should remain private.
    
    Args:
        deal_json: Complete deal JSON (may include private fields)
    
    Returns:
        Dict: Deal JSON with private fields removed
    """
    return {k: v for k, v in deal_json.items() if k not in PRIVATE_VALUE_FIELDS}

def last_round_window(transcript: List[str], k_rounds: int = 2) -> str:
    """
    Extract last k rounds of conversation
    
    k_rounds=2 means last 2 complete rounds (4 messages total: user, ai, user, ai)
    Identical to runner.py
    """
    pat = re.compile(r"^(.+?):\s", re.MULTILINE)
    
    speaker_changes = []
    for i, msg in enumerate(transcript):
        m = pat.match(msg)
        if m:
            speaker = m.group(1)
            speaker_changes.append((i, speaker))
    
    if len(speaker_changes) < 2 * k_rounds:
        return "\n\n".join(transcript)
    
    start_idx = speaker_changes[-(2 * k_rounds)][0]
    return "\n\n".join(transcript[start_idx:])

def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON from text
    Identical to runner.py
    """
    # Try to find JSON block
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Try direct parsing
    try:
        # Find { ... }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
    except:
        pass
    
    return None

def are_deals_equivalent(deal1: Dict, deal2: Dict, exclude_keys: List[str] = None) -> bool:
    """
    v2: Flexible comparison of two deal JSONs
    
    - Excludes AI-only fields (like total_value_of_deal_to_me)
    - Allows small numeric variations (floating point tolerance)
    - Case-insensitive string comparison with whitespace trimming
    
    Args:
        deal1: First deal JSON
        deal2: Second deal JSON
        exclude_keys: List of keys to ignore in comparison
    
    Returns:
        bool: True if deals are equivalent
    """
    if exclude_keys is None:
        exclude_keys = PRIVATE_VALUE_FIELDS.copy()
    
    # Filter out excluded keys
    filtered_deal1 = {k: v for k, v in deal1.items() if k not in exclude_keys}
    filtered_deal2 = {k: v for k, v in deal2.items() if k not in exclude_keys}
    
    # Check if keys match
    if set(filtered_deal1.keys()) != set(filtered_deal2.keys()):
        return False
    
    # Compare each value
    for key in filtered_deal1.keys():
        val1 = filtered_deal1[key]
        val2 = filtered_deal2[key]
        
        # Numeric comparison (with 0.01% tolerance for floating point issues)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Convert both to float for comparison
            v1, v2 = float(val1), float(val2)
            tolerance = max(abs(v1), abs(v2)) * 0.0001
            if abs(v1 - v2) > tolerance:
                return False
        
        # String comparison (case-insensitive, whitespace-trimmed)
        elif isinstance(val1, str) and isinstance(val2, str):
            if val1.strip().lower() != val2.strip().lower():
                return False
        
        # Boolean comparison
        elif isinstance(val1, bool) and isinstance(val2, bool):
            if val1 != val2:
                return False
        
        # Other types: direct comparison
        else:
            if val1 != val2:
                return False
    
    return True

# ============================================================================
# NegotiationSession Class
# ============================================================================
class NegotiationSession:
    """
    Negotiation Session Class
    Logic identical to runner.py with all prompts aligned
    
    v3 updates:
    - Added total_rounds parameter (customizable)
    - Fixed maximum rounds logic
    - Filter private fields from deal_terms
    """
    
    def __init__(
        self,
        session_id: str,
        student_id: str,
        student_name: str,
        scenario_name: str,
        student_role: str,
        ai_model: str,
        student_goes_first: bool,
        use_memory: bool,
        use_plan: bool,
        total_rounds: int = 10,  # ⭐ NEW v3: Customizable total rounds
    ):
        self.session_id = session_id
        self.student_id = student_id
        self.student_name = student_name
        self.scenario_name = scenario_name
        self.student_role = student_role
        self.ai_model = ai_model
        self.student_goes_first = student_goes_first
        self.use_memory = use_memory
        self.use_plan = use_plan
        
        # Load scenario config
        scenario_path = Path(SCENARIOS_DIR) / f"{scenario_name}.yaml"
        with open(scenario_path, 'r', encoding='utf-8') as f:
            self.scenario_config = yaml.safe_load(f)
        
        # Determine roles
        if student_role == "side1":
            self.ai_role = "side2"
        else:
            self.ai_role = "side1"
        
        # Initialize AI agent
        self.ai_agent = OpenAIWrapper(model=ai_model, label="AI")
        
        # Memory & Plan agents (if enabled)
        if use_memory:
            self.memory_agent = OpenAIWrapper(model=ai_model, label="Memory")
        if use_plan:
            self.plan_agent = OpenAIWrapper(model=ai_model, label="Plan")
        
        # Session state
        self.current_round = 1
        self.total_rounds = total_rounds  # ⭐ NEW v3: Use parameter instead of YAML
        self.transcript = []
        self.ai_memory = ""
        self.ai_plan = ""
        self.student_deal_json = None
        self.ai_deal_json = None
        self.deal_reached = False
        self.deal_failed = False
        self.status = "active"
        
        # Feedback fields
        self.feedback_text = None
        self.feedback_generated_at = None
        self.feedback_model = None
        
        # Timestamps
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
    
    # ========================================================================
    # Core Method 1: Generate AI Response (FULLY ALIGNED WITH runner.py)
    # ========================================================================
    def _generate_ai_response(self) -> str:
        """
        Generate AI's negotiation response
        
        Critical Fix:
        1. Always include context_prompt
        2. Distinguish Round 1 first turn (use initial_offer_prompt)
        3. Prompt text FULLY ALIGNED with runner.py
        4. Added turn_position and turn_action variables
        5. Added scenario-specific prompts
        """
        ai_cfg = self.scenario_config[self.ai_role]
        
        # 1. Get context_prompt (⭐ Critical: needed every time)
        context = ai_cfg["context_prompt"]
        
        # 2. Build complete history
        history = "\n\n".join(self.transcript)
        
        # 3. Check if this is Round 1 AND AI goes first
        is_round_1 = (self.current_round == 1)
        student_went_first = (len(self.transcript) > 0)
        ai_goes_first = is_round_1 and not student_went_first
        
        # 4. Determine turn position and action
        if self.student_goes_first:
            # Student goes first, AI goes second
            turn_position = "going second"
            turn_action = "finish"
        else:
            # AI goes first, student goes second
            turn_position = "going first"
            turn_action = "start"
        
        # 5. Choose prompt based on situation
        if ai_goes_first:
            # ============================================================
            # Round 1 first turn: use initial_offer_prompt
            # ============================================================
            base_prompt = ai_cfg["initial_offer_prompt"]
            user_prompt = f"{context}\n\n{base_prompt}"
        
        else:
            # ============================================================
            # Round 1 second turn OR Round 2+: use universal_continuation_prompt
            # ============================================================
            
            # Build JSON schema
            json_schema_raw = self.scenario_config["json_schema"]
            json_schema_dict = json.loads(json_schema_raw)
            json_schema_text = "\n" + json.dumps(json_schema_dict, indent=2)
            
            # Determine scenario-specific prompt components
            scenario_lower = self.scenario_name.lower()
            
            # ⭐ Scenario-specific value field instructions (runner.py lines 736-773)
            if scenario_lower == "top_talent":
                value_instruction = "Fill in 'total_points_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
            elif scenario_lower in ("z_deal", "zlab_split"):
                value_instruction = "Fill in 'expected_value_of_deal_to_me_in_millions' by calculating your own value using your private scoring rules described in your given prompts."
            elif scenario_lower == "twisted_tree":
                value_instruction = ""  # No value field for Twisted Tree
            elif scenario_lower == "vb_development":
                # NO-ZOPA scenario - different format
                value_instruction = ""
            else:
                # Default
                value_instruction = "Fill in 'total_value_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
            
            # ⭐ Universal continuation prompt (FULLY ALIGNED with runner.py lines 691-703)
            universal_continuation_prompt = (
                "CURRENT ROUND INFORMATION:\n"
                f"It is now round {self.current_round}/{self.total_rounds}. You are {turn_position} this round, so it is your turn to {turn_action} the round. "
                f"You have {self.total_rounds - self.current_round} rounds remaining after this one.\n\n"
                "<BEGIN COMPLETE NEGOTIATION TRANSCRIPT>\n{{history}}\n\n"
                "<END NEGOTIATION TRANSCRIPT>\n\n"
                "OUTPUT INSTRUCTIONS:\n"
                "1. Reminder: you can either continue the negotiation or accept the most recent terms offered by the other side by outputting the token '$DEAL_REACHED$' at the beginning of your output. If you do not reach a deal by the end of the last round, you get your BATNA.\n\n"
                "2. OUTPUT OPTIONS (YOU MUST CHOOSE ONE):\n"
            )
            
            # Option A: Continue negotiation
            option_a = (
                "OPTION A: Continue Negotiation\n"
                "Respond with your negotiation message. Be strategic, clear, and professional.\n\n"
            )
            
            # Option B: Accept deal
            option_b_intro = "OPTION B: Accept Deal\n"
            if value_instruction:
                option_b_body = (
                    f"If you want to accept the most recent terms offered by the other side, output ONLY the token '$DEAL_REACHED$' on the first line, "
                    f"then output the agreed terms in JSON format on subsequent lines. {value_instruction}\n"
                    f"JSON FORMAT (FOLLOW THIS EXACTLY IF CHOOSING OPTION B):\n{{json_schema}}\n\n"
                )
            else:
                option_b_body = (
                    "If you want to accept the most recent terms offered by the other side, output ONLY the token '$DEAL_REACHED$' on the first line, "
                    "then output the agreed terms in JSON format on subsequent lines.\n"
                    f"JSON FORMAT (FOLLOW THIS EXACTLY IF CHOOSING OPTION B):\n{{json_schema}}\n\n"
                )
            
            # Option C: Declare no deal possible
            option_c = (
                "OPTION C: No Deal Possible\n"
                "If you believe no mutually beneficial deal is achievable, output the token '$DEAL_FAILED$'.\n\n"
            )
            
            # Combine all options
            base_prompt = (
                universal_continuation_prompt +
                option_a +
                option_b_intro + option_b_body +
                option_c +
                "Choose one option and respond accordingly. DO NOT output multiple options."
            )
            
            # Format with history and json_schema
            # ⭐ CRITICAL: Use .format() to insert history and json_schema
            # json_schema needs escaping because it contains braces
            formatted_prompt = base_prompt.format(
                history=history,
                json_schema=_escape_braces(json_schema_text)
            )
            
            user_prompt = f"{context}\n\n{formatted_prompt}"
        
        # 6. Add Memory & Plan context (if enabled)
        if self.use_memory and self.ai_memory:
            user_prompt = f"=== Current State ===\n{self.ai_memory}\n\n{user_prompt}"
        
        if self.use_plan and self.ai_plan:
            user_prompt = f"=== Current Strategy ===\n{self.ai_plan}\n\n{user_prompt}"
        
        # 7. Make API call
        messages = [
            {"role": "system", "content": ai_cfg["system_prompt"]},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.ai_agent.chat(messages)
        return response["content"]
    
    # ========================================================================
    # Core Method 2: Update Memory (FULLY ALIGNED WITH runner.py)
    # ========================================================================
    def _update_memory(self):
        """
        Update memory state
        FULLY ALIGNED with runner.py lines 413-463
        """
        if not self.use_memory:
            return
        
        # Extract last 2 rounds of conversation
        recent_transcript = last_round_window(self.transcript, k_rounds=2)
        
        ai_cfg = self.scenario_config[self.ai_role]
        student_cfg = self.scenario_config[self.student_role]
        
        ai_label = ai_cfg.get("label", self.ai_role)
        student_label = student_cfg.get("label", self.student_role)
        
        # Get Rules & Objective and Facts & Value Model
        rules_objective = ai_cfg["system_prompt"]
        context = ai_cfg["context_prompt"]
        
        # ⭐ Memory system prompt (FULLY ALIGNED with runner.py lines 413-448)
        memory_system = (
            "You are a state tracking module for an AI negotiator. "
            "Produce a CONCISE, ACTIONABLE negotiation state that survives limited transcript windows.\n\n"
            "REQUIRED SECTIONS (keep each section ≤2 lines):\n"
            "OFFERS: [Them: <explicit numeric/boolean terms>; Us: <explicit numeric/boolean terms>; use old→new if changed]\n"
            "PATTERNS: [Concession/firmness behaviors observed across rounds]\n"
            "PRIORITIES: [Their priorities/red lines/BATNA if explicit; mark uncertain with 'Hypothesis:']\n"
            "OPPORTUNITIES: [Cross-issue trades, value-creation levers, give↔get pairs]\n"
            "INSIGHTS: [Durable facts and round-to-round deductions you will carry forward]\n\n"
            "HARD RULES:\n"
            "1) PERSISTENCE – Keep long-lived facts (BATNA, red lines, stable priorities, accepted constraints) even if they don't reappear in the latest window. Do NOT drop them unless contradicted.\n"
            "2) OFFERS COVERAGE – Always include the opponent's latest explicit terms (Them: …) AND our latest standing terms (Us: …). If no explicit numeric/boolean terms exist for a side, write '(none yet)'. Normalize numbers (e.g., '$150k', '12 months').\n"
            "3) FACT vs HYPOTHESIS – Only mark as FACT if explicitly stated; otherwise prefix with 'Hypothesis:'.\n"
            "4) CANONICALIZATION – Strip prose; compress to atomic items, use consistent units/symbols; show deltas as old→new.\n"
            "5) DEDUP & DELTA – Remove stale/duplicates; keep only the latest state; emphasize what materially changed.\n"
            "6) NO VERBATIM QUOTES – Summarize; do not copy transcript sentences.\n"
            "7) BINDING – When recording concessions, record the coupled give↔get condition if stated or implied (e.g., 'signing bonus −2k ↔ earlier start').\n"
            "8) SAFETY NET – If transcript is thin/ambiguous, prefer carrying forward last known stable state over inventing new facts.\n"
            "\n"
            "Output exactly these five sections, in this order. Keep total length tight. "
            "Keep each section to 1-2 lines max. Focus on critical information only."
        )
        
        # ⭐ Memory user prompt (FULLY ALIGNED with runner.py lines 453-463)
        memory_user = (
            f"=== Identity ===\n"
            f"Role: State tracker of {ai_label}\n"
            f"Round: {self.current_round}/{self.total_rounds}\n\n"
            f"=== Scenario Objective & Rules ===\n{rules_objective}\n\n"
            f"=== Scenario Facts & Value model ===\n{context}\n\n"
            f"=== Previous State ===\n{self.ai_memory or '(empty)'}\n\n"
            f"=== Recent Transcript Window (last 2 rounds) ===\n{recent_transcript}\n\n"
            f"=== Output State ===\n"
            "Update the negotiation state now, following HARD RULES."
        )
        
        messages = [
            {"role": "system", "content": memory_system},
            {"role": "user", "content": memory_user}
        ]
        
        response = self.memory_agent.chat(messages)
        self.ai_memory = response["content"]
    
    # ========================================================================
    # Core Method 3: Generate Plan (FULLY ALIGNED WITH runner.py)
    # ========================================================================
    def _generate_plan(self):
        """
        Generate plan for this round
        FULLY ALIGNED with runner.py lines 492-585
        """
        if not self.use_plan:
            return
        
        ai_cfg = self.scenario_config[self.ai_role]
        context = ai_cfg["context_prompt"]
        rules_objective = ai_cfg["system_prompt"]
        
        ai_label = ai_cfg.get("label", self.ai_role)
        
        # ⭐ Plan system prompt (FULLY ALIGNED with runner.py lines 502-551)
        plan_system = (
            "You are a strategic planning module for an AI negotiator. "
            "Generate a SMART and ASSERTIVE plan for THIS round based on current state and context.\n\n"
            "PLANNING RULES:\n"
            "- READ STATE – Use OFFERS, PATTERNS, PRIORITIES, OPPORTUNITIES, and INSIGHTS from the State exactly as the source of truth.\n"
            "- ROUND GOAL – Set a concrete, realistic target for THIS round.\n"
            "- TRADE-ACROSS-ISSUES – Do not concede on high-priority terms without gaining on other issues.\n"
            "- ADAPT & NON-REPETITION – If last round's plan didn't move the opponent, change tactics.\n"
            "- CLOSING OPTIONS – If the opponent's current terms already satisfy your core priorities, stay above BATNA, "
            "and do not cross your red lines, you may consider proposing acceptance; otherwise push toward your round target.\n"
            "- ZOPA AWARENESS – Only invoke ZOPA framing if it improves your outcome.\n"
            "- BREVITY – Be crisp and operational: no fluff, no transcript restatement.\n\n"
            "OUTPUT SKELETON (≤10 lines, short bullets):\n"
            "- ROUND GOAL: <one clear numeric/boolean target for this round (key term or concise combined terms), or 'consider accepting current offer'>\n"
            "- KEY LEVERS: <2–3 trade levers from PRIORITIES/OPPORTUNITIES; if accepting, note 'already satisfied'>\n"
            "- TACTICS: <2–3 concrete moves this round; if accepting, write 'consider proposing acceptance'>\n"
            "- OFFER SCAFFOLD: <one main package based on OFFERS; if accepting, refer to opponent's last offer>\n"
            "- RISK & RESPONSES: <if-then counterplans; if accepting, '(not applicable)'>\n"
        )
        
        # ⭐ Plan user prompt (FULLY ALIGNED with runner.py lines 554-563)
        plan_user = (
            f"=== Identity ===\n"
            f"Role: Strategy module of {ai_label}\n"
            f"Round: {self.current_round}/{self.total_rounds}\n\n"
            f"=== Scenario Rules & Objective ===\n{rules_objective}\n\n"
            f"=== Scenario Facts & Value model ===\n{context}\n\n"
            f"=== Current State ===\n{self.ai_memory or '(empty)'}\n\n"
            f"=== Previous Strategy ===\n{self.ai_plan or 'N/A'}\n\n"
            f"=== Output Strategy ===\n"
        )
        
        messages = [
            {"role": "system", "content": plan_system},
            {"role": "user", "content": plan_user}
        ]
        
        response = self.plan_agent.chat(messages)
        self.ai_plan = response["content"]
    
    # ========================================================================
    # Core Method 4: Process Student Message (⭐ MAJOR REFACTOR v3)
    # ========================================================================
    def process_student_message(self, message: str) -> Dict[str, Any]:
        """
        Process student message
        
        ⭐ v3 MAJOR FIXES:
        1. Fixed maximum rounds logic - check BEFORE generating AI response
        2. Allow overtime only when AI proposes deal at final round
        3. Filter private fields from all deal_terms outputs
        4. Removed redundant system_message when AI proposes deal
        5. Proper handling of all final round scenarios
        
        Scenarios at round 10:
        - 10.1 AI normal → 10.2 Student normal → Timeout (no 11.1)
        - 10.1 Student normal → 10.2 AI normal → Timeout (no 11.1)
        - 10.1 AI normal → 10.2 Student propose → AI sync confirm → Done at 10.2
        - 10.1 Student normal → 10.2 AI propose → Allow 11.1 Student confirm (overtime)
        - 10.1 AI propose → 10.2 Student confirm → Done at 10.2
        - 10.1 Student propose → 10.2 AI sync confirm → Done at 10.2
        """
        student_cfg = self.scenario_config[self.student_role]
        student_label = student_cfg.get("label", "Student")
        ai_cfg = self.scenario_config[self.ai_role]
        ai_label = ai_cfg.get("label", "AI")
        
        # Save student message to transcript
        self.transcript.append(f"{student_label}: {message}")
        
        # ========================================================================
        # SECTION A: Check if AI proposed deal in last message (HIGHEST PRIORITY)
        # ========================================================================
        last_ai_message = None
        for msg in reversed(self.transcript[:-1]):  # Exclude the message we just added
            if msg.startswith(f"{ai_label}:"):
                last_ai_message = msg
                break
        
        ai_just_proposed_deal = False
        if last_ai_message:
            message_content = last_ai_message.split(":", 1)[1].strip() if ":" in last_ai_message else ""
            ai_just_proposed_deal = message_content.startswith("$DEAL_REACHED$")
        
        if ai_just_proposed_deal:
            # ⭐ AI proposed deal - Student MUST respond (even if overtime)
            # This is the ONLY scenario where we allow going beyond total_rounds
            
            if "$DEAL_REACHED$" in message.upper():
                # Student accepts
                student_json = extract_json_from_text(message)
                ai_json = extract_json_from_text(last_ai_message)
                
                if student_json and ai_json:
                    if are_deals_equivalent(student_json, ai_json):
                        # ✅ Terms match
                        self.student_deal_json = student_json
                        self.ai_deal_json = ai_json
                        self.deal_reached = True
                        self.status = "completed"
                        return {
                            "ai_message": None,
                            "deal_reached": True,
                            "deal_terms": filter_private_fields(ai_json),  # ⭐ v3: Filter private fields
                            "system_message": "✅ Deal confirmed! Both parties agree on the terms.",
                            "round": self.current_round
                        }
                    else:
                        # ❌ Both said DEAL_REACHED but terms don't match
                        self.status = "failed"
                        return {
                            "ai_message": None,
                            "deal_reached": False,
                            "terms_mismatch": True,
                            "system_message": "❌ Both parties indicated DEAL_REACHED, but the final terms do not match. Negotiation ended.",
                            "round": self.current_round
                        }
                else:
                    # JSON parsing failed
                    return {
                        "ai_message": None,
                        "deal_reached": False,
                        "json_parse_error": True,
                        "system_message": "❌ Could not parse JSON from your message. Please check the format and try again.",
                        "round": self.current_round
                    }
            
            elif "$DEAL_MISUNDERSTANDING$" in message.upper():
                # Student rejects - claims terms don't match
                self.status = "failed"
                return {
                    "ai_message": None,
                    "deal_reached": False,
                    "misunderstanding": True,
                    "system_message": "❌ You indicated that the AI's proposed terms do not match your most recent offer. Negotiation ended.",
                    "round": self.current_round
                }
            
            else:
                # ⚠️ Invalid response - student must accept or reject
                return {
                    "ai_message": None,
                    "deal_reached": False,
                    "invalid_response": True,
                    "system_message": "⚠️ The AI has proposed a deal. You must either:\n• Accept with '$DEAL_REACHED$' followed by JSON terms\n• Reject with '$DEAL_MISUNDERSTANDING$'\n\nNo other responses are allowed at this point.",
                    "round": self.current_round
                }
        
        # ========================================================================
        # SECTION B: Check if student proposes deal
        # ========================================================================
        if "$DEAL_REACHED$" in message.upper():
            # Student proposes a deal - let AI confirm synchronously
            ai_confirm_response = self._request_ai_deal_confirmation(message)
            
            self.transcript.append(f"{ai_label}: {ai_confirm_response}")
            
            # Check AI's response (must start with the token)
            if ai_confirm_response.strip().startswith("$DEAL_REACHED$"):
                # AI confirms
                student_json = extract_json_from_text(message)
                ai_json = extract_json_from_text(ai_confirm_response)
                
                if student_json and ai_json:
                    if are_deals_equivalent(student_json, ai_json):
                        # ✅ Terms match
                        self.student_deal_json = student_json
                        self.ai_deal_json = ai_json
                        self.deal_reached = True
                        self.status = "completed"
                        return {
                            "ai_message": ai_confirm_response,
                            "deal_reached": True,
                            "deal_terms": filter_private_fields(ai_json),  # ⭐ v3: Filter private fields
                            "system_message": "✅ Deal confirmed! Both parties agree on the terms.",
                            "round": self.current_round
                        }
                    else:
                        # ❌ Both said DEAL_REACHED but terms don't match
                        self.status = "failed"
                        return {
                            "ai_message": ai_confirm_response,
                            "deal_reached": False,
                            "terms_mismatch": True,
                            "system_message": "❌ Both parties indicated DEAL_REACHED, but the final terms do not match. Negotiation ended.",
                            "round": self.current_round
                        }
                else:
                    # JSON parsing issue - but AI confirmed, so use AI's JSON
                    if ai_json:
                        self.ai_deal_json = ai_json
                        self.deal_reached = True
                        self.status = "completed"
                        return {
                            "ai_message": ai_confirm_response,
                            "deal_reached": True,
                            "deal_terms": filter_private_fields(ai_json),  # ⭐ v3: Filter private fields
                            "system_message": "✅ Deal confirmed! (Using AI's parsed terms)",
                            "round": self.current_round
                        }
            
            elif ai_confirm_response.strip().startswith("$DEAL_MISUNDERSTANDING$"):
                # AI rejects - terms don't match
                self.status = "failed"
                return {
                    "ai_message": ai_confirm_response,
                    "deal_reached": False,
                    "misunderstanding": True,
                    "system_message": "❌ AI believes the terms you specified do not match its most recent offer. Negotiation ended.",
                    "round": self.current_round
                }
        
        # ========================================================================
        # SECTION C: Check if student declares deal failed
        # ========================================================================
        if "$DEAL_FAILED$" in message.upper():
            self.deal_failed = True
            self.status = "failed"
            return {
                "ai_message": "You have indicated that no deal is possible. Negotiation ended.",
                "deal_failed": True,
                "system_message": "❌ You declared that no deal is possible. Negotiation ended.",
                "round": self.current_round
            }
        
        # ========================================================================
        # SECTION D: Normal negotiation - generate AI response
        # ========================================================================
        
        # Update Memory & Plan (if enabled)
        if self.current_round > 1 or len(self.transcript) > 1:
            self._update_memory()
            self._generate_plan()
        
        # Generate AI response
        ai_response = self._generate_ai_response()
        self.transcript.append(f"{ai_label}: {ai_response}")
        
        # ⭐ Check if AI proposed deal
        if ai_response.strip().startswith("$DEAL_REACHED$"):
            ai_json = extract_json_from_text(ai_response)
            if ai_json:
                self.ai_deal_json = ai_json
                
                # Determine if this is the final round
                is_final_round = (self.current_round == self.total_rounds)
                
                return {
                    "ai_message": ai_response,
                    "ai_proposed_deal": True,
                    "ai_deal_terms": filter_private_fields(ai_json),  # ⭐ v3: Filter private fields
                    # ⭐ v3: Removed redundant system_message (frontend handles UI)
                    "round": self.current_round,
                    "allow_overtime": is_final_round  # ⭐ v3: Signal overtime allowed
                }
        
        # Check if AI declared deal failed
        if "$DEAL_FAILED$" in ai_response:
            self.deal_failed = True
            self.status = "failed"
            return {
                "ai_message": ai_response,
                "deal_failed": True,
                "system_message": "❌ AI declared that no deal is possible. Negotiation ended.",
                "round": self.current_round
            }
        
        # ⭐ v3 KEY FIX: Check if reached last round AFTER generating response
        # Only end if AI gave normal response (not deal proposal)
        if self.current_round >= self.total_rounds:
            self.status = "completed"
            return {
                "ai_message": ai_response,
                "negotiation_ended": True,
                "reason": "max_rounds_reached",
                "system_message": "⏱️ Maximum rounds reached. Negotiation ended without a deal.",
                "round": self.current_round
            }
        
        # Move to next round
        self.current_round += 1
        
        return {
            "ai_message": ai_response,
            "deal_reached": False,
            "round": self.current_round - 1
        }
    
    # ========================================================================
    # Core Method 5: AI Confirm Deal (SIMPLIFIED v2)
    # ========================================================================
    def _request_ai_deal_confirmation(self, student_message: str) -> str:
        """
        Request AI to confirm student's proposed deal
        
        v2 CHANGES:
        1. Simplified prompt - no explanation required for MISUNDERSTANDING
        2. AI judges based on complete transcript
        3. Removed redundant "ignore value" instruction
        """
        ai_cfg = self.scenario_config[self.ai_role]
        context = ai_cfg["context_prompt"]
        
        # Build complete history
        history = "\n\n".join(self.transcript)
        
        # Get JSON schema
        json_schema_raw = self.scenario_config["json_schema"]
        json_schema_dict = json.loads(json_schema_raw)
        json_schema_text = "\n" + json.dumps(json_schema_dict, indent=2)
        
        # Determine scenario-specific value field instruction
        scenario_lower = self.scenario_name.lower()
        
        if scenario_lower == "top_talent":
            value_field_instruction = "Fill in 'total_points_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
        elif scenario_lower in ("z_deal", "zlab_split"):
            value_field_instruction = "Fill in 'expected_value_of_deal_to_me_in_millions' by calculating your own value using your private scoring rules described in your given prompts."
        elif scenario_lower in ("twisted_tree", "vb_development"):
            value_field_instruction = ""
        else:
            value_field_instruction = "Fill in 'total_value_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
        
        # ⭐ Simplified confirmation prompt
        conf_prompt = (
            "DEAL CONFIRMATION:\n"
            "The other side indicated that they accept a deal with '$DEAL_REACHED$' and specified deal terms. "
            "Look at the complete negotiation transcript below to determine if the terms they specified "
            "are consistent with the most recent offer YOU made.\n\n"
            "COMPLETE NEGOTIATION TRANSCRIPT:\n"
            "<BEGIN TRANSCRIPT>\n"
            f"{history}\n"
            "<END TRANSCRIPT>\n\n"
            "OUTPUT INSTRUCTIONS: CHOOSE ONE OF THE FOLLOWING TWO OPTIONS\n\n"
        )
        
        if value_field_instruction:
            conf_prompt += f"OPTION 1: If the terms ARE consistent with your most recent offer:\n"
            conf_prompt += f"Output ONLY the token '$DEAL_REACHED$' on the first line, then output the agreed terms in JSON format. {value_field_instruction}\n"
        else:
            conf_prompt += "OPTION 1: If the terms ARE consistent with your most recent offer:\n"
            conf_prompt += "Output ONLY the token '$DEAL_REACHED$' on the first line, then output the agreed terms in JSON format.\n"
        
        conf_prompt += (
            f"JSON FORMAT (FOLLOW THIS EXACTLY IF CHOOSING OPTION 1):\n{_escape_braces(json_schema_text)}\n\n"
            "OPTION 2: If the terms are NOT consistent with your most recent offer:\n"
            "Output ONLY the token '$DEAL_MISUNDERSTANDING$' (no explanation needed).\n\n"
            "DO NOT output anything else besides the token and JSON (for Option 1) or just the token (for Option 2)."
        )
        
        full_prompt = f"{context}\n\n{conf_prompt}"
        
        messages = [
            {"role": "system", "content": ai_cfg["system_prompt"]},
            {"role": "user", "content": full_prompt}
        ]
        
        response = self.ai_agent.chat(messages)
        return response["content"]
    
    # ========================================================================
    # Core Method 6: Generate Feedback (RELAXED CONDITIONS)
    # ========================================================================
    def generate_feedback(self) -> str:
        """
        Generate negotiation feedback
        
        Can be called at any point (active, completed, failed, ended_early)
        """
        feedback_agent = OpenAIWrapper(
            model=self.ai_model,
            label="FeedbackCoach"
        )
        
        # Build complete transcript
        complete_transcript = "\n\n".join(self.transcript)
        
        # Get scenario information
        student_cfg = self.scenario_config[self.student_role]
        ai_cfg = self.scenario_config[self.ai_role]
        scenario_context = student_cfg["context_prompt"]
        
        # Feedback system prompt
        feedback_system = (
            "You are an expert negotiation coach providing feedback to MBA students. "
            "Your feedback should be constructive, specific, and actionable. "
            "Focus on strategic insights, communication skills, value creation, and specific recommendations for improvement. "
            "If the negotiation is incomplete, provide feedback based on the progress so far and suggest how to proceed."
        )
        
        # Determine outcome status
        if self.deal_reached:
            outcome_status = "Deal reached successfully"
        elif self.deal_failed:
            outcome_status = "Deal explicitly failed (one party declared no deal possible)"
        elif self.status == "completed":
            outcome_status = "Maximum rounds reached without a deal"
        else:
            outcome_status = "Negotiation in progress or ended early"
        
        # Feedback user prompt
        feedback_user = (
            f"=== Negotiation Session ===\n"
            f"Scenario: {self.scenario_name}\n"
            f"Student Role: {student_cfg.get('label', self.student_role)}\n"
            f"AI Role: {ai_cfg.get('label', self.ai_role)}\n"
            f"Rounds Completed: {self.current_round}/{self.total_rounds}\n"
            f"Outcome: {outcome_status}\n\n"
            f"=== Student's Role Context ===\n{scenario_context}\n\n"
            f"=== Complete Negotiation Transcript ===\n{complete_transcript}\n\n"
            f"=== Feedback Request ===\n"
            "Provide comprehensive feedback on the student's negotiation performance. Structure your feedback as follows:\n\n"
            "1. **Overall Assessment**: Brief summary of the student's performance and outcome\n"
            "2. **Strengths**: What the student did well (be specific with examples from the transcript)\n"
            "3. **Areas for Improvement**: What could have been done better (with specific suggestions)\n"
            "4. **Strategic Insights**: Analysis of negotiation strategy, value creation opportunities, and tactical choices\n"
            "5. **Actionable Recommendations**: 3-5 concrete steps the student can take to improve in future negotiations\n\n"
            "Be encouraging but honest. Focus on helping the student learn and improve."
        )
        
        messages = [
            {"role": "system", "content": feedback_system},
            {"role": "user", "content": feedback_user}
        ]
        
        response = feedback_agent.chat(messages)
        
        # Store feedback
        self.feedback_text = response["content"]
        self.feedback_generated_at = datetime.utcnow().isoformat()
        self.feedback_model = self.ai_model
        
        return self.feedback_text
    
    # ========================================================================
    # Database Methods
    # ========================================================================
    def save_to_db(self):
        """Save session to database"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        self.updated_at = datetime.utcnow().isoformat()

        c.execute("""
            INSERT OR REPLACE INTO negotiation_sessions (
                session_id,
                student_id,
                student_name,
                scenario_name,
                student_role,
                ai_role,
                ai_model,
                student_goes_first,
                use_memory,
                use_plan,
                current_round,
                total_rounds,
                transcript,
                ai_memory,
                ai_plan,
                student_deal_json,
                ai_deal_json,
                deal_reached,
                deal_failed,
                status,
                created_at,
                updated_at,
                feedback_text,
                feedback_generated_at,
                feedback_model
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            self.session_id,
            self.student_id,
            self.student_name,
            self.scenario_name,
            self.student_role,
            self.ai_role,
            self.ai_model,
            int(self.student_goes_first),
            int(self.use_memory),
            int(self.use_plan),
            self.current_round,
            self.total_rounds,
            json.dumps(self.transcript),
            self.ai_memory,
            self.ai_plan,
            json.dumps(self.student_deal_json) if self.student_deal_json else None,
            json.dumps(self.ai_deal_json) if self.ai_deal_json else None,
            int(self.deal_reached),
            int(self.deal_failed),
            self.status,
            self.created_at,
            self.updated_at,
            self.feedback_text,
            self.feedback_generated_at,
            self.feedback_model,
        ))

        conn.commit()
        conn.close()
    
    @staticmethod
    def load_from_db(session_id: str) -> Optional['NegotiationSession']:
        """Load session from database"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            SELECT * FROM negotiation_sessions WHERE session_id = ?
        """, (session_id,))
        
        row = c.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Create session object
        session = NegotiationSession(
            session_id=row[0],
            student_id=row[1],
            student_name=row[2],
            scenario_name=row[3],
            student_role=row[4],
            ai_model=row[6],
            student_goes_first=bool(row[7]),
            use_memory=bool(row[8]),
            use_plan=bool(row[9]),
            total_rounds=row[11]  # ⭐ v3: Load total_rounds from DB
        )
        
        # Restore state
        session.ai_role = row[5]
        session.current_round = row[10]
        session.transcript = json.loads(row[12])
        session.ai_memory = row[13] or ""
        session.ai_plan = row[14] or ""
        session.student_deal_json = json.loads(row[15]) if row[15] else None
        session.ai_deal_json = json.loads(row[16]) if row[16] else None
        session.deal_reached = bool(row[17])
        session.deal_failed = bool(row[18])
        session.status = row[19]
        session.created_at = row[20]
        session.updated_at = row[21]
        session.feedback_text = row[22]
        session.feedback_generated_at = row[23]
        session.feedback_model = row[24]
        
        return session
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "student_id": self.student_id,
            "student_name": self.student_name,
            "scenario_name": self.scenario_name,
            "student_role": self.student_role,
            "ai_role": self.ai_role,
            "ai_model": self.ai_model,
            "student_goes_first": self.student_goes_first,
            "use_memory": self.use_memory,
            "use_plan": self.use_plan,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "transcript": self.transcript,
            "deal_reached": self.deal_reached,
            "deal_failed": self.deal_failed,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

# ============================================================================
# Request/Response Models
# ============================================================================
class StartNegotiationRequest(BaseModel):
    student_id: str
    student_name: str
    scenario_name: str
    student_role: str
    ai_model: str = "anthropic/claude-3-sonnet"
    randomize_first_turn: bool = True
    use_memory: bool = True
    use_plan: bool = True
    total_rounds: int = 10  # ⭐ NEW v3: Allow customizable total rounds

class SendMessageRequest(BaseModel):
    message: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def read_root():
    """Health check"""
    return {"status": "ok", "message": "Negotiation Practice API v3"}

@app.get("/scenarios")
def list_scenarios():
    """List all available scenarios"""
    scenarios_path = Path(SCENARIOS_DIR)
    if not scenarios_path.exists():
        return {"scenarios": []}
    
    scenarios = []
    for yaml_file in scenarios_path.glob("*.yaml"):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                scenarios.append({
                    "id": yaml_file.stem,
                    "name": config.get("name", yaml_file.stem),
                    "side1_label": config.get("side1", {}).get("label", "Side 1"),
                    "side2_label": config.get("side2", {}).get("label", "Side 2"),
                })
        except:
            continue
    
    return {"scenarios": scenarios}

@app.post("/negotiation/start")
def start_negotiation(request: StartNegotiationRequest):
    """Start new negotiation session"""
    try:
        session_id = str(uuid.uuid4())
        
        if request.randomize_first_turn:
            student_goes_first = random.choice([True, False])
        else:
            student_goes_first = True
        
        session = NegotiationSession(
            session_id=session_id,
            student_id=request.student_id,
            student_name=request.student_name,
            scenario_name=request.scenario_name,
            student_role=request.student_role,
            ai_model=request.ai_model,
            student_goes_first=student_goes_first,
            use_memory=request.use_memory,
            use_plan=request.use_plan,
            total_rounds=request.total_rounds  # ⭐ v3: Pass custom total_rounds
        )
        
        ai_first_message = None
        if not student_goes_first:
            ai_response = session._generate_ai_response()
            ai_cfg = session.scenario_config[session.ai_role]
            ai_label = ai_cfg.get("label", "AI")
            session.transcript.append(f"{ai_label}: {ai_response}")
            ai_first_message = ai_response
        
        session.save_to_db()
        
        return {
            "session_id": session_id,
            "student_goes_first": student_goes_first,
            "ai_first_message": ai_first_message,
            "total_rounds": session.total_rounds,
            "scenario_name": request.scenario_name,
            "student_role": request.student_role,
            "ai_role": session.ai_role,
            "ai_model": request.ai_model
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario '{request.scenario_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/negotiation/{session_id}/message")
def send_message(session_id: str, request: SendMessageRequest):
    """Send student message and get AI response"""
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status not in ["active", "completed", "failed"]:
        raise HTTPException(status_code=400, detail="Session is not in a valid state")
    
    result = session.process_student_message(request.message)
    session.save_to_db()
    
    return result

@app.get("/negotiation/{session_id}/status")
def get_status(session_id: str):
    """Get session status"""
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()

@app.get("/negotiation/{session_id}/transcript")
def get_transcript(session_id: str):
    """Get complete conversation transcript"""
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "transcript": session.transcript,
        "current_round": session.current_round,
        "total_rounds": session.total_rounds,
        "deal_reached": session.deal_reached,
        "deal_failed": session.deal_failed,
        "status": session.status
    }

@app.get("/negotiation/{session_id}/role_info")
def get_role_info(session_id: str):
    """
    Get the student's complete role information for display
    v2: Also return json_schema for frontend to generate examples
    """
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    student_cfg = session.scenario_config[session.student_role]
    
    return {
        "session_id": session_id,
        "role_label": student_cfg.get("label", session.student_role),
        "batna": student_cfg.get("batna", 0),
        "system_prompt": student_cfg["system_prompt"],
        "context_prompt": student_cfg["context_prompt"],
        "scenario_name": session.scenario_name,
        "total_rounds": session.total_rounds,
        "json_schema": session.scenario_config["json_schema"]
    }

@app.get("/negotiation/{session_id}/feedback")
def get_feedback(session_id: str):
    """Get or generate negotiation feedback"""
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.feedback_text:
        return {
            "session_id": session_id,
            "feedback": session.feedback_text,
            "generated_at": session.feedback_generated_at,
            "model": session.feedback_model,
            "status_when_generated": session.status,
            "cached": True
        }
    
    try:
        feedback = session.generate_feedback()
        session.save_to_db()
        
        return {
            "session_id": session_id,
            "feedback": feedback,
            "generated_at": session.feedback_generated_at,
            "model": session.feedback_model,
            "status_when_generated": session.status,
            "cached": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate feedback: {str(e)}")

@app.post("/negotiation/{session_id}/regenerate_feedback")
def regenerate_feedback(session_id: str):
    """Force regenerate feedback"""
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        feedback = session.generate_feedback()
        session.save_to_db()
        
        return {
            "session_id": session_id,
            "feedback": feedback,
            "generated_at": session.feedback_generated_at,
            "model": session.feedback_model,
            "status_when_generated": session.status,
            "regenerated": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to regenerate feedback: {str(e)}")

@app.get("/download_db")
def download_db(secret: Optional[str] = None):
    """Download database file"""
    allowed = os.getenv("DOWNLOAD_KEY")
    if allowed and secret != allowed:
        raise HTTPException(status_code=403, detail="unauthorized")
    return FileResponse(DB_PATH, filename="negotiations.db")

@app.get("/health")
def health_check():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ============================================================================
# Run
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)