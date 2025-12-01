"""
Optimized Negotiation Backend - Complete Fixed Version v5 FINAL

Key Fixes in v5 (Simplified Round Management):
1. ✅ Removed messages_in_current_round - use len(transcript) directly
2. ✅ Removed complex current_round tracking - calculate from transcript
3. ✅ Added round information to transcript: "Round X.Y - Label: message"
4. ✅ Fixed all special token detection to use .strip().startswith()
5. ✅ Check round completion BEFORE generating AI response (avoid wasted API calls)
6. ✅ Clear overtime policy: only when AI/Student propose at final round

Previous fixes from v4:
1. ✅ Clear round definition: 1 round = exactly 2 messages (Student + AI)
2. ✅ Fixed overtime logic: allow overtime only when deal proposed at final round
3. ✅ AI's synchronous confirmation does NOT add new message to round count

Previous fixes from v3:
1. ✅ Filter private value fields from deal_terms
2. ✅ Fixed maximum rounds logic
3. ✅ Removed redundant system_message when AI proposes deal
4. ✅ Added total_rounds parameter
5. ✅ Consistent deal confirmation logic

Previous fixes from v2:
1. ✅ Fixed {{history}} double braces issue
2. ✅ Added transcript in deal confirmation
3. ✅ Relaxed $DEAL_REACHED$ detection
4. ✅ Added are_deals_equivalent()
5. ✅ Simplified AI confirmation prompt
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
    """Initialize database - removed messages_in_current_round field"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    c = conn.cursor()
    
    # Create table 

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
            total_rounds INTEGER NOT NULL,
            transcript TEXT NOT NULL,
            ai_memory TEXT,
            ai_plan TEXT,
            ai_memory_history TEXT,
            ai_plan_history TEXT,
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
    """Remove AI's private value calculations from deal terms"""
    return {k: v for k, v in deal_json.items() if k not in PRIVATE_VALUE_FIELDS}

def calculate_round_info(transcript_length: int, student_goes_first: bool) -> tuple:
    """
    Calculate round number and message position from transcript length
    
    Args:
        transcript_length: Current length of transcript
        student_goes_first: Whether student goes first
    
    Returns:
        (round_number, message_in_round, is_round_complete)
    """
    if student_goes_first:
        # Student first: message 1,3,5,... = Student; 2,4,6,... = AI
        # Round 1 = messages 1-2, Round 2 = messages 3-4, etc.
        round_number = (transcript_length + 1) // 2
        is_student_turn = (transcript_length % 2 == 0)  # Even = Student's turn
        is_round_complete = (transcript_length % 2 == 0)  # Odd = AI just spoke, round complete
    else:
        # AI first: message 1,3,5,... = AI; 2,4,6,... = Student
        # Round 1 = messages 1-2, Round 2 = messages 3-4, etc.
        round_number = (transcript_length + 1) // 2
        is_student_turn = (transcript_length % 2 == 1)  # Odd = Student's turn
        is_round_complete = (transcript_length % 2 == 0)  # Even = Student just spoke, round complete
    
    return round_number, is_round_complete


def last_round_window(transcript: List[str], k_rounds: int = 2) -> str:
    """
    Extract last k rounds of conversation
    k_rounds=2 means last 2 complete rounds (4 messages total)
    """
    # Since each round = 2 messages (one from each party)
    # k_rounds = k * 2 messages
    messages_needed = k_rounds * 2
    
    if len(transcript) <= messages_needed:
        return "\n\n".join(transcript)
    
    return "\n\n".join(transcript[-messages_needed:])



def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from text"""
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
    Flexible comparison of two deal JSONs
    - Excludes AI-only fields
    - Allows small numeric variations
    - Case-insensitive string comparison
    """
    if exclude_keys is None:
        exclude_keys = PRIVATE_VALUE_FIELDS.copy()
    
    filtered_deal1 = {k: v for k, v in deal1.items() if k not in exclude_keys}
    filtered_deal2 = {k: v for k, v in deal2.items() if k not in exclude_keys}
    
    if set(filtered_deal1.keys()) != set(filtered_deal2.keys()):
        return False
    
    for key in filtered_deal1.keys():
        val1 = filtered_deal1[key]
        val2 = filtered_deal2[key]
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            v1, v2 = float(val1), float(val2)
            tolerance = max(abs(v1), abs(v2)) * 0.0001
            if abs(v1 - v2) > tolerance:
                return False
        elif isinstance(val1, str) and isinstance(val2, str):
            if val1.strip().lower() != val2.strip().lower():
                return False
        elif isinstance(val1, bool) and isinstance(val2, bool):
            if val1 != val2:
                return False
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
    
    v5 FINAL updates:
    - Removed messages_in_current_round field
    - Removed complex current_round tracking
    - Use len(transcript) to calculate everything
    - Added round info to transcript format
    - Fixed all special token detection
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
        total_rounds: int = 6,
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
        self.total_rounds = total_rounds
        
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
        self.transcript = []
        self.ai_memory = ""
        self.ai_plan = ""
        self.ai_memory_history = []  
        self.ai_plan_history = []    
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
    
    def get_current_round(self) -> int:
        """Calculate current round from transcript length"""
        round_num, _ = calculate_round_info(len(self.transcript), self.student_goes_first)
        return round_num
    
    def get_ai_next_round_info(self) -> tuple:
        """
        Calculate which round and position AI will respond to
        
        Returns:
            (round_number, message_in_round)
            e.g., (3, 1) means AI will respond as Round 3.1
        """
        transcript_len = len(self.transcript)
        next_message_num = transcript_len + 1  # AI's next message position
        
        round_num = (next_message_num + 1) // 2
        message_in_round = 1 if next_message_num % 2 == 1 else 2
        
        return round_num, message_in_round
    
    # ========================================================================
    # Core Method 1: Generate AI Response
    # ========================================================================
    def _generate_ai_response(self) -> str:
        """Generate AI's negotiation response"""
        ai_cfg = self.scenario_config[self.ai_role]
        context = ai_cfg["context_prompt"]
        
        # Build complete history
        history = "\n\n".join(self.transcript)
        
        # Check if this is Round 1 AND AI goes first
        is_round_1 = (len(self.transcript) == 0)
        ai_goes_first = is_round_1 and not self.student_goes_first
        
        # Determine turn position and action
        if self.student_goes_first:
            turn_position = "going second"
            turn_action = "finish"
        else:
            turn_position = "going first"
            turn_action = "start"
        
        # Choose prompt based on situation
        if ai_goes_first:
            base_prompt = ai_cfg["initial_offer_prompt"]
            user_prompt = f"{context}\n\n{base_prompt}"
        else:
            ai_round, ai_position = self.get_ai_next_round_info()
            
            json_schema_raw = self.scenario_config["json_schema"]
            json_schema_dict = json.loads(json_schema_raw)
            json_schema_text = "\n" + json.dumps(json_schema_dict, indent=2)
            
            scenario_lower = self.scenario_name.lower()
            
            if scenario_lower == "top_talent":
                value_instruction = "Fill in 'total_points_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
            elif scenario_lower in ("z_deal", "zlab_split"):
                value_instruction = "Fill in 'expected_value_of_deal_to_me_in_millions' by calculating your own value using your private scoring rules described in your given prompts."
            elif scenario_lower in ("twisted_tree", "vb_development", "main_street"):
                value_instruction = ""
            else:
                value_instruction = "Fill in 'total_value_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
            
            universal_continuation_prompt = (
                "CURRENT ROUND INFORMATION:\n"
                f"It is now round {ai_round}/{self.total_rounds}. You are {turn_position} this round, so it is your turn to {turn_action} the round. "
                f"You have {self.total_rounds - ai_round} rounds remaining after this one.\n\n"
                f"<BEGIN COMPLETE NEGOTIATION TRANSCRIPT>\n{history}\n\n"
                "<END NEGOTIATION TRANSCRIPT>\n\n"
                "OUTPUT INSTRUCTIONS:\n"
                "1. Reminder: you can either continue the negotiation or accept the most recent terms offered by the other side by outputting the token '$DEAL_REACHED$' at the beginning of your output. If neither you nor the other side sends that token by the end of the last round, no deal will have been reached and you get your BATNA.\n\n"
                "2. OUTPUT OPTIONS (YOU MUST CHOOSE ONE):\n"
            )
            
            option_a = (
                "OPTION A: Continue Negotiation\n"
                "Respond with your negotiation message. Be strategic, clear, and professional.\n\n"
            )
            
            option_b_intro = "OPTION B: Accept Deal\n"
            if value_instruction:
                option_b_body = (
                    f"If you want to accept the most recent terms offered by the other side, output ONLY the token '$DEAL_REACHED$' on the first line, "
                    f"then output the agreed terms in JSON format on subsequent lines. {value_instruction}\n"
                    f"JSON FORMAT: Output a JSON instance with actual values, NOT the schema definition.\n"
                    f"The schema below shows the structure - fill in the actual deal values:\n{json_schema_text}\n\n"
                )
            else:
                option_b_body = (
                    "If you want to accept the most recent terms offered by the other side, output ONLY the token '$DEAL_REACHED$' on the first line, "
                    "then output the agreed terms in JSON format on subsequent lines.\n"
                    f"JSON FORMAT: Output a JSON instance with actual values, NOT the schema definition.\n"
                    f"The schema below shows the structure - fill in the actual deal values:\n{json_schema_text}\n\n"
                )
            
            option_c = (
                "OPTION C: No Deal Possible\n"
                "If you believe no mutually beneficial deal is achievable, output the token '$DEAL_FAILED$'.\n\n"
            )
            

            formatted_prompt = (
                universal_continuation_prompt +
                option_a +
                option_b_intro + option_b_body +
                option_c +
                "Choose one option and respond accordingly. DO NOT output multiple options."
            )
            
            user_prompt = f"{context}\n\n"
            
            if self.use_memory and self.ai_memory:
                user_prompt += f"=== Current State Tracking Module ===\n{self.ai_memory}\n\n"
            
            if self.use_plan and self.ai_plan:
                user_prompt += f"=== Current Strategy Planning Module ===\n{self.ai_plan}\n\n"
            
            user_prompt += formatted_prompt
        
        messages = [
            {"role": "system", "content": ai_cfg["system_prompt"]},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.ai_agent.chat(messages)
        return response["content"]
    
    # ========================================================================
    # Core Method 2: Update Memory
    # ========================================================================
    def _update_memory(self):
        """Update memory state"""
        if not self.use_memory:
            return
        
        recent_transcript = last_round_window(self.transcript, k_rounds=2)
        
        ai_cfg = self.scenario_config[self.ai_role]
        student_cfg = self.scenario_config[self.student_role]
        
        ai_label = ai_cfg.get("label", self.ai_role)
        student_label = student_cfg.get("label", self.student_role)
        
        rules_objective = ai_cfg["system_prompt"]
        context = ai_cfg["context_prompt"]
        
        ai_round, ai_position = self.get_ai_next_round_info()
        position_text = "going first" if ai_position == 1 else "going second"
        turn_action = "start" if ai_position == 1 else "finish"  

        
        memory_system = (
            "You are a state tracking module for an AI negotiator. "
            "Produce a CONCISE, USABLE negotiation state that survives limited transcript windows.\n\n"
            "REQUIRED SECTIONS:\n"
            "OFFERS: [Us: <our latest position on each issue>; Them: <their latest position on each issue>; Them-best-for-us: <track complete offer they've proposed historically that has highest value to us>] (Format for Us/Them excluding Them-best-for-us: 'issue: old→new [updated]' if value changed this round; 'issue: value [unchanged]' if value unchanged from earlier rounds; 'issue: not yet' if never discussed by this party; If multiple offer packages proposed, state primary + note alternatives exist with their key trade-offs)\n"
            "OPPONENT PATTERNS: [Concession/firmness behaviors on issues; any acceptance of our proposals (specify full package or partial issues); strong commitments they stated; questions/requests they raised]\n"
            "OPPONENT PRIORITIES: [What matters to them across issues: mark EXPLICIT if they stated directly; mark Hypothesis if inferred from resistance/concession patterns]\n"
            "OPPONENT CONSTRAINTS: [Opponent's stated boundaries, red lines, or requirements]\n\n"
            "HARD RULES:\n"
            "1) PERSISTENCE – Keep long-lived critical facts (opponent's priorities, constraints, strong commitments, best historical offers they've proposed) even if they don't reappear in the latest window. Do NOT drop or change them unless updated, corrected or contradicted.\n"
            "2) EXPLICITNESS – Mark as EXPLICIT only if opponent truly stated; otherwise prefix 'Hypothesis:'.\n"
            "3) CANONICALIZATION – Strip unnecessary prose; use consistent formats; normalize numbers and terms for easy comparison.\n"
            "4) DEDUP & DELTA – Remove superseded info; show deltas as old→new.\n"
            "5) NO VERBATIM QUOTES – Summarize; do not copy transcript sentences.\n"
            "6) COUPLING– When recording conditional concessions, show conditional exchanges as 'X for Y' or 'will do X if we do Y' if stated or implied.\n"
            "7) NO FABRICATION – Do not invent facts with no basis in transcript or previous state. Mark absent information as 'not yet'.\n"
            "\n"
            "Output exactly these four sections in order. Aim for ≤10 lines; maximum 12 lines if complexity requires."
        )
        
        memory_user = (
            f"=== Identity ===\n"
            f"Role: State tracker of {ai_label}\n"
            f"You are now in round {ai_round} out of {self.total_rounds} total rounds. You are {position_text} this round, so it is your turn to {turn_action} the round.\n\n"
            f"=== Scenario Objective & Rules ===\n{rules_objective}\n\n"
            f"=== Scenario Facts & Value model ===\n{context}\n\n"
            f"=== Previous State ===\n{self.ai_memory or '(empty)'}\n\n"
            f"=== Recent Transcript Window ===\n{recent_transcript}\n\n"
            f"=== Output State ===\n"
            "Update the negotiation state now, following HARD RULES."
        )
        
        messages = [
            {"role": "system", "content": memory_system},
            {"role": "user", "content": memory_user}
        ]
        
        response = self.memory_agent.chat(messages)
        self.ai_memory = response["content"]
        # Append to history
        self.ai_memory_history.append({
            "round": f"{ai_round}.{ai_position}",
            "content": response["content"]
        })
    
    # ========================================================================
    # Core Method 3: Generate Plan
    # ========================================================================
    def _generate_plan(self):
        """Generate plan for this round"""
        if not self.use_plan:
            return
        
        ai_cfg = self.scenario_config[self.ai_role]
        context = ai_cfg["context_prompt"]
        rules_objective = ai_cfg["system_prompt"]
        ai_label = ai_cfg.get("label", self.ai_role)
        
        ai_round, ai_position = self.get_ai_next_round_info()
        position_text = "going first" if ai_position == 1 else "going second"
        turn_action = "start" if ai_position == 1 else "finish" 
        
        plan_system = (
            "You are a strategic planning module for an AI negotiator. "
            "Generate a SMART and ACTIONABLE plan for THIS round based on current state and context.\n\n"
            "OUTPUT SKELETON (aim for ≤10 lines; maximum 12 lines if complexity requires):\n"
            "- ROUND GOAL: <concrete objectives for this round; if accepting, write 'consider accepting current offer'>\n"
            "- KEY LEVERS: <issues where we have flexibility to trade or push; if none or accepting, write 'N/A'>\n"
            "- TACTICS: <specific actions this round; if accepting, write 'consider accepting current offer'>\n"
            "- OFFER SCAFFOLD: <our package to propose for this round; if accepting, refer to opponent's current offer>\n"
            "PLANNING RULES:\n"
            "- READ STATE – Use OFFERS, PATTERNS, PRIORITIES, and CONSTRAINTS from the State exactly as the source of truth.\n"
            "- ANCHORING – In early rounds, consider anchoring aggressively to create concession space.\n"
            "- VALUE MAXIMIZATION – Focus on securing high-value outcomes. When appropriate, consider probing for and using opponent's priorities and constraints to look for opportunities to trade lower-value items for ambitious and credible gains.\n"
            "- COMMUNICATION – When you want to explain your positions or respond to questions, consider providing brief rationale that frames your offers/response as reasonable. Be strategic: you could explain to build credibility, but avoid over-justifying or revealing weaknesses."
            "- ADAPT & NON-REPETITION – If last round's strategy didn't move the opponent, consider adapting tactics.\n"
            "- CLOSING DECISION – If you judge that now is the right time to close, consider accepting current offer.\n"
            "- HONOR ACCEPTANCES - If STATE indicates the opponent has fully and accurately accepted one of your proposed offers/packages, plan to reach the deal with those exact terms. Do not attempt to extract additional value.\n"
            "- WALKBACK RESISTANCE – If opponent proposes worse offer than their Them-best-for-us, resist by anchoring to that better historical offer.\n"
            "- BREVITY – Be crisp and operational: no fluff, no transcript restatement.\n\n"
        )
        
        plan_user = (
            f"=== Identity ===\n"
            f"Role: Strategy module of {ai_label}\n"
            f"You are now in round {ai_round} out of {self.total_rounds} total rounds. You are {position_text} this round, so it is your turn to {turn_action} the round.\n\n"
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
        # Append to history
        self.ai_plan_history.append({
            "round": f"{ai_round}.{ai_position}",
            "content": response["content"]
        })
    
    # ========================================================================
    # Core Method 4: Process Student Message (⭐ v5 FINAL)
    # ========================================================================
    def process_student_message(self, message: str) -> Dict[str, Any]:
        """
        Process student message
        
        v5 FINAL - Simplified round management:
        - Use len(transcript) to calculate everything
        - No more messages_in_current_round or current_round tracking
        - Check round completion BEFORE generating AI response
        """
        student_cfg = self.scenario_config[self.student_role]
        student_label = student_cfg.get("label", "Student")
        ai_cfg = self.scenario_config[self.ai_role]
        ai_label = ai_cfg.get("label", "AI")
        
        # Calculate round info for student's message
        transcript_len = len(self.transcript)
        next_message_num = transcript_len + 1

        # Simplified: message numbering is always 1.1, 1.2, 2.1, 2.2, ... regardless of who goes first
        round_num = (next_message_num + 1) // 2
        message_in_round = 1 if next_message_num % 2 == 1 else 2
        
        # Save student message with round info
        self.transcript.append(f"Round {round_num}.{message_in_round} - {student_label}: {message}")
        
        # ========================================================================
        # SECTION A: Check if AI proposed deal in last message
        # ========================================================================
        last_ai_message = None
        for msg in reversed(self.transcript[:-1]):
            if f"- {ai_label}:" in msg:
                last_ai_message = msg
                break
        
        ai_just_proposed_deal = False
        if last_ai_message and f"- {ai_label}:" in last_ai_message:
            message_content = last_ai_message.split(f"- {ai_label}:", 1)[1].strip()
            ai_just_proposed_deal = message_content.startswith("$DEAL_REACHED$")
        
        if ai_just_proposed_deal:
            if message.strip().upper().startswith("$DEAL_REACHED$"):
                student_json = extract_json_from_text(message)
                ai_json = extract_json_from_text(last_ai_message)
                
                if student_json and ai_json:
                    if are_deals_equivalent(student_json, ai_json):
                        self.student_deal_json = student_json
                        self.ai_deal_json = ai_json
                        self.deal_reached = True
                        self.status = "completed"
                        return {
                            "ai_message": None,
                            "deal_reached": True,
                            "deal_terms": filter_private_fields(ai_json),
                            "system_message": "✅ Deal confirmed! Both parties agree on the terms.",
                            "round": f"{round_num}.{message_in_round}"
                        }
                    else:
                        self.status = "failed"
                        return {
                            "ai_message": None,
                            "deal_reached": False,
                            "terms_mismatch": True,
                            "system_message": "❌ Both parties indicated DEAL_REACHED, but the final terms do not match. Negotiation ended.",
                            "round": f"{round_num}.{message_in_round}"
                        }
                else:
                    return {
                        "ai_message": None,
                        "deal_reached": False,
                        "json_parse_error": True,
                        "system_message": "❌ Could not parse JSON from your message. Please check the format and try again.",
                        "round": f"{round_num}.{message_in_round}"
                    }
            
            elif message.strip().upper().startswith("$DEAL_MISUNDERSTANDING$"):
                self.status = "failed"
                return {
                    "ai_message": None,
                    "deal_reached": False,
                    "misunderstanding": True,
                    "system_message": "❌ You indicated that the AI's proposed terms do not match your most recent offer. Negotiation ended.",
                    "round": f"{round_num}.{message_in_round}"
                }
            
            else:
                return {
                    "ai_message": None,
                    "deal_reached": False,
                    "invalid_response": True,
                    "system_message": "⚠️ The AI has proposed a deal. You must either:\n• Accept with '$DEAL_REACHED$' followed by JSON terms\n• Reject with '$DEAL_MISUNDERSTANDING$'\n\nNo other responses are allowed at this point.",
                    "round": f"{round_num}.{message_in_round}"
                }
        
        # ========================================================================
        # SECTION B: Check if student proposes deal
        # ========================================================================
        if message.strip().upper().startswith("$DEAL_REACHED$"):
            ai_confirm_response = self._request_ai_deal_confirmation(message)
            
            # Add AI confirmation with same round number (synchronous)
            self.transcript.append(f"Round {round_num}.{message_in_round} - {ai_label}: {ai_confirm_response}")
            
            if ai_confirm_response.strip().startswith("$DEAL_REACHED$"):
                student_json = extract_json_from_text(message)
                ai_json = extract_json_from_text(ai_confirm_response)
                
                if student_json and ai_json:
                    if are_deals_equivalent(student_json, ai_json):
                        self.student_deal_json = student_json
                        self.ai_deal_json = ai_json
                        self.deal_reached = True
                        self.status = "completed"
                        return {
                            "ai_message": ai_confirm_response,
                            "deal_reached": True,
                            "deal_terms": filter_private_fields(ai_json),
                            "system_message": "✅ Deal confirmed! Both parties agree on the terms.",
                            "round": f"{round_num}.{message_in_round}"
                        }
                    else:
                        self.status = "failed"
                        return {
                            "ai_message": ai_confirm_response,
                            "deal_reached": False,
                            "terms_mismatch": True,
                            "system_message": "❌ Both parties indicated DEAL_REACHED, but the final terms do not match. Negotiation ended.",
                            "round": f"{round_num}.{message_in_round}"
                        }
                else:
                    if ai_json:
                        self.ai_deal_json = ai_json
                        self.deal_reached = True
                        self.status = "completed"
                        return {
                            "ai_message": ai_confirm_response,
                            "deal_reached": True,
                            "deal_terms": filter_private_fields(ai_json),
                            "system_message": "✅ Deal confirmed! (Using AI's parsed terms)",
                            "round": f"{round_num}.{message_in_round}"
                        }
            
            elif ai_confirm_response.strip().startswith("$DEAL_MISUNDERSTANDING$"):
                self.status = "failed"
                return {
                    "ai_message": ai_confirm_response,
                    "deal_reached": False,
                    "misunderstanding": True,
                    "system_message": "❌ AI believes the terms you specified do not match its most recent offer. Negotiation ended.",
                    "round": f"{round_num}.{message_in_round}"
                }
        
        # ========================================================================
        # SECTION C: Check if student declares deal failed
        # ========================================================================
        if message.strip().upper().startswith("$DEAL_FAILED$"):
            self.deal_failed = True
            self.status = "failed"
            return {
                "ai_message": "You have indicated that no deal is possible. Negotiation ended.",
                "deal_failed": True,
                "system_message": "❌ You declared that no deal is possible. Negotiation ended.",
                "round": f"{round_num}.{message_in_round}"
            }
        
        # ========================================================================
        # SECTION D: Normal negotiation - generate AI response
        # ========================================================================
        
        # Check if round is complete BEFORE generating AI response
        current_transcript_len = len(self.transcript)
        round_number, is_round_complete = calculate_round_info(current_transcript_len, self.student_goes_first)
        
        if is_round_complete and round_number >= self.total_rounds:
            # Calculate which message (.1 or .2) caused the end
            last_msg_in_round = 1 if current_transcript_len % 2 == 1 else 2
            self.status = "completed"
            return {
                "ai_message": None,
                "negotiation_ended": True,
                "reason": "max_rounds_reached",
                "system_message": "⏱️ Maximum rounds reached. Negotiation ended without a deal.",
                "round": f"{round_number}.{last_msg_in_round}"
            }
        
        # Update Memory & Plan
        if len(self.transcript) >= 1:
            self._update_memory()
            self._generate_plan()
        
        # Generate AI response
        ai_response = self._generate_ai_response()
        
        # Calculate AI's round info
        ai_next_message_num = len(self.transcript) + 1

        # Simplified: Round numbering is 1.1, 1.2, 2.1, 2.2, ... regardless of who goes first
        ai_round_num = (ai_next_message_num + 1) // 2
        ai_message_in_round = 1 if ai_next_message_num % 2 == 1 else 2
        
        self.transcript.append(f"Round {ai_round_num}.{ai_message_in_round} - {ai_label}: {ai_response}")
        
        # Check round status after AI response
        new_transcript_len = len(self.transcript)
        round_after, is_complete_after = calculate_round_info(new_transcript_len, self.student_goes_first)
        
        # Check if AI proposed deal
        if ai_response.strip().startswith("$DEAL_REACHED$"):
            ai_json = extract_json_from_text(ai_response)
            if ai_json:
                self.ai_deal_json = ai_json
                allow_overtime = (is_complete_after and round_after >= self.total_rounds)
                
                return {
                    "ai_message": ai_response,
                    "ai_proposed_deal": True,
                    "ai_deal_terms": filter_private_fields(ai_json),
                    "round": f"{ai_round_num}.{ai_message_in_round}",
                    "allow_overtime": allow_overtime
                }
        
        # Check if AI declared deal failed
        if ai_response.strip().startswith("$DEAL_FAILED$"):
            self.deal_failed = True
            self.status = "failed"
            return {
                "ai_message": ai_response,
                "deal_failed": True,
                "system_message": "❌ AI declared that no deal is possible. Negotiation ended.",
                "round": f"{ai_round_num}.{ai_message_in_round}"
            }
        
        # Check if round completed after AI's normal response
        if is_complete_after and round_after >= self.total_rounds:
            self.status = "completed"
            return {
                "ai_message": ai_response,
                "negotiation_ended": True,
                "reason": "max_rounds_reached",
                "system_message": "⏱️ Maximum rounds reached. Negotiation ended without a deal.",
                "round": f"{ai_round_num}.{ai_message_in_round}"
            }
        
        return {
            "ai_message": ai_response,
            "deal_reached": False,
            "round": f"{ai_round_num}.{ai_message_in_round}"
        }
    
    # ========================================================================
    # Core Method 5: AI Confirm Deal
    # ========================================================================
    def _request_ai_deal_confirmation(self, student_message: str) -> str:
        """Request AI to confirm student's proposed deal"""
        ai_cfg = self.scenario_config[self.ai_role]
        context = ai_cfg["context_prompt"]
        
        history = "\n\n".join(self.transcript) 
        
        json_schema_raw = self.scenario_config["json_schema"]
        json_schema_dict = json.loads(json_schema_raw)
        json_schema_text = "\n" + json.dumps(json_schema_dict, indent=2)
        
        scenario_lower = self.scenario_name.lower()
        
        if scenario_lower == "top_talent":
            value_field_instruction = "Fill in 'total_points_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
        elif scenario_lower in ("z_deal", "zlab_split"):
            value_field_instruction = "Fill in 'expected_value_of_deal_to_me_in_millions' by calculating your own value using your private scoring rules described in your given prompts."
        elif scenario_lower in ("twisted_tree", "vb_development", "main_street"):
            value_field_instruction = ""
        else:
            value_field_instruction = "Fill in 'total_value_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
        
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
            conf_prompt += f"Output ONLY the token '$DEAL_REACHED$' on the first line, then output the agreed terms in JSON format on subsequent lines. {value_field_instruction}\n"
        else:
            conf_prompt += "OPTION 1: If the terms ARE consistent with your most recent offer:\n"
            conf_prompt += "Output ONLY the token '$DEAL_REACHED$' on the first line, then output the agreed terms in JSON format on subsequent lines.\n"
        
        conf_prompt += (
            f"JSON FORMAT: Output a JSON instance with actual values, NOT the schema definition.\n"
            f"The schema below shows the structure - fill in the actual deal values:\n{json_schema_text}\n\n"
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
    # Core Method 6: Generate Feedback (⭐ IMPROVED v5.1)
    # ========================================================================
    def generate_feedback(self) -> str:
        """
        Generate negotiation feedback
        
        v7 - Simplified & High Quality:
        - Use context_prompt directly (like plan/memory do)
        - Remove outcome determination (it's in transcript)
        - 3 focused sections instead of 5
        - Concise, high-signal feedback (400-500 words)
        """
        feedback_agent = OpenAIWrapper(model=self.ai_model, label="FeedbackCoach")
        
        # Build complete transcript (strip round info)
        transcript_lines = []
        for msg in self.transcript:
            if " - " in msg:
                parts = msg.split(" - ", 1)
                if len(parts) == 2:
                    transcript_lines.append(parts[1])
                else:
                    transcript_lines.append(msg)
            else:
                transcript_lines.append(msg)
        complete_transcript = "\n\n".join(transcript_lines)
        
        student_cfg = self.scenario_config[self.student_role]
        ai_cfg = self.scenario_config[self.ai_role]
        
        # Use context_prompt directly (like plan and memory do)
        context = student_cfg["context_prompt"]
        rules_objective = student_cfg["system_prompt"]
        
        current_round = self.get_current_round()

        feedback_system = (
            "You are a negotiation coach providing CONCISE, high-quality feedback to MBA students.\n\n"
            
            "CORE PRINCIPLE:\n"
            "The student's score comes ONLY from options explicitly defined in their context. "
            "Never suggest strategies, terms, or compromises that don't have explicit point/monetary values in the scenario.\n\n"
            
            "FORBIDDEN:\n"
            "- Creating new issues or terms not in the scenario\n"
            "- Suggesting organizational processes not mentioned\n"
            "- Proposing staged/phased deals unless those are scored options\n"
            "- Inventing creative workarounds that aren't actually available\n\n"

            "OUTPUT FORMAT (Plain Text with Visual Structure):\n"
            "- Use section dividers: ═══ for major sections\n"
            "- Use emojis for visual anchors (📊 🎯 💡)\n"
            "- Keep paragraphs SHORT (2-3 sentences max)\n"
            "- Use clear spacing: blank lines between sections\n"
            "- Use bullet points (•) for lists\n\n"
            
            "FEEDBACK STRUCTURE:\n\n"
            
            "═══════════════════════════════════════\n"
            "📊 PERFORMANCE SUMMARY\n"
            "═══════════════════════════════════════\n\n"
            "- Final outcome: total points/value achieved (or BATNA if no deal)\n"
            "- Overall assessment\n\n\n"
            
            "═══════════════════════════════════════\n"
            "🎯 STRATEGIC ANALYSIS\n"
            "═══════════════════════════════════════\n\n"
            
            "For major issues, analyze BOTH value and tactics:\n"
            "• What you got vs. what was possible (specific points/dollars/value)\n"
            "• WHY you got this result (your negotiation moves)\n"
            "• What different tactics could have captured more value\n\n"
            
            "Focus on:\n"
            "• Anchoring: Did you open strong on high-value issues?\n"
            "• Information: Did you discover opponent's priorities before conceding?\n"
            "• Trade-offs: Did you trade low-value items for high-value gains?\n"
            "• Timing: Did you concede too early or hold out effectively?\n"
            "• Patterns: Did opponent exploit your concession patterns?\n\n\n"
            
            "═══════════════════════════════════════\n"
            "💡 KEY IMPROVEMENT\n"
            "═══════════════════════════════════════\n\n"
            
            "One concrete tactical change that would have gained the most value:\n"
            "• Specific negotiation move (e.g., 'anchor at X instead of Y')\n"
            "• Which issue to apply it to\n"
            "• Expected value gain from this tactic\n"
        )

        feedback_user = (
            f"SCENARIO: {self.scenario_name}\n"
            f"STUDENT: {student_cfg.get('label', self.student_role)}\n"
            f"OPPONENT: {ai_cfg.get('label', self.ai_role)}\n"
            f"NEGOTIATION LENGTH: Ended in round {current_round} (max {self.total_rounds} rounds)\n\n"    
            f"OBJECTIVES & RULES:\n{rules_objective}\n\n"
            f"CONTEXT & VALUE MODEL:\n{context}\n\n"
            f"TRANSCRIPT:\n{complete_transcript}\n\n"
            "=== Generate Feedback ===\n"
            
        )
        
        messages = [
            {"role": "system", "content": feedback_system},
            {"role": "user", "content": feedback_user}
        ]
        
        response = feedback_agent.chat(messages)
        
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
                total_rounds,
                transcript,
                ai_memory,
                ai_plan,
                ai_memory_history,
                ai_plan_history,
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
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
            self.total_rounds,
            json.dumps(self.transcript),
            self.ai_memory,
            self.ai_plan,
            json.dumps(self.ai_memory_history),
            json.dumps(self.ai_plan_history),
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
            total_rounds=row[10]
        )
        
        session.ai_role = row[5]
        session.transcript = json.loads(row[11])
        session.ai_memory = row[12] or ""
        session.ai_plan = row[13] or ""
        session.ai_memory_history = json.loads(row[14]) if row[14] else []
        session.ai_plan_history = json.loads(row[15]) if row[15] else []

        session.student_deal_json = json.loads(row[16]) if row[16] else None  
        session.ai_deal_json = json.loads(row[17]) if row[17] else None       
        session.deal_reached = bool(row[18])                                   
        session.deal_failed = bool(row[19])                                    
        session.status = row[20]                                               
        session.created_at = row[21]                                           
        session.updated_at = row[22]                                           
        session.feedback_text = row[23]                                        
        session.feedback_generated_at = row[24]                               
        session.feedback_model = row[25]                                       
        
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
            "current_round": self.get_current_round(),
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
    total_rounds: int = 10

class SendMessageRequest(BaseModel):
    message: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Negotiation Practice API", "version": "v5-final"}

@app.get("/scenarios")
def list_scenarios():
    """List available negotiation scenarios"""
    scenarios_path = Path(SCENARIOS_DIR)
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
            total_rounds=request.total_rounds
        )
        
        ai_first_message = None
        if not student_goes_first:
            ai_response = session._generate_ai_response()
            ai_cfg = session.scenario_config[session.ai_role]
            ai_label = ai_cfg.get("label", "AI")
            session.transcript.append(f"Round 1.1 - {ai_label}: {ai_response}")
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
        "current_round": session.get_current_round(),
        "total_rounds": session.total_rounds,
        "deal_reached": session.deal_reached,
        "deal_failed": session.deal_failed,
        "status": session.status
    }

@app.get("/negotiation/{session_id}/role_info")
def get_role_info(session_id: str):
    """Get the student's complete role information for display"""
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
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA wal_checkpoint(FULL)")
    conn.close()

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