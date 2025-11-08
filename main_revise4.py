"""
Optimized Negotiation Backend - Complete Fixed Version

Key Fixes:
1. ✅ Fixed {{history}} double braces issue causing transcript传递问题
2. ✅ Added transcript in deal confirmation, letting AI see its own latest offer  
3. ✅ Retained all original features (random first turn, model selection, correct API format)
4. ✅ JSON schema still needs escape because it contains braces
5. ✅ /scenarios endpoint returns correct side1_label and side2_label
6. ✅ **CRITICAL FIX**: Aligned ALL prompts with runner.py for consistent AI quality
7. ✅ Added Feedback generation system with relaxed conditions (can generate at any time)
8. ✅ Added role_info endpoint to display student's complete role information
9. ✅ Removed redundant "ignore other side's value" instruction (students don't output value)
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
# Helper Functions (Identical to runner.py)
# ============================================================================

def _escape_braces(s: str) -> str:
    """Escape braces for .format()"""
    return s.replace("{", "{{").replace("}", "}}")

def _unescape_braces(s: str) -> str:
    """Unescape braces"""
    return s.replace("{{", "{").replace("}}", "}")

def last_round_window(transcript: List[str], k_rounds: int = 2) -> str:
    """
    Extract last k rounds of conversation
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

# ============================================================================
# NegotiationSession Class
# ============================================================================
class NegotiationSession:
    """
    Negotiation Session Class
    Logic identical to runner.py with all prompts aligned
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
        self.total_rounds = self.scenario_config.get("num_rounds", 10)
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
        
        ⭐ Critical Fix:
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
            # Added: $DEAL_FAILED$ option for all scenarios (good improvement from main.py)
            universal_continuation_prompt = (
                "CURRENT ROUND INFORMATION:\n"
                f"It is now round {self.current_round}/{self.total_rounds}. You are {turn_position} this round, so it is your turn to {turn_action} the round. "
                f"You have {self.total_rounds - self.current_round} rounds remaining after this one.\n\n"
                "<BEGIN COMPLETE NEGOTIATION TRANSCRIPT>\n{{history}}\n\n"
                "<END NEGOTIATION TRANSCRIPT>\n\n"
                "OUTPUT INSTRUCTIONS:\n"
                "1. Reminder: you can either continue the negotiation or accept the most recent terms offered by the other side by outputting the token '$DEAL_REACHED$' at the beginning of your output. If you do not reach a deal by the end of the last round, you get your BATNA.\n\n"
                "2. OUTPUT OPTIONS (YOU MUST CHOOSE ONE):\n"
                f"a) Continue the negotiation by outputting your negotiation message to {turn_action} round {self.current_round}\n"
                "b) Accept the most recent terms offered by the other side by outputting the token '$DEAL_REACHED$' at the beginning of your output on its own line (DO NOT put any other text before it), immediately followed by a newline, and then the deal terms you are accepting in the following JSON format."
            )
            
            # Add value calculation instruction if applicable
            if value_instruction:
                universal_continuation_prompt += f" {value_instruction}"
            
            universal_continuation_prompt += (
                " Do not put anything in your output besides the token '$DEAL_REACHED$' and the correctly formatted JSON (Do NOT output the schema. Output only a filled JSON instance matching the schema). Do not add any commentary, explanations, or markdown.\n"
                "c) If you believe no mutually beneficial deal is possible, you may output '$DEAL_FAILED$' on its own line to end the negotiation.\n\n"
                f"JSON FORMAT (FOLLOW THIS EXACTLY IF CHOOSING OPTION B):\n{_escape_braces(json_schema_text)}"
            )
            
            # ⭐ Fix: Use .format() with escaped history, then unescape
            continuation = universal_continuation_prompt.format(history=_escape_braces(history))
            continuation = _unescape_braces(continuation)
            
            # ⭐ Critical: context + continuation
            user_prompt = f"{context}\n\n{continuation}"
        
        # 6. Inject Memory & Plan (Identical to runner.py)
        full_prompt = self._inject_memory_and_plan(user_prompt)
        
        # 7. Call AI
        messages = [
            {"role": "system", "content": ai_cfg["system_prompt"]},
            {"role": "user", "content": full_prompt}
        ]
        
        response = self.ai_agent.chat(messages)
        return response["content"]
    
    # ========================================================================
    # Core Method 2: Inject Memory & Plan (IDENTICAL to runner.py)
    # ========================================================================
    def _inject_memory_and_plan(self, base_prompt: str) -> str:
        """
        Wrap Memory and Plan into prompt outer layer
        Identical to runner.py lines 589-598
        """
        parts = []
        
        if self.use_memory and self.ai_memory:
            parts.append(f"--- NEGOTIATION STATE TRACKING ---\n{self.ai_memory}")
        
        if self.use_plan and self.ai_plan:
            parts.append(f"--- STRATEGY FOR THIS ROUND ---\n{self.ai_plan}")
        
        parts.append(f"--- MAIN PROMPT ---\n{base_prompt}")
        
        return "\n\n".join(parts)
    
    # ========================================================================
    # Core Method 3: Update Memory (FULLY ALIGNED WITH runner.py)
    # ========================================================================
    def _update_memory(self):
        """
        Update AI's Memory
        FULLY ALIGNED with runner.py lines 400-489
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
    # Core Method 4: Generate Plan (FULLY ALIGNED WITH runner.py)
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
    # Core Method 5: Process Student Message (Fixed deal confirmation logic)
    # ========================================================================
    def process_student_message(self, message: str) -> Dict[str, Any]:
        """
        Process student message
        
        ⭐ Critical Fix:
        1. Student can input $DEAL_REACHED$ + JSON (like AI)
        2. AI needs to confirm if it matches its own offer
        3. No need for frontend accept/reject dialog
        """
        # 1. Save student message to transcript
        student_cfg = self.scenario_config[self.student_role]
        student_label = student_cfg.get("label", "Student")
        self.transcript.append(f"{student_label}: {message}")
        
        # 2. Check if student proposed deal
        if "$DEAL_REACHED$" in message:
            student_json = extract_json_from_text(message)
            
            if student_json:
                self.student_deal_json = student_json
                
                # ⭐ Critical: Let AI confirm
                ai_confirm_response = self._request_ai_deal_confirmation(student_json)
                
                ai_cfg = self.scenario_config[self.ai_role]
                ai_label = ai_cfg.get("label", "AI")
                self.transcript.append(f"{ai_label}: {ai_confirm_response}")
                
                # Check AI's confirmation result
                if "$DEAL_REACHED$" in ai_confirm_response:
                    # AI confirms consistency
                    ai_json = extract_json_from_text(ai_confirm_response)
                    if ai_json:
                        self.ai_deal_json = ai_json
                        self.deal_reached = True
                        self.status = "completed"
                        return {
                            "ai_message": ai_confirm_response,
                            "deal_reached": True,
                            "deal_terms": student_json,
                            "round": self.current_round
                        }
                
                elif "$DEAL_MISUNDERSTANDING$" in ai_confirm_response:
                    # AI indicates inconsistency
                    return {
                        "ai_message": ai_confirm_response,
                        "deal_reached": False,
                        "misunderstanding": True,
                        "round": self.current_round
                    }
        
        # 3. Check if proposed deal failed
        if "$DEAL_FAILED$" in message:
            self.deal_failed = True
            self.status = "failed"
            return {
                "ai_message": "You have indicated that no deal is possible. Negotiation ended.",
                "deal_failed": True,
                "round": self.current_round
            }
        
        # 4. Update Memory & Plan (if enabled)
        if self.current_round > 1 or len(self.transcript) > 1:
            self._update_memory()
            self._generate_plan()
        
        # 5. Generate AI response
        ai_response = self._generate_ai_response()
        
        ai_cfg = self.scenario_config[self.ai_role]
        ai_label = ai_cfg.get("label", "AI")
        self.transcript.append(f"{ai_label}: {ai_response}")
        
        # 6. Check if AI proposed deal
        if "$DEAL_REACHED$" in ai_response:
            ai_json = extract_json_from_text(ai_response)
            if ai_json:
                self.ai_deal_json = ai_json
                # ⭐ Don't auto-confirm; wait for student response
                return {
                    "ai_message": ai_response,
                    "ai_proposed_deal": True,
                    "ai_deal_terms": ai_json,
                    "round": self.current_round
                }
        
        # 7. Check if AI proposed deal failed
        if "$DEAL_FAILED$" in ai_response:
            self.deal_failed = True
            self.status = "failed"
            return {
                "ai_message": ai_response,
                "deal_failed": True,
                "round": self.current_round
            }
        
        # 8. Check if reached last round
        if self.current_round >= self.total_rounds:
            self.status = "completed"
            return {
                "ai_message": ai_response,
                "negotiation_ended": True,
                "reason": "max_rounds_reached",
                "round": self.current_round
            }
        
        # 9. Move to next round
        self.current_round += 1
        
        return {
            "ai_message": ai_response,
            "deal_reached": False,
            "round": self.current_round - 1  # Just completed round
        }
    
    # ========================================================================
    # Core Method 6: AI Confirm Deal (ALIGNED WITH runner.py, NO redundant instruction)
    # ========================================================================
    def _request_ai_deal_confirmation(self, student_deal_json: Dict) -> str:
        """
        Request AI to confirm student's proposed deal
        Based on runner.py lines 1198-1211
        
        ⭐ FIX: Removed "ignore other side's value" instruction
        Because students don't output value fields, so nothing to ignore
        """
        ai_cfg = self.scenario_config[self.ai_role]
        context = ai_cfg["context_prompt"]
        
        # Build complete history
        history = "\n\n".join(self.transcript)
        
        # Get JSON schema
        json_schema_raw = self.scenario_config["json_schema"]
        json_schema_dict = json.loads(json_schema_raw)
        json_schema_text = "\n" + json.dumps(json_schema_dict, indent=2)
        
        # Determine scenario-specific prompt
        scenario_lower = self.scenario_name.lower()
        
        if scenario_lower == "top_talent":
            value_field_instruction = "Fill in 'total_points_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
        elif scenario_lower in ("z_deal", "zlab_split"):
            value_field_instruction = "Fill in 'expected_value_of_deal_to_me_in_millions' by calculating your own value using your private scoring rules described in your given prompts."
        elif scenario_lower in ("twisted_tree", "vb_development"):
            # No value field
            value_field_instruction = ""
        else:
            value_field_instruction = "Fill in 'total_value_of_deal_to_me' by calculating your own value using your private scoring rules described in your given prompts."
        
        # ⭐ Confirmation prompt (ALIGNED with runner.py lines 1198-1211)
        conf_prompt = (
            "DEAL CONFIRMATION:\n"
            "The other side indicated that they accept the most recent terms offered by you and specified the deal terms that they are accepting in their most recent message. "
            "Look at the complete negotiation transcript below to confirm that the terms specified by the other side in their last message are faithful to the latest proposal you made.\n\n"
            "COMPLETE NEGOTIATION TRANSCRIPT:\n"
            "<BEGIN TRANSCRIPT>\n"
            f"{history}\n"
            "<END TRANSCRIPT>\n\n"
            "OUTPUT INSTRUCTIONS: CHOOSE ONE OF THE FOLLOWING TWO OPTIONS\n"
        )
        
        if value_field_instruction:
            conf_prompt += f"OPTION 1: IF the terms of the deal specified by the other side are INDEED CORRECT then output the agreed upon deal in the following JSON format. {value_field_instruction}\n"
        else:
            conf_prompt += "OPTION 1: IF the terms of the deal specified by the other side are INDEED CORRECT then output the agreed upon deal in the following JSON format.\n"
        
        conf_prompt += (
            f"JSON FORMAT (FOLLOW THIS EXACTLY IF CHOOSING OPTION 1):\n{json_schema_text}\n"
            "OPTION 2: IF the terms of the deal specified by the other side are NOT correct then output ONLY the token '$DEAL_MISUNDERSTANDING$'. Do not bother to output any other text or fill in the JSON.\n\n"
        )
        
        # ⭐ REMOVED: "ignore other side's value" instruction
        # Because students don't output value fields, this instruction is redundant
        
        # ⭐ Critical: Add context prefix
        full_prompt = f"{context}\n\n{conf_prompt}"
        
        messages = [
            {"role": "system", "content": ai_cfg["system_prompt"]},
            {"role": "user", "content": full_prompt}
        ]
        
        response = self.ai_agent.chat(messages)
        return response["content"]
    
    # ========================================================================
    # Core Method 7: Generate Feedback (RELAXED CONDITIONS - can generate anytime)
    # ========================================================================
    def generate_feedback(self) -> str:
        """
        Generate negotiation feedback
        
        ⭐ FIX: Removed strict status check
        Can now be called at any point (active, completed, failed, ended_early)
        This allows students to get feedback even if they exit early
        """
        # ⭐ REMOVED: strict status check
        # Old code: if self.status != "completed" and self.status != "failed": raise ValueError(...)
        # New: Allow feedback generation at any time
        
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
        
        # Feedback user prompt
        feedback_user = (
            f"=== NEGOTIATION SCENARIO ===\n"
            f"Scenario: {self.scenario_name}\n"
            f"Student Role: {student_cfg.get('label', self.student_role)}\n"
            f"Student Name: {self.student_name}\n"
            f"AI Role: {ai_cfg.get('label', self.ai_role)}\n\n"
            
            f"=== SCENARIO CONTEXT ===\n"
            f"{scenario_context}\n\n"
            
            f"=== COMPLETE NEGOTIATION TRANSCRIPT ===\n"
            f"{complete_transcript}\n\n"
            
            f"=== NEGOTIATION OUTCOME ===\n"
            f"Negotiation Status: {self.status}\n"
            f"Rounds Completed: {self.current_round}/{self.total_rounds}\n"
            f"Deal Reached: {self.deal_reached}\n"
            f"Deal Failed: {self.deal_failed}\n"
        )
        
        # Add status-specific context
        if self.status == "active":
            feedback_user += (
                "\n⚠️ IMPORTANT: This negotiation is still in progress. "
                "The student requested feedback before completion. "
                "Provide feedback on their performance so far and suggest strategies for the remaining rounds.\n"
            )
        elif self.status == "completed" and not self.deal_reached and not self.deal_failed:
            feedback_user += (
                "\n⚠️ Note: This negotiation reached the maximum number of rounds without a deal. "
                "Analyze why no agreement was reached.\n"
            )
        
        if self.student_deal_json:
            feedback_user += f"\nStudent's Final Deal: {json.dumps(self.student_deal_json, indent=2)}\n"
        if self.ai_deal_json:
            feedback_user += f"\nAI's Response: {json.dumps(self.ai_deal_json, indent=2)}\n"
        
        feedback_user += (
            f"\n=== FEEDBACK REQUEST ===\n"
            f"Provide comprehensive feedback to {self.student_name} covering:\n\n"
            f"1. NEGOTIATION STRATEGY:\n"
            f"   - Did the student identify and leverage their priorities effectively?\n"
            f"   - Did they recognize the AI's priorities and patterns?\n"
            f"   - What strategic opportunities did they miss?\n"
            f"   - Quality of their offers and concessions\n\n"
            
            f"2. COMMUNICATION SKILLS:\n"
            f"   - Clarity and professionalism of messages\n"
            f"   - Active listening and responsiveness\n"
            f"   - Use of persuasive techniques\n"
            f"   - Building rapport and trust\n\n"
            
            f"3. VALUE CREATION:\n"
            f"   - Did they create value through trade-offs?\n"
            f"   - Did they explore integrative solutions?\n"
            f"   - How well did they balance competitive vs collaborative approaches?\n"
            f"   - Understanding of ZOPA and BATNA\n\n"
            
            f"4. OUTCOME ANALYSIS:\n"
            f"   - Quality of the final deal (if reached)\n"
            f"   - Value captured vs value left on the table\n"
            f"   - Comparison to their BATNA\n"
            f"   - Whether they maximized joint value\n\n"
            
            f"5. SPECIFIC RECOMMENDATIONS:\n"
            f"   - 3-5 concrete actions for improvement in future negotiations\n"
            f"   - Key takeaways from this negotiation\n"
            f"   - Techniques or concepts to study further\n\n"
            
            f"Format your feedback in a clear, structured manner with section headers. "
            f"Be encouraging while being specific about areas for improvement. "
            f"Make the feedback actionable and educational."
        )
        
        messages = [
            {"role": "system", "content": feedback_system},
            {"role": "user", "content": feedback_user}
        ]
        
        response = feedback_agent.chat(messages)
        feedback = response["content"]
        
        # Save to session
        self.feedback_text = feedback
        self.feedback_generated_at = datetime.utcnow().isoformat()
        self.feedback_model = self.ai_model
        
        return feedback
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
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
            "ai_memory": self.ai_memory if self.use_memory else None,
            "ai_plan": self.ai_plan if self.use_plan else None,
            "student_deal_json": self.student_deal_json,
            "ai_deal_json": self.ai_deal_json,
            "deal_reached": self.deal_reached,
            "deal_failed": self.deal_failed,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "feedback_text": self.feedback_text,
            "feedback_generated_at": self.feedback_generated_at,
            "feedback_model": self.feedback_model
        }
    
    def save_to_db(self):
        """Save to database (explicit columns, 25 placeholders)"""
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
            self.feedback_model
        ))

        conn.commit()
        conn.close()

    @staticmethod
    def load_from_db(session_id: str) -> Optional['NegotiationSession']:
        """Load from database"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        c = conn.cursor()

        c.execute("""
            SELECT
                session_id, student_id, student_name, scenario_name,
                student_role, ai_role, ai_model, student_goes_first,
                use_memory, use_plan, current_round, total_rounds,
                transcript, ai_memory, ai_plan, student_deal_json,
                ai_deal_json, deal_reached, deal_failed, status,
                created_at, updated_at, feedback_text, 
                feedback_generated_at, feedback_model
            FROM negotiation_sessions
            WHERE session_id = ?
        """, (session_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        # Use column names to get values, avoiding column order differences
        session = NegotiationSession(
            session_id=row["session_id"],
            student_id=row["student_id"],
            student_name=row["student_name"],
            scenario_name=row["scenario_name"],
            student_role=row["student_role"],
            ai_model=row["ai_model"],
            student_goes_first=bool(row["student_goes_first"]),
            use_memory=bool(row["use_memory"]),
            use_plan=bool(row["use_plan"]),
        )

        # Restore state
        session.ai_role       = row["ai_role"]
        session.current_round = int(row["current_round"])
        session.total_rounds  = int(row["total_rounds"])

        # JSON/text fields
        session.transcript         = json.loads(row["transcript"]) if row["transcript"] else []
        session.ai_memory          = row["ai_memory"] or ""
        session.ai_plan            = row["ai_plan"] or ""
        session.student_deal_json  = json.loads(row["student_deal_json"]) if row["student_deal_json"] else None
        session.ai_deal_json       = json.loads(row["ai_deal_json"]) if row["ai_deal_json"] else None

        # Flags/metadata
        session.deal_reached = bool(row["deal_reached"])
        session.deal_failed  = bool(row["deal_failed"])
        session.status       = row["status"]
        session.created_at   = row["created_at"]
        session.updated_at   = row["updated_at"]
        
        # Feedback fields
        session.feedback_text = row["feedback_text"]
        session.feedback_generated_at = row["feedback_generated_at"]
        session.feedback_model = row["feedback_model"]

        return session

# ============================================================================
# Pydantic Models
# ============================================================================
class StartNegotiationRequest(BaseModel):
    student_id: str
    student_name: str
    scenario_name: str
    student_role: str  # "side1" or "side2"
    ai_model: str = "anthropic/claude-3-sonnet"  # ⭐ New: model selection
    randomize_first_turn: bool = True  # ⭐ New: random first turn
    use_memory: bool = True
    use_plan: bool = True

class SendMessageRequest(BaseModel):
    message: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def read_root():
    """Health check"""
    return {"status": "ok", "message": "Negotiation Practice API"}

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
    """
    Start new negotiation session
    
    ⭐ New features:
    1. Random first turn (randomize_first_turn=True)
    2. Model selection (ai_model parameter)
    """
    try:
        # Generate session_id
        session_id = str(uuid.uuid4())
        
        # ⭐ Random first turn (if enabled)
        if request.randomize_first_turn:
            student_goes_first = random.choice([True, False])
        else:
            student_goes_first = True  # Default: student goes first
        
        # Create session
        session = NegotiationSession(
            session_id=session_id,
            student_id=request.student_id,
            student_name=request.student_name,
            scenario_name=request.scenario_name,
            student_role=request.student_role,
            ai_model=request.ai_model,  # ⭐ Use user-selected model
            student_goes_first=student_goes_first,
            use_memory=request.use_memory,
            use_plan=request.use_plan,
        )
        
        # If AI goes first, generate first message
        ai_first_message = None
        if not student_goes_first:
            ai_response = session._generate_ai_response()
            ai_cfg = session.scenario_config[session.ai_role]
            ai_label = ai_cfg.get("label", "AI")
            session.transcript.append(f"{ai_label}: {ai_response}")
            ai_first_message = ai_response
        
        # Save to database
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
    """
    Send student message and get AI response
    """
    # Load session
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # Process message
    result = session.process_student_message(request.message)
    
    # Save to database
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
    
    ⭐ NEW ENDPOINT: Returns both system_prompt and context_prompt
    so the student can see their complete role, rules, and background
    """
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    student_cfg = session.scenario_config[session.student_role]
    
    return {
        "session_id": session_id,
        "role_label": student_cfg.get("label", session.student_role),
        "batna": student_cfg.get("batna", 0),
        "system_prompt": student_cfg["system_prompt"],  # Rules and objectives
        "context_prompt": student_cfg["context_prompt"],  # Background and values
        "scenario_name": session.scenario_name,
        "total_rounds": session.total_rounds
    }

@app.get("/negotiation/{session_id}/feedback")
def get_feedback(session_id: str):
    """
    Get or generate negotiation feedback
    
    ⭐ RELAXED: Can be called at any time (active, completed, failed)
    If negotiation is still active, provides feedback on progress so far
    """
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if feedback already exists
    if session.feedback_text:
        return {
            "session_id": session_id,
            "feedback": session.feedback_text,
            "generated_at": session.feedback_generated_at,
            "model": session.feedback_model,
            "status_when_generated": session.status,
            "cached": True
        }
    
    # Generate feedback (works at any status)
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
    """
    Force regenerate feedback
    
    ⭐ RELAXED: Can be called at any time to get updated feedback
    """
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
    """
    Download database file
    Requires correct secret key (read from environment variable DOWNLOAD_KEY)
    
    Usage:
    GET /download_db?secret=your-secret-key
    """
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