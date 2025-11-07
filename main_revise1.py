from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx, sqlite3, json, yaml, re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import os
import uuid
from openai_wrapper import OpenAIWrapper

# === CONFIG ===
API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_PATH = os.getenv("DB_PATH", "/data/negotiations.db")
SCENARIOS_DIR = os.getenv("SCENARIOS_DIR", "/app/scenarios")  # 场景 YAML 文件目录

# === APP ===
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DB helpers ===
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def ensure_schema():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    c = conn.cursor()
    
    # 原有的 transcripts 表（保持向后兼容）
    c.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT,
            timestamp TEXT NOT NULL,
            transcript TEXT NOT NULL,
            feedback TEXT
        )
    """)
    
    # 新增：谈判会话表
    c.execute("""
        CREATE TABLE IF NOT EXISTS negotiation_sessions (
            session_id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            student_name TEXT,
            scenario_name TEXT NOT NULL,
            student_role TEXT NOT NULL,
            ai_model TEXT NOT NULL,
            use_memory INTEGER NOT NULL,
            use_plan INTEGER NOT NULL,
            current_round INTEGER DEFAULT 1,
            total_rounds INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            
            transcript TEXT,
            ai_memory TEXT,
            ai_plan TEXT,
            ai_last_plan TEXT,
            
            deal_reached INTEGER DEFAULT 0,
            deal_failed INTEGER DEFAULT 0,
            final_student_offer TEXT,
            final_ai_offer TEXT,
            deal_summary TEXT
        )
    """)
    
    c.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_student ON transcripts(student_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_student ON negotiation_sessions(student_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON negotiation_sessions(status)")
    
    conn.commit()
    conn.close()

ensure_schema()


# === Helper Functions  ===

def last_round_window(transcript: list, k_rounds: int = 2) -> str:
    """提取最近 k 轮的对话"""
    import re
    pat = re.compile(r"^.+? \(round (\d+)\):\n", re.S)
    
    items = []
    for i, t in enumerate(transcript):
        m = pat.match(t)
        if not m:
            continue
        rnd = int(m.group(1))
        items.append((rnd, i, t))
    
    if not items:
        return ""
    
    rounds_sorted = sorted({r for r, _, _ in items})
    keep_rounds = set(rounds_sorted[-k_rounds:])
    
    window = [t for (r, i, t) in sorted(items, key=lambda x: x[1]) if r in keep_rounds]
    return "\n\n".join(window)

def _update_memory(
    agent: OpenAIWrapper,
    memory: str,
    fact_values: str,
    round_id: int,
    total_rounds: int,
    agent_label: str,
    rules_objective: str,
    transcript_window: str,
    k_rounds: int = 2
) -> str:
    """更新谈判状态记忆（来自 runner.py）"""
    
    system_prompt = (
        "You are a state tracking module for an AI negotiator. "
        "Produce a CONCISE, ACTIONABLE negotiation state that survives limited transcript windows.\n\n"
        "REQUIRED SECTIONS (keep each section ≤2 lines):\n"
        "OFFERS: [Them: <explicit numeric/boolean terms>; Us: <explicit numeric/boolean terms>; use old→new if changed]\n"
        "PATTERNS: [Concession/firmness behaviors observed across rounds]\n"
        "PRIORITIES: [Their priorities/red lines/BATNA if explicit; mark uncertain with 'Hypothesis:']\n"
        "OPPORTUNITIES: [Cross-issue trades, value-creation levers, give↔get pairs]\n"
        "INSIGHTS: [Durable facts and round-to-round deductions you will carry forward]\n\n"
        "HARD RULES:\n"
        "1) PERSISTENCE — Keep long-lived facts even if they don't reappear in latest window\n"
        "2) OFFERS COVERAGE — Always include opponent's latest terms AND our standing terms\n"
        "3) FACT vs HYPOTHESIS — Only mark as FACT if explicitly stated\n"
        "4) CANONICALIZATION — Strip prose; compress to atomic items\n"
        "5) DEDUP & DELTA — Remove stale/duplicates; keep only latest state\n"
        "6) NO VERBATIM QUOTES — Summarize\n"
        "7) BINDING — Record coupled give↔get conditions\n"
        "8) SAFETY NET — Prefer carrying forward stable state over inventing facts\n"
    )
    
    prev_memory = memory or ""
    
    user_prompt = (
        f"=== Identity ===\n"
        f"Role: State tracker of {agent_label}\n"
        f"Round: {round_id}/{total_rounds}\n\n"
        f"=== Scenario Objective & Rules ===\n{rules_objective}\n\n"
        f"=== Scenario Facts & Value model ===\n{fact_values}\n\n"
        f"=== Previous State ===\n{prev_memory or '(empty)'}\n\n"
        f"=== Recent Transcript Window (last {k_rounds} rounds) ===\n{transcript_window}\n\n"
        f"=== Output State ===\n"
        "Update the negotiation state now, following HARD RULES."
    )
    
    response = agent.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    return (response["content"] or "").strip()

def _generate_plan(
    agent: OpenAIWrapper,
    memory: str,
    fact_values: str,
    round_id: int,
    total_rounds: int,
    agent_label: str,
    last_plan: str,
    rules_objective: str
) -> str:
    """生成本轮谈判策略（来自 runner.py）"""
    
    system_prompt = (
        "You are a strategic planning module for an AI negotiator. "
        "Generate a SMART and ASSERTIVE plan for THIS round based on current state and context.\n\n"
        "PLANNING RULES:\n"
        "- READ STATE — Use OFFERS, PATTERNS, PRIORITIES, OPPORTUNITIES, and INSIGHTS from the State exactly as the source of truth.\n"
        "- ROUND GOAL — Set a concrete, realistic target for THIS round.\n"
        "- TRADE-ACROSS-ISSUES — Do not concede on high-priority terms without gaining on other issues.\n"
        "- ADAPT & NON-REPETITION — If last round's plan didn't move the opponent, change tactics.\n"
        "- CLOSING OPTIONS — If opponent's terms satisfy core priorities and stay above BATNA, consider accepting.\n"
        "- ZOPA AWARENESS — Only invoke ZOPA framing if it improves your outcome.\n"
        "- BREVITY — Be crisp and operational: no fluff.\n\n"
        "OUTPUT SKELETON (≤10 lines, short bullets):\n"
        "- ROUND GOAL: <one clear target for this round>\n"
        "- KEY LEVERS: <2–3 trade levers>\n"
        "- TACTICS: <2–3 concrete moves>\n"
        "- OFFER SCAFFOLD: <one main package>\n"
        "- RISK & RESPONSES: <if-then counterplans>\n"
    )
    
    user_prompt = (
        f"=== Identity ===\n"
        f"Role: Strategy module of {agent_label}\n"
        f"Round: {round_id}/{total_rounds}\n\n"
        f"=== Scenario Rules & Objective ===\n{rules_objective}\n\n"
        f"=== Scenario Facts & Value model ===\n{fact_values}\n\n"
        f"=== Current State ===\n{memory or '(empty)'}\n\n"
        f"=== Previous Strategy ===\n{last_plan or 'N/A'}\n\n"
        f"=== Output Strategy ===\n"
    )
    
    response = agent.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    return response["content"].strip()

def _inject_memory_and_plan(
    prompt: str, 
    memory: str, 
    plan: str,
    enable_memory: bool, 
    enable_plan: bool
) -> str:
    """将 memory 和 plan 注入到主 prompt"""
    parts = []
    if enable_memory and memory and memory.strip():
        parts.append(f"--- NEGOTIATION STATE TRACKING ---\n{memory}")
    if enable_plan and plan and plan.strip():
        parts.append(f"--- STRATEGY FOR THIS ROUND ---\n{plan}")
    parts.append(f"--- MAIN PROMPT ---\n{prompt}")
    return "\n\n".join(parts).strip()

_TOKEN_RE = re.compile(
    r"""^\s*(?:```[a-zA-Z]*\s*)?[\$]+DEAL_REACHED[\$]+(?:\s*```)?(?:\s|$)""",
    re.IGNORECASE | re.VERBOSE
)

def _is_deal_token(msg: str) -> bool:
    """检测是否包含 $DEAL_REACHED$ token"""
    return bool(_TOKEN_RE.match(msg or ""))

_MISUNDERSTANDING_RE = re.compile(
    r"""^\s*\$+\s*DEAL[_\-\s]?MISUNDERSTANDING\s*\$+""",
    re.IGNORECASE | re.VERBOSE
)

def _is_misunderstanding_token(msg: str) -> bool:
    """检测是否包含 $DEAL_MISUNDERSTANDING$ token"""
    return bool(_MISUNDERSTANDING_RE.match(msg or ""))

# === Scenario Loading ===
def load_scenario(scenario_name: str) -> dict:
    """从 YAML 文件加载场景配置"""
    scenario_path = Path(SCENARIOS_DIR) / f"{scenario_name}.yaml"
    if not scenario_path.exists():
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found")
    
    with open(scenario_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def list_available_scenarios() -> list:
    """列出所有可用的场景 - 支持没有 name/description/num_rounds 字段的 YAML 文件"""
    scenarios_path = Path(SCENARIOS_DIR)
    
    print(f"Loading scenarios from {SCENARIOS_DIR}...")
    
    if not scenarios_path.exists():
        print(f"Warning: Scenarios directory not found: {SCENARIOS_DIR}")
        return []
    
    scenarios = []
    for yaml_file in scenarios_path.glob("*.yaml"):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Generate friendly name from filename if not provided
            # Example: "Top_talent.yaml" → "Top Talent"
            default_name = yaml_file.stem.replace('_', ' ').title()
            
            # Extract role labels with safe defaults
            side1_label = config.get("side1", {}).get("label", "Side 1")
            side2_label = config.get("side2", {}).get("label", "Side 2")
            
            # Generate description from side labels if not provided
            default_description = f"Negotiation between {side1_label} and {side2_label}"
            
            scenarios.append({
                "id": yaml_file.stem,
                "name": config.get("name", default_name),
                "description": config.get("description", default_description),
                "side1_label": side1_label,
                "side2_label": side2_label,
            })
            
            print(f"  - {yaml_file.name}")
            
        except Exception as e:
            print(f"Warning: Failed to load scenario {yaml_file}: {e}")
    
    print(f"Found {len(scenarios)} scenarios")
    
    return scenarios

# === Session Management ===
class NegotiationSession:
    """管理单个人机谈判会话"""
    
    def __init__(
        self,
        session_id: str,
        scenario_config: dict,
        scenario_name: str,
        student_role: str,
        student_id: str,
        student_name: str,
        ai_model: str,
        use_memory: bool,
        use_plan: bool,
        total_rounds: int
    ):
        self.session_id = session_id
        self.scenario_config = scenario_config
        self.scenario_name = scenario_name
        self.student_role = student_role  # "side1" or "side2"
        self.ai_role = "side2" if student_role == "side1" else "side1"
        self.student_id = student_id
        self.student_name = student_name
        self.ai_model = ai_model
        self.use_memory = use_memory
        self.use_plan = use_plan
        self.total_rounds = total_rounds
        
        # 初始化 AI agents
        ai_cfg = scenario_config[self.ai_role]
        self.ai_agent = OpenAIWrapper(model=ai_model, label=ai_cfg["label"])
        
        if use_memory:
            self.ai_memory_agent = OpenAIWrapper(
                model=ai_model,
                label=f"Memory-{ai_cfg['label']}"
            )
        
        if use_plan:
            self.ai_plan_agent = OpenAIWrapper(
                model=ai_model,
                label=f"Planner-{ai_cfg['label']}"
            )
        
        # 状态
        self.current_round = 1
        self.transcript = []
        self.ai_memory = ""
        self.ai_plan = ""
        self.ai_last_plan = ""
        self.deal_reached = False
        self.deal_failed = False
        self.status = "active"
        
        # Deal tracking
        self.deal_initiator = None  # "student" or "ai"
        self.student_deal_json = None
        self.ai_deal_json = None
    
    def save_to_db(self):
        """保存会话状态到数据库"""
        conn = get_conn()
        c = conn.cursor()
        
        c.execute("""
            INSERT OR REPLACE INTO negotiation_sessions 
            (session_id, student_id, student_name, scenario_name, student_role, 
             ai_model, use_memory, use_plan, current_round, total_rounds,
             created_at, updated_at, status, transcript, ai_memory, ai_plan, 
             ai_last_plan, deal_reached, deal_failed, final_student_offer, 
             final_ai_offer, deal_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.session_id, self.student_id, self.student_name, self.scenario_name,
            self.student_role, self.ai_model, int(self.use_memory), int(self.use_plan),
            self.current_round, self.total_rounds,
            datetime.utcnow().isoformat(), datetime.utcnow().isoformat(),
            self.status, json.dumps(self.transcript), self.ai_memory, self.ai_plan,
            self.ai_last_plan, int(self.deal_reached), int(self.deal_failed),
            json.dumps(self.student_deal_json) if self.student_deal_json else None,
            json.dumps(self.ai_deal_json) if self.ai_deal_json else None,
            None  # deal_summary 将在最后填充
        ))
        
        conn.commit()
        conn.close()
    
    @classmethod
    def load_from_db(cls, session_id: str) -> 'NegotiationSession':
        """从数据库加载会话"""
        conn = get_conn()
        c = conn.cursor()
        
        c.execute("SELECT * FROM negotiation_sessions WHERE session_id = ?", (session_id,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 解析行数据
        cols = [desc[0] for desc in c.description]
        data = dict(zip(cols, row))
        
        # 加载场景配置
        scenario_config = load_scenario(data["scenario_name"])
        
        # 创建会话对象
        session = cls(
            session_id=data["session_id"],
            scenario_config=scenario_config,
            scenario_name=data["scenario_name"],
            student_role=data["student_role"],
            student_id=data["student_id"],
            student_name=data["student_name"],
            ai_model=data["ai_model"],
            use_memory=bool(data["use_memory"]),
            use_plan=bool(data["use_plan"]),
            total_rounds=data["total_rounds"]
        )
        
        # 恢复状态
        session.current_round = data["current_round"]
        session.transcript = json.loads(data["transcript"]) if data["transcript"] else []
        session.ai_memory = data["ai_memory"] or ""
        session.ai_plan = data["ai_plan"] or ""
        session.ai_last_plan = data["ai_last_plan"] or ""
        session.deal_reached = bool(data["deal_reached"])
        session.deal_failed = bool(data["deal_failed"])
        session.status = data["status"]
        
        if data["final_student_offer"]:
            session.student_deal_json = json.loads(data["final_student_offer"])
        if data["final_ai_offer"]:
            session.ai_deal_json = json.loads(data["final_ai_offer"])
        
        return session
    
    def get_initial_prompt_for_student(self) -> str:
        """生成学生的初始提示"""
        student_cfg = self.scenario_config[self.student_role]
        
        # 构建初始提示
        prompt = f"{student_cfg['context_prompt']}\n\n{student_cfg['initial_offer_prompt']}"
        
        return prompt
    
    def process_student_message(self, student_msg: str) -> dict:
        """
        处理学生消息并生成 AI 回复
        
        返回格式:
        {
            "ai_response": str,
            "current_round": int,
            "rounds_remaining": int,
            "deal_reached": bool,
            "needs_confirmation": bool,  # 学生喊了 deal，需要 AI 确认
            "session_status": str
        }
        """
        student_cfg = self.scenario_config[self.student_role]
        ai_cfg = self.scenario_config[self.ai_role]
        
        # 1. 记录学生消息到 transcript
        self.transcript.append(f"{student_cfg['label']} (round {self.current_round}):\n{student_msg}")
        
        # 2. 检测学生是否喊了 $DEAL_REACHED$
        if _is_deal_token(student_msg):
            return self._handle_student_deal_initiation(student_msg)
        
        # 3. 更新 AI 的 memory（如果启用且不是第一轮）
        if self.use_memory and len(self.transcript) > 0:
            recent_window = last_round_window(self.transcript, k_rounds=2)
            self.ai_memory = _update_memory(
                agent=self.ai_memory_agent,
                memory=self.ai_memory,
                fact_values=ai_cfg["context_prompt"],
                round_id=self.current_round,
                total_rounds=self.total_rounds,
                agent_label=ai_cfg["label"],
                rules_objective=ai_cfg["system_prompt"],
                transcript_window=recent_window,
                k_rounds=2
            )
        
        # 4. 更新 AI 的 plan（如果启用且不是第一轮）
        if self.use_plan and len(self.transcript) > 0:
            self.ai_plan = _generate_plan(
                agent=self.ai_plan_agent,
                memory=self.ai_memory,
                fact_values=ai_cfg["context_prompt"],
                round_id=self.current_round,
                total_rounds=self.total_rounds,
                agent_label=ai_cfg["label"],
                last_plan=self.ai_last_plan,
                rules_objective=ai_cfg["system_prompt"]
            )
            self.ai_last_plan = self.ai_plan
        
        # 5. 生成 AI 回复
        ai_response = self._generate_ai_response()
        
        # 6. 记录 AI 回复
        self.transcript.append(f"{ai_cfg['label']} (round {self.current_round}):\n{ai_response}")
        
        # 7. 检测 AI 是否喊了 $DEAL_REACHED$
        if _is_deal_token(ai_response):
            return self._handle_ai_deal_initiation(ai_response)
        
        # 8. 推进到下一轮
        self.current_round += 1
        
        # 9. 检查是否超时
        if self.current_round > self.total_rounds:
            self.status = "completed"
            self.deal_reached = False
            self.save_to_db()
            
            return {
                "ai_response": ai_response,
                "current_round": self.current_round - 1,
                "rounds_remaining": 0,
                "deal_reached": False,
                "timed_out": True,
                "session_status": "completed"
            }
        
        # 10. 保存状态
        self.save_to_db()
        
        return {
            "ai_response": ai_response,
            "current_round": self.current_round,
            "rounds_remaining": self.total_rounds - self.current_round + 1,
            "deal_reached": False,
            "session_status": "active"
        }
    
    def _generate_ai_response(self) -> str:
        """生成 AI 的谈判回复"""
        ai_cfg = self.scenario_config[self.ai_role]
        
        # 构建完整历史
        history = "\n\n".join(self.transcript)
        
        # 构建 continuation prompt
        json_schema_raw = self.scenario_config["json_schema"]
        json_schema_dict = json.loads(json_schema_raw)
        json_schema_text = "\n" + json.dumps(json_schema_dict, indent=2)
        
        # 转义大括号
        def _escape_braces(s: str) -> str:
            return s.replace("{", "{{").replace("}", "}}")
        
        def _unescape_braces(s: str) -> str:
            return s.replace("{{", "{").replace("}}", "}")
        
        # Universal continuation prompt
        universal_continuation_prompt = (
            "CURRENT ROUND INFORMATION:\n"
            f"It is now round {self.current_round}/{self.total_rounds}. "
            f"You have {self.total_rounds - self.current_round} rounds remaining after this one.\n\n"
            "<BEGIN COMPLETE NEGOTIATION TRANSCRIPT>\n{history}\n\n"
            "<END NEGOTIATION TRANSCRIPT>\n\n"
            "OUTPUT INSTRUCTIONS:\n"
            "1. You can either continue the negotiation or accept the most recent terms by outputting '$DEAL_REACHED$' at the beginning.\n\n"
            "2. OUTPUT OPTIONS:\n"
            "a) Continue negotiating\n"
            "b) Accept terms by outputting '$DEAL_REACHED$' on its own line, then the JSON deal terms.\n"
            f"JSON FORMAT:\n{_escape_braces(json_schema_text)}"
        )
        
        prompt = universal_continuation_prompt.format(history=_escape_braces(history))
        prompt = _unescape_braces(prompt)
        
        # 注入 memory & plan
        full_prompt = _inject_memory_and_plan(
            prompt, 
            self.ai_memory, 
            self.ai_plan,
            self.use_memory, 
            self.use_plan
        )
        
        # 调用 AI
        response = self.ai_agent.chat([
            {"role": "system", "content": ai_cfg["system_prompt"]},
            {"role": "user", "content": full_prompt}
        ])
        
        return response["content"]
    
    def _handle_student_deal_initiation(self, student_msg: str) -> dict:
        """处理学生发起的 deal"""
        # 提取 JSON
        brace_idx = student_msg.find("{")
        if brace_idx != -1:
            json_part = student_msg[brace_idx:]
            try:
                self.student_deal_json = json.loads(json_part)
            except json.JSONDecodeError:
                # JSON 解析失败
                return {
                    "error": "Invalid JSON in deal proposal",
                    "ai_response": "I noticed you tried to propose a deal, but the JSON format was invalid. Please try again.",
                    "session_status": "active"
                }
        else:
            return {
                "error": "No JSON found after $DEAL_REACHED$",
                "ai_response": "I noticed you signaled a deal, but no JSON terms were provided. Please include the deal terms in JSON format.",
                "session_status": "active"
            }
        
        self.deal_initiator = "student"
        self.deal_reached = True
        
        # 现在需要 AI 确认
        ai_confirmation_response = self._request_ai_confirmation()
        
        # 检测 AI 是否说 MISUNDERSTANDING
        if _is_misunderstanding_token(ai_confirmation_response):
            self.status = "completed"
            self.deal_reached = False
            self.save_to_db()
            
            return {
                "ai_response": ai_confirmation_response,
                "deal_reached": False,
                "misunderstanding": True,
                "session_status": "completed",
                "message": "The AI detected a misunderstanding in the deal terms."
            }
        
        # AI 确认了，提取 AI 的 JSON
        brace_idx = ai_confirmation_response.find("{")
        if brace_idx != -1:
            json_part = ai_confirmation_response[brace_idx:]
            try:
                self.ai_deal_json = json.loads(json_part)
            except json.JSONDecodeError:
                self.ai_deal_json = {}
        
        # 检查一致性
        verified = self._verify_deal_consistency()
        
        self.status = "completed"
        self.save_to_db()
        
        return {
            "ai_response": ai_confirmation_response,
            "deal_reached": True,
            "verified_agreement": verified,
            "student_offer": self.student_deal_json,
            "ai_offer": self.ai_deal_json,
            "session_status": "completed"
        }
    
    def _handle_ai_deal_initiation(self, ai_response: str) -> dict:
        """处理 AI 发起的 deal"""
        # 提取 JSON
        brace_idx = ai_response.find("{")
        if brace_idx != -1:
            json_part = ai_response[brace_idx:]
            try:
                self.ai_deal_json = json.loads(json_part)
            except json.JSONDecodeError:
                self.ai_deal_json = {}
        
        self.deal_initiator = "ai"
        self.deal_reached = True
        
        # 告诉前端：AI 发起了 deal，需要学生确认
        return {
            "ai_response": ai_response,
            "deal_reached": True,
            "needs_student_confirmation": True,
            "ai_offer": self.ai_deal_json,
            "session_status": "awaiting_confirmation",
            "message": "The AI has proposed a deal. Please review and confirm or reject."
        }
    
    def handle_student_confirmation(self, student_confirmation: dict) -> dict:
        """
        处理学生对 AI 提出的 deal 的确认
        
        student_confirmation: {
            "confirmed": bool,
            "deal_terms": dict (如果 confirmed=True)
        }
        """
        if not student_confirmation.get("confirmed"):
            # 学生拒绝
            self.deal_reached = False
            self.status = "completed"
            self.save_to_db()
            
            return {
                "deal_reached": False,
                "misunderstanding": True,
                "session_status": "completed",
                "message": "You rejected the AI's proposed deal terms."
            }
        
        # 学生确认
        self.student_deal_json = student_confirmation.get("deal_terms", {})
        
        # 检查一致性
        verified = self._verify_deal_consistency()
        
        self.status = "completed"
        self.save_to_db()
        
        return {
            "deal_reached": True,
            "verified_agreement": verified,
            "student_offer": self.student_deal_json,
            "ai_offer": self.ai_deal_json,
            "session_status": "completed"
        }
    
    def _request_ai_confirmation(self) -> str:
        """请求 AI 确认学生提出的 deal"""
        ai_cfg = self.scenario_config[self.ai_role]
        history = "\n\n".join(self.transcript)
        
        json_schema_raw = self.scenario_config["json_schema"]
        json_schema_dict = json.loads(json_schema_raw)
        json_schema_text = json.dumps(json_schema_dict, indent=2)
        
        conf_prompt = (
            "DEAL CONFIRMATION:\n"
            "The other side has proposed a deal and specified the terms in their last message. "
            "Review the complete transcript to confirm the terms are correct.\n\n"
            "COMPLETE NEGOTIATION TRANSCRIPT:\n"
            f"<BEGIN TRANSCRIPT>\n{history}\n<END TRANSCRIPT>\n\n"
            "OUTPUT INSTRUCTIONS:\n"
            "OPTION 1: If the terms are CORRECT, output the agreed deal in JSON format.\n"
            f"JSON FORMAT:\n{json_schema_text}\n\n"
            "OPTION 2: If the terms are NOT correct, output ONLY '$DEAL_MISUNDERSTANDING$'.\n"
        )
        
        response = self.ai_agent.chat([
            {"role": "system", "content": ai_cfg["system_prompt"]},
            {"role": "user", "content": conf_prompt}
        ])
        
        return response["content"]
    
    def _verify_deal_consistency(self) -> bool:
        """验证双方 JSON 的一致性（忽略 value 字段）"""
        if not self.student_deal_json or not self.ai_deal_json:
            return False
        
        # 忽略的字段（各自的 value 计算）
        IGNORE = {
            "total_value_of_deal_to_me",
            "total_points_of_deal_to_me",
            "expected_value_of_deal_to_me_in_millions",
        }
        
        student_normalized = {k: v for k, v in self.student_deal_json.items() if k not in IGNORE}
        ai_normalized = {k: v for k, v in self.ai_deal_json.items() if k not in IGNORE}
        
        return student_normalized == ai_normalized

# === API Routes ===

@app.get("/")
def root():
    return {"status": "ok", "message": "Negotiation backend is running!"}

@app.get("/scenarios")
def get_scenarios():
    """获取所有可用的谈判场景"""
    scenarios = list_available_scenarios()
    return {"scenarios": scenarios}

@app.get("/scenarios/{scenario_name}")
def get_scenario_details(scenario_name: str):
    """获取特定场景的详细信息（不含敏感信息） - 支持没有 name/description/num_rounds 的 YAML"""
    try:
        config = load_scenario(scenario_name)
    except HTTPException:
        raise
    
    # Generate friendly defaults if fields are missing
    default_name = scenario_name.replace('_', ' ').title()
    side1_label = config.get("side1", {}).get("label", "Side 1")
    side2_label = config.get("side2", {}).get("label", "Side 2")
    default_description = f"Negotiation between {side1_label} and {side2_label}"
    
    return {
        "scenario_name": scenario_name,
        "name": config.get("name", default_name),
        "description": config.get("description", default_description),
        "total_rounds": config.get("num_rounds", 10),
        "side1": {
            "label": side1_label,
            "batna": config.get("side1", {}).get("batna"),
        },
        "side2": {
            "label": side2_label,
            "batna": config.get("side2", {}).get("batna"),
        }
    }

@app.post("/negotiation/start")
async def start_negotiation(request: Request):
    """
    开始一个新的谈判会话
    
    请求体:
    {
        "student_id": str,
        "student_name": str,
        "scenario_name": str,
        "student_role": "side1" | "side2",
        "ai_model": str (optional, default: "openai/gpt-4o-mini"),
        "use_memory": bool (optional, default: true),
        "use_plan": bool (optional, default: true),
        "total_rounds": int (optional, uses scenario default or 10)
    }
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    
    data = await request.json()
    
    student_id = data.get("student_id")
    student_name = data.get("student_name")
    scenario_name = data.get("scenario_name")
    student_role = data.get("student_role")
    
    if not all([student_id, student_name, scenario_name, student_role]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    if student_role not in ["side1", "side2"]:
        raise HTTPException(status_code=400, detail="student_role must be 'side1' or 'side2'")
    
    # 加载场景
    scenario_config = load_scenario(scenario_name)
    
    # 可选参数 - 支持没有 num_rounds 的 YAML，默认 10 轮
    ai_model = data.get("ai_model", "openai/gpt-4o-mini")
    use_memory = data.get("use_memory", True)
    use_plan = data.get("use_plan", True)
    total_rounds = data.get("total_rounds", scenario_config.get("num_rounds", 10))
    
    # 创建会话
    session_id = str(uuid.uuid4())
    session = NegotiationSession(
        session_id=session_id,
        scenario_config=scenario_config,
        scenario_name=scenario_name,
        student_role=student_role,
        student_id=student_id,
        student_name=student_name,
        ai_model=ai_model,
        use_memory=use_memory,
        use_plan=use_plan,
        total_rounds=total_rounds
    )
    
    # 保存到数据库
    session.save_to_db()
    
    # 返回初始信息
    student_cfg = scenario_config[student_role]
    ai_cfg = scenario_config[session.ai_role]
    
    return {
        "session_id": session_id,
        "scenario_name": scenario_name,
        "your_role": student_cfg["label"],
        "ai_role": ai_cfg["label"],
        "total_rounds": total_rounds,
        "initial_prompt": session.get_initial_prompt_for_student(),
        "settings": {
            "ai_model": ai_model,
            "use_memory": use_memory,
            "use_plan": use_plan
        }
    }

@app.post("/negotiation/{session_id}/message")
async def negotiation_message(session_id: str, request: Request):
    """
    在谈判会话中发送消息
    
    请求体:
    {
        "message": str
    }
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    
    data = await request.json()
    message = data.get("message")
    
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' field")
    
    # 加载会话
    session = NegotiationSession.load_from_db(session_id)
    
    if session.status != "active":
        raise HTTPException(status_code=400, detail=f"Session is {session.status}, cannot send messages")
    
    # 处理消息
    result = session.process_student_message(message)
    
    return result

@app.post("/negotiation/{session_id}/confirm")
async def confirm_deal(session_id: str, request: Request):
    """
    确认 AI 提出的 deal
    
    请求体:
    {
        "confirmed": bool,
        "deal_terms": dict (if confirmed=true)
    }
    """
    data = await request.json()
    
    # 加载会话
    session = NegotiationSession.load_from_db(session_id)
    
    if session.status != "awaiting_confirmation":
        raise HTTPException(status_code=400, detail="Session is not awaiting confirmation")
    
    # 处理确认
    result = session.handle_student_confirmation(data)
    
    return result

@app.get("/negotiation/{session_id}/status")
def get_negotiation_status(session_id: str):
    """获取谈判会话状态"""
    session = NegotiationSession.load_from_db(session_id)
    
    return {
        "session_id": session.session_id,
        "status": session.status,
        "current_round": session.current_round,
        "total_rounds": session.total_rounds,
        "deal_reached": session.deal_reached,
        "transcript_length": len(session.transcript)
    }

@app.post("/negotiation/{session_id}/feedback")
async def get_negotiation_feedback(session_id: str):
    """
    谈判结束后获取反馈
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    
    session = NegotiationSession.load_from_db(session_id)
    
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Session is not completed yet")
    
    # 生成反馈
    transcript_text = "\n\n".join(session.transcript)
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a negotiation coach. Give constructive, specific, and actionable feedback on the student's negotiation performance. Focus on their strategy, communication, value creation, and deal-making skills."
            },
            {
                "role": "user",
                "content": f"Here is a negotiation transcript. The student was playing the role of {session.scenario_config[session.student_role]['label']}. Provide detailed feedback:\n\n{transcript_text}"
            }
        ]
    }
    
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
    
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e.response.text[:300]}")
    
    resp_json = r.json()
    try:
        feedback_text = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail=f"Unexpected LLM response: {resp_json}")
    
    return {
        "feedback": feedback_text,
        "deal_reached": session.deal_reached,
        "verified_agreement": session._verify_deal_consistency() if session.deal_reached else False
    }

# Optional: download the raw SQLite file
@app.get("/download_db")
def download_db(secret: str | None = None):
    allowed = os.getenv("DOWNLOAD_KEY")
    if allowed and secret != allowed:
        raise HTTPException(status_code=403, detail="unauthorized")
    return FileResponse(DB_PATH, filename="negotiations.db")

# Health check
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}