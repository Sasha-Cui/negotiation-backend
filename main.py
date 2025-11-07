"""
优化后的 Negotiation Backend - 与 runner.py 完全一致的逻辑

关键修复:
1. ✅ 每次调用都包含 context_prompt
2. ✅ Round 1 先手使用 initial_offer_prompt
3. ✅ Universal continuation prompt 文本完全一致
4. ✅ Memory & Plan 逻辑完全一致
5. ✅ Deal 确认流程完全一致
6. ✅ 随机先手功能
7. ✅ 模型选择功能
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
# 导入 OpenAIWrapper
# ============================================================================
from openai_wrapper import OpenAIWrapper

# ============================================================================
# 配置
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
# 数据库初始化
# ============================================================================
def init_db():
    """初始化数据库"""
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
            updated_at TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

# 启动时初始化
init_db()

# ============================================================================
# 辅助函数（与 runner.py 完全一致）
# ============================================================================

def _escape_braces(s: str) -> str:
    """转义大括号用于 .format()"""
    return s.replace("{", "{{").replace("}", "}}")

def _unescape_braces(s: str) -> str:
    """反转义大括号"""
    return s.replace("{{", "{").replace("}}", "}")

def last_round_window(transcript: List[str], k_rounds: int = 2) -> str:
    """
    提取最近 k 轮的对话
    与 runner.py 完全一致
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
    从文本中提取 JSON
    与 runner.py 完全一致
    """
    # 尝试查找 JSON block
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # 尝试直接解析
    try:
        # 查找 { ... }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
    except:
        pass
    
    return None

# ============================================================================
# NegotiationSession 类
# ============================================================================
class NegotiationSession:
    """
    谈判会话类
    与 runner.py 逻辑完全一致
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
        
        # 加载场景配置
        scenario_path = Path(SCENARIOS_DIR) / f"{scenario_name}.yaml"
        with open(scenario_path, 'r', encoding='utf-8') as f:
            self.scenario_config = yaml.safe_load(f)
        
        # 确定角色
        if student_role == "side1":
            self.ai_role = "side2"
        else:
            self.ai_role = "side1"
        
        # 初始化 AI agent
        self.ai_agent = OpenAIWrapper(model=ai_model, label="AI")
        
        # Memory & Plan agents（如果启用）
        if use_memory:
            self.memory_agent = OpenAIWrapper(model=ai_model, label="Memory")
        if use_plan:
            self.plan_agent = OpenAIWrapper(model=ai_model, label="Plan")
        
        # 会话状态
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
        
        # 时间戳
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
    
    # ========================================================================
    # 核心方法 1: 生成 AI 回复（与 runner.py 完全一致）
    # ========================================================================
    def _generate_ai_response(self) -> str:
        """
        生成 AI 的谈判回复
        
        ⭐ 关键修复:
        1. 每次都包含 context_prompt
        2. 区分 Round 1 先手（用 initial_offer_prompt）
        3. Prompt 文本与 runner.py 完全一致
        """
        ai_cfg = self.scenario_config[self.ai_role]
        
        # 1. 获取 context_prompt（⭐ 关键：每次都需要）
        context = ai_cfg["context_prompt"]
        
        # 2. 构建完整历史
        history = "\n\n".join(self.transcript)
        
        # 3. 判断是否是 Round 1 且 AI 先手
        is_round_1 = (self.current_round == 1)
        student_went_first = (len(self.transcript) > 0)
        ai_goes_first = is_round_1 and not student_went_first
        
        # 4. 根据情况选择 prompt
        if ai_goes_first:
            # ============================================================
            # Round 1 先手：使用 initial_offer_prompt
            # ============================================================
            base_prompt = ai_cfg["initial_offer_prompt"]
            user_prompt = f"{context}\n\n{base_prompt}"
        
        else:
            # ============================================================
            # Round 1 后手 或 Round 2+：使用 universal_continuation_prompt
            # ============================================================
            
            # 构建 JSON schema
            json_schema_raw = self.scenario_config["json_schema"]
            json_schema_dict = json.loads(json_schema_raw)
            json_schema_text = "\n" + json.dumps(json_schema_dict, indent=2)
            
            # ⭐ Universal continuation prompt（与 runner.py 完全一致）
            universal_continuation_prompt = (
                "CURRENT ROUND INFORMATION:\n"
                f"It is now round {self.current_round}/{self.total_rounds}. "
                f"You have {self.total_rounds - self.current_round} rounds remaining after this one.\n\n"
                "<BEGIN COMPLETE NEGOTIATION TRANSCRIPT>\n{{history}}\n\n"
                "<END NEGOTIATION TRANSCRIPT>\n\n"
                "OUTPUT INSTRUCTIONS:\n"
                "1. You can either continue the negotiation or propose a final deal by outputting '$DEAL_REACHED$' at the beginning of your message.\n\n"
                "2. OUTPUT OPTIONS:\n"
                "   a) Continue negotiating: Respond naturally to continue the discussion, making counteroffers or asking questions.\n"
                "   b) Propose final deal: Output '$DEAL_REACHED$' on its own line, then output a valid JSON object with the deal terms.\n\n"
                f"JSON FORMAT (if proposing deal):\n{_escape_braces(json_schema_text)}\n\n"
                "3. IMPORTANT: If you believe no mutually beneficial deal is possible, you may output '$DEAL_FAILED$' instead of continuing.\n\n"
                "4. Remember: You must maximize your own value while reaching agreement."
            )
            
            # 填充 history
            continuation = universal_continuation_prompt.format(history=_escape_braces(history))
            continuation = _unescape_braces(continuation)
            
            # ⭐ 关键：context + continuation
            user_prompt = f"{context}\n\n{continuation}"
        
        # 5. 注入 Memory & Plan（与 runner.py 完全一致）
        full_prompt = self._inject_memory_and_plan(user_prompt)
        
        # 6. 调用 AI
        messages = [
            {"role": "system", "content": ai_cfg["system_prompt"]},
            {"role": "user", "content": full_prompt}
        ]
        
        response = self.ai_agent.chat(messages)
        return response["content"]
    
    # ========================================================================
    # 核心方法 2: 注入 Memory & Plan（与 runner.py 完全一致）
    # ========================================================================
    def _inject_memory_and_plan(self, base_prompt: str) -> str:
        """
        将 Memory 和 Plan 包装到 prompt 外层
        与 runner.py 完全一致
        """
        parts = []
        
        if self.use_memory and self.ai_memory:
            parts.append(f"--- NEGOTIATION STATE TRACKING ---\n{self.ai_memory}\n")
        
        if self.use_plan and self.ai_plan:
            parts.append(f"--- STRATEGY FOR THIS ROUND ---\n{self.ai_plan}\n")
        
        parts.append(f"--- MAIN PROMPT ---\n{base_prompt}")
        
        return "\n".join(parts)
    
    # ========================================================================
    # 核心方法 3: 更新 Memory（与 runner.py 完全一致）
    # ========================================================================
    def _update_memory(self):
        """
        更新 AI 的 Memory
        与 runner.py 完全一致
        """
        if not self.use_memory:
            return
        
        # 提取最近 2 轮对话
        recent_transcript = last_round_window(self.transcript, k_rounds=2)
        
        ai_cfg = self.scenario_config[self.ai_role]
        student_cfg = self.scenario_config[self.student_role]
        
        ai_label = ai_cfg.get("label", self.ai_role)
        student_label = student_cfg.get("label", self.student_role)
        
        # ⭐ Memory system prompt（与 runner.py 完全一致）
        memory_system = (
            "You are a memory assistant for a negotiation agent. "
            "Your job is to track key information about the negotiation state, "
            "including the opponent's bargaining patterns, priorities, and opportunities for value creation."
        )
        
        # ⭐ Memory user prompt（与 runner.py 完全一致）
        memory_user = (
            f"--- PREVIOUS MEMORY ---\n{self.ai_memory}\n\n" if self.ai_memory else ""
            f"--- RECENT TRANSCRIPT (last 2 rounds) ---\n{recent_transcript}\n\n"
            "Based on the recent transcript, update the memory with any new insights about the negotiation state. "
            "Format your response as:\n\n"
            f"PATTERNS: [What patterns do you observe in {student_label}'s bargaining behavior?]\n"
            f"PRIORITIES: [What seems most important to {student_label}?]\n"
            f"OPPORTUNITIES: [What opportunities for value creation do you see?]\n"
            "INSIGHTS: [Any other important observations?]"
        )
        
        messages = [
            {"role": "system", "content": memory_system},
            {"role": "user", "content": memory_user}
        ]
        
        response = self.memory_agent.chat(messages)
        self.ai_memory = response["content"]
    
    # ========================================================================
    # 核心方法 4: 生成 Plan（与 runner.py 完全一致）
    # ========================================================================
    def _generate_plan(self):
        """
        生成本轮的 Plan
        与 runner.py 完全一致
        """
        if not self.use_plan:
            return
        
        ai_cfg = self.scenario_config[self.ai_role]
        context = ai_cfg["context_prompt"]
        
        # 提取最近 2 轮对话
        recent_transcript = last_round_window(self.transcript, k_rounds=2)
        
        ai_label = ai_cfg.get("label", self.ai_role)
        
        # ⭐ Plan system prompt（与 runner.py 完全一致）
        plan_system = (
            "You are a strategic planning assistant for a negotiation agent. "
            f"Your job is to generate a concrete, actionable plan for {ai_label} for the current round of negotiation."
        )
        
        # ⭐ Plan user prompt（与 runner.py 完全一致）
        plan_user = (
            f"--- CONTEXT ---\n{context}\n\n"
            f"--- MEMORY (if available) ---\n{self.ai_memory}\n\n" if self.use_memory and self.ai_memory else ""
            f"--- PREVIOUS PLAN (if available) ---\n{self.ai_plan}\n\n" if self.ai_plan else ""
            f"--- CURRENT SITUATION ---\n"
            f"Round {self.current_round}/{self.total_rounds}\n\n"
            f"Recent transcript:\n{recent_transcript}\n\n"
            f"Generate a strategic plan for {ai_label} for this round. Format:\n\n"
            "ROUND GOAL: [What specific objective should be achieved this round?]\n"
            "KEY LEVERS: [What aspects of the deal should be emphasized?]\n"
            "TACTICS: [What specific negotiation tactics should be used?]\n"
            "OFFER SCAFFOLD: [What rough structure should the offer have?]\n"
            "RISK & RESPONSES: [What are potential issues and how to address them?]"
        )
        
        messages = [
            {"role": "system", "content": plan_system},
            {"role": "user", "content": plan_user}
        ]
        
        response = self.plan_agent.chat(messages)
        self.ai_plan = response["content"]
    
    # ========================================================================
    # 核心方法 5: 处理 Student 消息（修复 deal 确认逻辑）
    # ========================================================================
    def process_student_message(self, message: str) -> Dict[str, Any]:
        """
        处理学生消息
        
        ⭐ 关键修复:
        1. 学生可以输入 $DEAL_REACHED$ + JSON（就像 AI 一样）
        2. AI 需要确认是否和自己的 offer 一致
        3. 不需要前端的 accept/reject dialog
        """
        # 1. 保存学生消息到 transcript
        student_cfg = self.scenario_config[self.student_role]
        student_label = student_cfg.get("label", "Student")
        self.transcript.append(f"{student_label}: {message}")
        
        # 2. 检查学生是否提出 deal
        if "$DEAL_REACHED$" in message:
            student_json = extract_json_from_text(message)
            
            if student_json:
                self.student_deal_json = student_json
                
                # ⭐ 关键：让 AI 确认
                ai_confirm_response = self._request_ai_deal_confirmation(student_json)
                
                ai_cfg = self.scenario_config[self.ai_role]
                ai_label = ai_cfg.get("label", "AI")
                self.transcript.append(f"{ai_label}: {ai_confirm_response}")
                
                # 检查 AI 的确认结果
                if "$DEAL_REACHED$" in ai_confirm_response:
                    # AI 确认一致
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
                    # AI 表示不一致
                    return {
                        "ai_message": ai_confirm_response,
                        "deal_reached": False,
                        "misunderstanding": True,
                        "round": self.current_round
                    }
        
        # 3. 检查是否提出 deal failed
        if "$DEAL_FAILED$" in message:
            self.deal_failed = True
            self.status = "failed"
            return {
                "ai_message": "You have indicated that no deal is possible. Negotiation ended.",
                "deal_failed": True,
                "round": self.current_round
            }
        
        # 4. 更新 Memory & Plan（如果启用）
        if self.current_round > 1 or len(self.transcript) > 1:
            self._update_memory()
            self._generate_plan()
        
        # 5. 生成 AI 回复
        ai_response = self._generate_ai_response()
        
        ai_cfg = self.scenario_config[self.ai_role]
        ai_label = ai_cfg.get("label", "AI")
        self.transcript.append(f"{ai_label}: {ai_response}")
        
        # 6. 检查 AI 是否提出 deal
        if "$DEAL_REACHED$" in ai_response:
            ai_json = extract_json_from_text(ai_response)
            if ai_json:
                self.ai_deal_json = ai_json
                # ⭐ 不自动确认！等学生回复
                return {
                    "ai_message": ai_response,
                    "ai_proposed_deal": True,
                    "ai_deal_terms": ai_json,
                    "round": self.current_round
                }
        
        # 7. 检查 AI 是否提出 deal failed
        if "$DEAL_FAILED$" in ai_response:
            self.deal_failed = True
            self.status = "failed"
            return {
                "ai_message": ai_response,
                "deal_failed": True,
                "round": self.current_round
            }
        
        # 8. 检查是否到达最后一轮
        if self.current_round >= self.total_rounds:
            self.status = "completed"
            return {
                "ai_message": ai_response,
                "negotiation_ended": True,
                "reason": "max_rounds_reached",
                "round": self.current_round
            }
        
        # 9. 进入下一轮
        self.current_round += 1
        
        return {
            "ai_message": ai_response,
            "deal_reached": False,
            "round": self.current_round - 1  # 刚完成的轮次
        }
    
    # ========================================================================
    # 核心方法 6: AI 确认 Deal（与 runner.py 完全一致）
    # ========================================================================
    def _request_ai_deal_confirmation(self, student_deal_json: Dict) -> str:
        """
        请求 AI 确认学生提出的 deal
        与 runner.py 完全一致
        """
        ai_cfg = self.scenario_config[self.ai_role]
        context = ai_cfg["context_prompt"]
        
        # ⭐ Confirmation prompt（与 runner.py 完全一致）
        confirm_prompt = (
            f"The other party has proposed the following deal:\n\n"
            f"{json.dumps(student_deal_json, indent=2)}\n\n"
            "Please review this proposal carefully. "
            "If this matches your most recent offer exactly (ignoring any value calculations), "
            "output '$DEAL_REACHED$' on its own line followed by the same JSON. "
            "If it differs from your most recent offer in any way, "
            "output '$DEAL_MISUNDERSTANDING$' and explain the discrepancy."
        )
        
        # ⭐ 关键：也需要 context_prompt
        full_prompt = f"{context}\n\n{confirm_prompt}"
        
        messages = [
            {"role": "system", "content": ai_cfg["system_prompt"]},
            {"role": "user", "content": full_prompt}
        ]
        
        response = self.ai_agent.chat(messages)
        return response["content"]
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict"""
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
            "updated_at": self.updated_at
        }
    

    def save_to_db(self):
        """保存到数据库（显式列名，22 个占位符）"""
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
                updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
            self.updated_at
        ))

        conn.commit()
        conn.close()
    
    @staticmethod
    def load_from_db(session_id: str) -> Optional['NegotiationSession']:
        """从数据库加载"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("SELECT * FROM negotiation_sessions WHERE session_id = ?", (session_id,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # 重建 session
        session = NegotiationSession(
            session_id=row[0],
            student_id=row[1],
            student_name=row[2],
            scenario_name=row[3],
            student_role=row[4],
            ai_model=row[6],
            student_goes_first=row[7],
            use_memory=row[8],
            use_plan=row[9],
        )
        
        # 恢复状态
        session.ai_role = row[5]
        session.current_round = row[10]
        session.total_rounds = row[11]
        session.transcript = json.loads(row[12])
        session.ai_memory = row[13] or ""
        session.ai_plan = row[14] or ""
        session.student_deal_json = json.loads(row[15]) if row[15] else None
        session.ai_deal_json = json.loads(row[16]) if row[16] else None
        session.deal_reached = row[17]
        session.deal_failed = row[18]
        session.status = row[19]
        session.created_at = row[20]
        session.updated_at = row[21]
        
        return session

# ============================================================================
# Pydantic Models
# ============================================================================
class StartNegotiationRequest(BaseModel):
    student_id: str
    student_name: str
    scenario_name: str
    student_role: str  # "side1" or "side2"
    ai_model: str = "anthropic/claude-3-sonnet"  # ⭐ 新增：模型选择
    randomize_first_turn: bool = True  # ⭐ 新增：随机先手
    use_memory: bool = True
    use_plan: bool = True

class SendMessageRequest(BaseModel):
    message: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def read_root():
    """健康检查"""
    return {"status": "ok", "message": "Negotiation Practice API"}

@app.get("/scenarios")
def list_scenarios():
    """列出所有可用场景"""
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
                    "description": config.get("description", ""),
                    "side1_label": config.get("side1", {}).get("label", "Side 1"),
                    "side2_label": config.get("side2", {}).get("label", "Side 2"),
                })
        except:
            continue
    
    return {"scenarios": scenarios}

@app.post("/negotiation/start")
def start_negotiation(request: StartNegotiationRequest):
    """
    开始新的谈判会话
    
    ⭐ 新增功能:
    1. 随机先手（randomize_first_turn=True）
    2. 模型选择（ai_model 参数）
    """
    try:
        # 生成 session_id
        session_id = str(uuid.uuid4())
        
        # ⭐ 随机先手（如果启用）
        if request.randomize_first_turn:
            student_goes_first = random.choice([True, False])
        else:
            student_goes_first = True  # 默认学生先手
        
        # 创建会话
        session = NegotiationSession(
            session_id=session_id,
            student_id=request.student_id,
            student_name=request.student_name,
            scenario_name=request.scenario_name,
            student_role=request.student_role,
            ai_model=request.ai_model,  # ⭐ 使用用户选择的模型
            student_goes_first=student_goes_first,
            use_memory=request.use_memory,
            use_plan=request.use_plan,
        )
        
        # 如果 AI 先手，生成第一条消息
        ai_first_message = None
        if not student_goes_first:
            ai_response = session._generate_ai_response()
            ai_cfg = session.scenario_config[session.ai_role]
            ai_label = ai_cfg.get("label", "AI")
            session.transcript.append(f"{ai_label}: {ai_response}")
            ai_first_message = ai_response
        
        # 保存到数据库
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
    发送学生消息并获取 AI 回复
    """
    # 加载会话
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # 处理消息
    result = session.process_student_message(request.message)
    
    # 保存到数据库
    session.save_to_db()
    
    return result

@app.get("/negotiation/{session_id}/status")
def get_status(session_id: str):
    """获取会话状态"""
    session = NegotiationSession.load_from_db(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()

@app.get("/negotiation/{session_id}/transcript")
def get_transcript(session_id: str):
    """获取完整对话记录"""
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

@app.get("/download_db")
def download_db(secret: Optional[str] = None):
    """
    下载数据库文件
    需要提供正确的 secret key（从环境变量 DOWNLOAD_KEY 读取）
    
    使用方法：
    GET /download_db?secret=your-secret-key
    """
    allowed = os.getenv("DOWNLOAD_KEY")
    if allowed and secret != allowed:
        raise HTTPException(status_code=403, detail="unauthorized")
    return FileResponse(DB_PATH, filename="negotiations.db")

@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ============================================================================
# 运行
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)