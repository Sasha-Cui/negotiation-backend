# Frontend-Backend Integration Guide

## Overview

This document explains the complete integration between the new frontend and your existing backend.

---

## 🎯 What Changed

### OLD Frontend Problems ❌
- No scenario selection
- No session management
- Called wrong endpoints (`/message` instead of `/negotiation/{session_id}/message`)
- Frontend maintained its own transcript (conflict with backend)
- Incompatible data formats

### NEW Frontend Features ✅
- Complete scenario selection UI
- Full session lifecycle management
- Correct API endpoint calls
- Deal confirmation flow
- Round tracking
- Automatic feedback request
- Error handling

---

## 🔄 Complete User Flow

### 1. Scenario Selection Phase

```
User opens page
  ↓
Frontend calls: GET /scenarios
  ↓
User selects:
  - Name
  - Scenario
  - Role (side1/side2)
  - Number of rounds
  ↓
User clicks "Start Negotiation"
  ↓
Frontend calls: POST /negotiation/start
  Body: {
    student_id: string,
    student_name: string,
    scenario_name: string,
    student_role: "side1" | "side2",
    total_rounds: number,
    use_memory: true,
    use_plan: true
  }
  ↓
Backend returns: {
    session_id: string,
    your_role: string,
    ai_role: string,
    total_rounds: number,
    initial_prompt: string
  }
  ↓
Frontend stores session_id
Frontend displays initial_prompt
```

### 2. Negotiation Phase

```
User types message and clicks "Send"
  ↓
Frontend calls: POST /negotiation/{session_id}/message
  Body: { message: string }
  ↓
Backend processes:
  - Updates Memory (if enabled)
  - Generates Plan (if enabled)
  - Generates AI response
  - Checks for $DEAL_REACHED$ token
  - Increments round counter
  ↓
Backend returns: {
    ai_response: string,
    current_round: number,
    rounds_remaining: number,
    deal_reached: bool,
    session_status: string,
    // Optional fields:
    needs_student_confirmation: bool,
    ai_offer: dict,
    verified_agreement: bool,
    timed_out: bool
  }
  ↓
Frontend updates UI:
  - Display AI response
  - Update round counter
  - Handle deal scenarios
```

### 3. Deal Scenarios

#### Scenario A: AI Initiates Deal

```
AI response contains $DEAL_REACHED$ token
  ↓
Backend sets needs_student_confirmation: true
  ↓
Frontend shows modal dialog with AI's proposed terms
  ↓
User clicks "Accept" or "Reject"
  ↓
Frontend calls: POST /negotiation/{session_id}/confirm
  Body: {
    confirmed: bool,
    deal_terms: dict (if confirmed=true)
  }
  ↓
Backend verifies consistency
  ↓
Frontend displays result
```

#### Scenario B: Student Initiates Deal

```
Student types $DEAL_REACHED$ + JSON in message
  ↓
Backend extracts JSON
  ↓
Backend requests AI confirmation
  ↓
AI confirms or sends $DEAL_MISUNDERSTANDING$
  ↓
Backend returns result
  ↓
Frontend displays outcome
```

### 4. Feedback Phase

```
User clicks "End Negotiation & Get Feedback"
OR negotiation completes (deal/timeout)
  ↓
Frontend calls: POST /negotiation/{session_id}/feedback
  ↓
Backend:
  - Loads complete transcript
  - Calls GPT-4o for analysis
  - Returns detailed feedback
  ↓
Frontend displays feedback in special message box
```

---

## 📡 API Endpoint Mapping

| Frontend Action | HTTP Call | Backend Endpoint |
|----------------|-----------|------------------|
| Load scenarios | `GET /scenarios` | `@app.get("/scenarios")` |
| Get scenario details | `GET /scenarios/{name}` | `@app.get("/scenarios/{scenario_name}")` |
| Start negotiation | `POST /negotiation/start` | `@app.post("/negotiation/start")` |
| Send message | `POST /negotiation/{session_id}/message` | `@app.post("/negotiation/{session_id}/message")` |
| Confirm deal | `POST /negotiation/{session_id}/confirm` | `@app.post("/negotiation/{session_id}/confirm")` |
| Get feedback | `POST /negotiation/{session_id}/feedback` | `@app.post("/negotiation/{session_id}/feedback")` |

---

## 🎨 UI Components

### 1. Scenario Selection Screen
- Student name input (stored in localStorage)
- Scenario dropdown (dynamically loaded from backend)
- Role selection (side1/side2, labels from scenario config)
- Round number input
- Start button

### 2. Chat Screen
- Session info bar (scenario name, role, round counter)
- Chat history area (60vh height)
- Message composer (multi-line textarea)
- Send button
- End Negotiation button
- Reset button

### 3. Message Types
- **User messages**: Blue background, right-aligned
- **AI messages**: Green background, left-aligned
- **System messages**: Yellow background, centered, italic
- **Feedback messages**: Orange background, centered
- **Deal messages**: Light green background, centered, special border

### 4. Deal Confirmation Dialog
- Modal overlay
- Display AI's proposed JSON terms
- Accept/Reject buttons

---

## 🔧 Key Frontend Functions

### State Management
```javascript
let sessionId = null;           // Current negotiation session ID
let studentId = "student1234";  // Random ID for tracking
let studentName = "";           // Stored in localStorage
let currentRound = 1;           // Current round number
let totalRounds = 10;           // Total rounds for this session
let scenariosData = [];         // List of available scenarios
let currentScenarioDetails = null;  // Details of selected scenario
let pendingAiDeal = null;       // AI's deal offer awaiting confirmation
```

### Core Functions
- `loadScenarios()`: Fetch all available scenarios from backend
- `loadScenarioDetails(scenarioId)`: Fetch specific scenario info
- `startNegotiation()`: Initialize new session
- `sendMessage()`: Send user message and handle response
- `showDealConfirmationDialog(aiOffer)`: Display deal modal
- `handleDealConfirmation(confirmed)`: Process deal acceptance/rejection
- `requestFeedback()`: Get final performance feedback
- `resetSession()`: Reload page for new negotiation

---

## 🛡️ Error Handling

### Frontend Validation
- Name cannot be empty
- Scenario must be selected
- Role must be selected
- Rounds must be between 3-20

### API Error Handling
```javascript
try {
  const res = await fetch(...);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Request failed: ${res.status}`);
  }
  // Process successful response
} catch (err) {
  console.error("Error:", err);
  appendMessage("system", "❌ Error: " + err.message);
}
```

### Network Errors
- Displays user-friendly error messages in chat
- Does not crash the application
- Allows retry

---

## 💾 Data Persistence

### LocalStorage
```javascript
// Stores student name for convenience
localStorage.setItem("student_name", studentName);
let studentName = localStorage.getItem("student_name") || "";
```

### Backend Session Storage
- All negotiation state stored in SQLite database
- Session can be recovered using `session_id`
- Transcript, memory, plan all persisted server-side

---

## 🎯 Deal Detection & Confirmation

### Student-Initiated Deal
1. Student types `$DEAL_REACHED$` followed by JSON
2. Backend detects token in `_is_deal_token()`
3. Backend extracts JSON using `json.loads()`
4. Backend requests AI confirmation
5. AI responds with JSON or `$DEAL_MISUNDERSTANDING$`
6. Backend verifies JSON consistency
7. Result returned to frontend

### AI-Initiated Deal
1. AI generates `$DEAL_REACHED$` token in response
2. Backend detects token
3. Backend extracts AI's JSON offer
4. Backend returns `needs_student_confirmation: true`
5. Frontend shows modal with terms
6. Student accepts/rejects
7. Frontend calls `/confirm` endpoint
8. Backend verifies and completes deal

---

## 🚀 Deployment Checklist

### Frontend
- [ ] Upload `index.html` to hosting (Vercel, Netlify, etc.)
- [ ] Verify `BACKEND_URL` points to your Render backend
- [ ] Test in browser

### Backend
- [ ] Ensure `OPENROUTER_API_KEY` environment variable is set
- [ ] Create `/app/scenarios` directory
- [ ] Upload YAML scenario files
- [ ] Verify `/data` directory exists for SQLite
- [ ] Deploy to Render
- [ ] Test `/health` endpoint

### YAML Scenarios
Required structure:
```yaml
name: "Scenario Name"
description: "Brief description"
num_rounds: 10

side1:
  label: "Side 1 Label"
  batna: "Best Alternative"
  system_prompt: "Role instructions..."
  context_prompt: "Scenario facts..."
  initial_offer_prompt: "Opening instructions..."

side2:
  label: "Side 2 Label"
  batna: "Best Alternative"
  system_prompt: "Role instructions..."
  context_prompt: "Scenario facts..."
  initial_offer_prompt: "Opening instructions..."

json_schema: '{"property": "type", ...}'
```

---

## 🐛 Debugging Tips

### Check Backend Logs
```bash
# On Render dashboard, view logs to see:
# - Scenario loading
# - Session creation
# - Message processing
# - Error traces
```

### Browser Console
```javascript
// Check for errors
console.log("Session ID:", sessionId);
console.log("Current Round:", currentRound);

// Test API directly
fetch("https://negotiation-backend-ut5t.onrender.com/scenarios")
  .then(r => r.json())
  .then(console.log);
```

### Common Issues

**Issue**: "Scenario not found"
- **Fix**: Ensure YAML file exists in `/app/scenarios/`
- **Fix**: Check file name matches `scenario_name` parameter

**Issue**: "Session not found"
- **Fix**: Verify `session_id` is being stored correctly
- **Fix**: Check if session was created successfully

**Issue**: AI not responding
- **Fix**: Check `OPENROUTER_API_KEY` is set
- **Fix**: Verify OpenRouter API credits

**Issue**: Deal not detecting
- **Fix**: Ensure `$DEAL_REACHED$` is at start of message
- **Fix**: Check JSON format matches schema

---

## 📊 Session Flow Diagram

```
┌─────────────────┐
│  User Opens App │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Load Scenarios  │◄─── GET /scenarios
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Select Options  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Start Session   │◄─── POST /negotiation/start
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Round 1...N     │◄─── POST /negotiation/{id}/message
└────────┬────────┘      (loop until deal or timeout)
         │
         ▼
    ┌───┴───┐
    │ Deal? │
    └───┬───┘
        │
   ┌────┴────┐
   │   Yes   │   No
   │         │    │
   ▼         ▼    ▼
┌──────┐  ┌──────────┐
│Verify│  │ Timeout  │
└──┬───┘  └────┬─────┘
   │           │
   └─────┬─────┘
         │
         ▼
┌─────────────────┐
│  Get Feedback   │◄─── POST /negotiation/{id}/feedback
└─────────────────┘
```

---

## ✅ Testing Checklist

- [ ] Can load scenarios
- [ ] Can select scenario and role
- [ ] Can start negotiation
- [ ] Initial prompt displays
- [ ] Can send messages
- [ ] AI responds correctly
- [ ] Round counter increments
- [ ] Can propose deal as student
- [ ] Can accept/reject AI's deal
- [ ] Deal verification works
- [ ] Timeout detection works
- [ ] Can get feedback
- [ ] Can reset session
- [ ] Error messages display properly
- [ ] Works on mobile devices

---

## 🎓 Usage Guide for Students

1. **Enter your full name** (required for feedback)
2. **Select a negotiation scenario** from the dropdown
3. **Choose your role** (which side you want to represent)
4. **Set number of rounds** (default: 10)
5. **Click "Start Negotiation"**
6. **Read the initial prompt** carefully
7. **Type your messages** in the text box
8. **Press Ctrl/Cmd + Enter** to send (or click Send button)
9. **Negotiate strategically** - the AI has Memory and Planning systems
10. **Propose a deal** by typing `$DEAL_REACHED$` followed by JSON, OR
11. **Accept AI's deal** when the confirmation dialog appears
12. **Click "End Negotiation"** to get detailed feedback on your performance

---

## 📝 Notes

- Sessions are stored server-side, so browser refresh will lose current session
- Student name is saved in localStorage for convenience
- Round limit enforced by backend
- Feedback uses GPT-4o for high-quality analysis
- All negotiation data stored in SQLite for future analysis

---

## 🔗 Related Files

- `index.html` - Complete frontend application (THIS FILE)
- `main.py` - Backend FastAPI application
- `openai_wrapper.py` - OpenAI/OpenRouter API wrapper
- `scenarios/*.yaml` - Negotiation scenario configurations
- `negotiations.db` - SQLite database (auto-created)
