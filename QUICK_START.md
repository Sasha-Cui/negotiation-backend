# Quick Start Guide

## 🚀 Get Your System Running in 5 Minutes

### Step 1: Deploy Frontend

#### Option A: Vercel (Recommended)
```bash
1. Go to https://vercel.com
2. Click "New Project"
3. Upload index.html
4. Deploy
```

#### Option B: Netlify
```bash
1. Go to https://netlify.com
2. Drag and drop index.html
3. Deploy
```

#### Option C: GitHub Pages
```bash
1. Create a new GitHub repo
2. Upload index.html
3. Enable GitHub Pages in Settings
```

### Step 2: Verify Backend

Your backend is already deployed at:
```
https://negotiation-backend-ut5t.onrender.com
```

Test it:
```bash
curl https://negotiation-backend-ut5t.onrender.com/health
# Should return: {"status":"healthy","timestamp":"..."}
```

### Step 3: Add Scenario Files

Your backend needs scenario YAML files. Here's a minimal example:

Create `/app/scenarios/example.yaml` on your Render deployment:

```yaml
name: "Simple Negotiation"
description: "A basic two-party negotiation"
num_rounds: 10

side1:
  label: "Buyer"
  batna: "$50,000"
  system_prompt: |
    You are a buyer negotiating a purchase.
    Be fair but firm. Your goal is to get the best price.
  
  context_prompt: |
    SCENARIO: You are buying equipment.
    BUDGET: $100,000
    PRIORITIES:
    - Price (most important)
    - Delivery time
    - Warranty
  
  initial_offer_prompt: |
    Make your opening offer. Be specific about:
    1. Price you're willing to pay
    2. Delivery expectations
    3. Warranty requirements

side2:
  label: "Seller"
  batna: "$60,000"
  system_prompt: |
    You are a seller negotiating a sale.
    Be professional and aim for mutual benefit.
  
  context_prompt: |
    SCENARIO: You are selling equipment.
    MINIMUM PRICE: $70,000
    PRIORITIES:
    - Price (most important)
    - Quick sale
    - Future business relationship
  
  initial_offer_prompt: |
    Respond to the buyer's offer. Address:
    1. Your asking price
    2. Delivery timeline
    3. Warranty terms

json_schema: '{
  "type": "object",
  "properties": {
    "price": {"type": "number"},
    "delivery_days": {"type": "number"},
    "warranty_months": {"type": "number"}
  },
  "required": ["price", "delivery_days", "warranty_months"]
}'
```

### Step 4: Test the System

1. Open your deployed frontend URL
2. Enter your name
3. Select "Simple Negotiation" scenario
4. Choose "Buyer" or "Seller" role
5. Click "Start Negotiation"
6. Try a few messages
7. Click "End Negotiation & Get Feedback"

### Step 5: Verify Everything Works

✅ Checklist:
- [ ] Scenarios load in dropdown
- [ ] Can start negotiation
- [ ] AI responds to messages
- [ ] Round counter increments
- [ ] Can get feedback at end
- [ ] No console errors

---

## 🔧 Troubleshooting

### "No scenarios available"
**Problem**: Backend can't find scenario files

**Solutions**:
1. Check `/app/scenarios/` directory exists
2. Verify YAML files are valid
3. Check Render logs for errors

```bash
# On Render, check environment variable
echo $SCENARIOS_DIR
# Should be: /app/scenarios
```

### "Failed to start negotiation"
**Problem**: Missing OPENROUTER_API_KEY

**Solution**:
1. Go to Render dashboard
2. Click your service
3. Go to "Environment" tab
4. Add: `OPENROUTER_API_KEY = your-key-here`
5. Redeploy

### "AI not responding"
**Problem**: OpenRouter API issue

**Solutions**:
1. Check API key is valid
2. Verify OpenRouter account has credits
3. Check backend logs for error messages

### Frontend shows old version
**Problem**: Browser cache

**Solution**:
```bash
# Hard refresh
Windows/Linux: Ctrl + Shift + R
Mac: Cmd + Shift + R
```

---

## 📂 File Structure

```
Your Project/
├── index.html                    ← Frontend (deploy to Vercel/Netlify)
├── main.py                       ← Backend (already on Render)
├── openai_wrapper.py            ← Backend helper
├── scenarios/
│   ├── example.yaml             ← Add your scenarios here
│   ├── negotiation1.yaml
│   └── negotiation2.yaml
└── data/
    └── negotiations.db          ← Auto-created by backend
```

---

## 🎯 Next Steps

1. **Create more scenarios**: Add YAML files to `/app/scenarios/`
2. **Customize styling**: Edit CSS in index.html
3. **Add analytics**: Track student performance
4. **Export data**: Use `/download_db` endpoint to get SQLite file
5. **Add authentication**: Implement user login if needed

---

## 📞 Need Help?

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| Can't load scenarios | Check backend logs, verify YAML syntax |
| API errors | Verify OPENROUTER_API_KEY environment variable |
| Frontend not connecting | Check BACKEND_URL in index.html matches Render URL |
| Database errors | Verify /data directory exists and is writable |

---

## ✨ You're Done!

Your negotiation system is now live and ready for students to use!

Students can:
- ✅ Choose different scenarios
- ✅ Practice negotiation skills
- ✅ Get AI-powered feedback
- ✅ Track their progress

Next: Create more sophisticated scenarios and invite students to practice!
