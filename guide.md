# 部署步骤指南 - 一步一步来

## 📁 第一步：整理你的文件结构

根据你的截图，你现在的文件夹结构是这样的：

```
negotiation-backend/
├── .gitignore
├── index 1.html          ⚠️ 旧的前端（不需要了）
├── index.html            ✅ 新的前端（刚下载的）
├── INTEGRATION_GUIDE.md  📄 文档
├── main_add.py          ⚠️ 不知道是什么
├── main.py              ✅ 你的后端代码
├── openai_wrapper.py    ✅ 后端辅助文件
├── QUICK_START.md       📄 文档
├── README.md            📄 文档
└── requirements.txt     ✅ Python 依赖
```

### 你需要创建一个新文件夹：

```
negotiation-backend/
└── scenarios/           ← 创建这个文件夹！
    ├── scenario1.yaml   ← 把你的 YAML 文件放这里
    ├── scenario2.yaml
    └── scenario3.yaml
```

---

## 🔧 第二步：创建 scenarios 文件夹并添加 YAML 文件

### 在本地电脑操作：

1. **在 `negotiation-backend` 文件夹里创建新文件夹**
   ```
   右键 → 新建文件夹 → 命名为 "scenarios"
   ```

2. **把你所有的 YAML 场景文件移动到 `scenarios` 文件夹**
   ```
   比如：
   scenarios/
   ├── buyer_seller.yaml
   ├── employer_employee.yaml
   └── landlord_tenant.yaml
   ```

3. **检查 YAML 文件格式**
   
   每个 YAML 文件必须包含这些字段：
   ```yaml
   name: "场景名称"
   description: "场景描述"
   num_rounds: 10
   
   side1:
     label: "角色1名称"
     batna: "替代方案"
     system_prompt: "系统提示..."
     context_prompt: "背景信息..."
     initial_offer_prompt: "开场指示..."
   
   side2:
     label: "角色2名称"
     batna: "替代方案"
     system_prompt: "系统提示..."
     context_prompt: "背景信息..."
     initial_offer_prompt: "开场指示..."
   
   json_schema: '{"type": "object", ...}'
   ```

---

## 🚀 第三步：部署到 Render

### 3.1 推送代码到 GitHub

在你的项目文件夹打开终端：

```bash
# 1. 查看当前状态
git status

# 2. 添加新文件
git add scenarios/
git add index.html

# 3. 提交更改
git commit -m "Add scenarios folder and new frontend"

# 4. 推送到 GitHub
git push origin main
```

### 3.2 在 Render 更新部署

1. **打开 Render Dashboard**
   - 访问：https://dashboard.render.com
   
2. **找到你的服务** `negotiation-backend`

3. **触发重新部署**
   - 点击 "Manual Deploy" → "Deploy latest commit"
   - 或者等待自动部署（如果你开启了 auto-deploy）

4. **等待部署完成**（大约 2-3 分钟）

### 3.3 验证后端配置

确保这些环境变量已设置：

```
OPENROUTER_API_KEY = your-api-key-here
DB_PATH = /data/negotiations.db
SCENARIOS_DIR = /app/scenarios
```

在 Render Dashboard 中检查：
- 点击你的服务
- 进入 "Environment" 标签
- 确认以上变量存在

---

## 🌐 第四步：部署前端

### 选项 A：使用 Vercel（推荐）

1. **访问** https://vercel.com/new

2. **选择部署方式**
   - "Import Git Repository"（如果你的前端在 GitHub）
   - 或者 "Deploy with file upload"

3. **上传 index.html**
   - 拖拽 `index.html` 文件到 Vercel
   - 或者连接你的 GitHub 仓库

4. **配置**
   - Framework Preset: 选择 "Other"
   - Build Command: 留空
   - Output Directory: `.`

5. **Deploy！**

6. **获取 URL**
   - 部署完成后，你会得到一个 URL
   - 例如：`https://your-project.vercel.app`

### 选项 B：使用 Netlify

1. **访问** https://app.netlify.com/drop

2. **拖拽文件**
   - 直接把 `index.html` 拖进去

3. **部署完成！**
   - 你会得到一个 URL
   - 例如：`https://random-name.netlify.app`

---

## ✅ 第五步：测试系统

### 5.1 测试后端

在浏览器访问：
```
https://negotiation-backend-ut5t.onrender.com/scenarios
```

你应该看到：
```json
{
  "scenarios": [
    {
      "id": "scenario1",
      "name": "场景名称",
      "description": "...",
      ...
    },
    ...
  ]
}
```

如果看到 `{"scenarios": []}`，说明：
- ❌ scenarios 文件夹没有正确上传
- ❌ YAML 文件格式错误
- ❌ SCENARIOS_DIR 环境变量设置错误

### 5.2 测试前端

1. **打开前端 URL**（Vercel 或 Netlify 给你的）

2. **检查场景下拉框**
   - 应该显示你的所有场景
   - 如果显示空白，检查浏览器 Console（F12）

3. **完整测试流程**
   - 输入姓名
   - 选择场景
   - 选择角色
   - 点击 "Start Negotiation"
   - 发送几条消息
   - 检查 AI 回复
   - 点击 "End Negotiation & Get Feedback"

---

## 🐛 常见问题排查

### 问题1：前端显示 "Failed to load scenarios"

**原因**：前端无法连接后端

**解决**：
1. 检查 `index.html` 第 168 行：
   ```javascript
   const BACKEND_URL = "https://negotiation-backend-ut5t.onrender.com";
   ```
   确保这个 URL 正确

2. 测试后端是否在线：
   ```
   https://negotiation-backend-ut5t.onrender.com/health
   ```
   应该返回：`{"status":"healthy","timestamp":"..."}`

### 问题2：场景列表为空

**原因**：scenarios 文件夹没有正确部署

**解决**：
1. 检查 Render 日志：
   ```
   Dashboard → Your Service → Logs
   ```

2. 看是否有错误消息如：
   ```
   FileNotFoundError: [Errno 2] No such file or directory: '/app/scenarios'
   ```

3. 如果有这个错误，确保：
   - scenarios 文件夹在 Git 仓库中
   - 已经 push 到 GitHub
   - Render 已经重新部署

### 问题3：YAML 加载失败

**原因**：YAML 文件格式错误

**解决**：
1. 检查 Render 日志，看错误消息

2. 常见 YAML 错误：
   - 缩进不一致（必须用空格，不能用 Tab）
   - 引号不匹配
   - 冒号后面没有空格

3. 在线验证 YAML：
   - 访问 https://www.yamllint.com/
   - 粘贴你的 YAML 内容
   - 检查语法错误

### 问题4：AI 不回复

**原因**：OpenRouter API Key 问题

**解决**：
1. 检查 Render 环境变量：
   - `OPENROUTER_API_KEY` 是否设置

2. 检查 OpenRouter 账户：
   - 访问 https://openrouter.ai/
   - 查看 API credits 余额
   - 查看 API 使用日志

---

## 📋 完整检查清单

部署前检查：

- [ ] 创建了 `scenarios/` 文件夹
- [ ] 所有 YAML 文件在 `scenarios/` 文件夹中
- [ ] YAML 文件格式正确（可以用 yamllint 验证）
- [ ] 下载了新的 `index.html`
- [ ] Git add + commit + push 所有文件

Render 后端检查：

- [ ] 代码已推送到 GitHub
- [ ] Render 自动部署完成（或手动触发部署）
- [ ] 环境变量设置正确：
  - [ ] OPENROUTER_API_KEY
  - [ ] DB_PATH = /data/negotiations.db
  - [ ] SCENARIOS_DIR = /app/scenarios
- [ ] 访问 `/health` 端点返回正常
- [ ] 访问 `/scenarios` 端点返回场景列表

前端部署检查：

- [ ] 前端已部署到 Vercel/Netlify
- [ ] 获得前端 URL
- [ ] BACKEND_URL 指向正确的 Render 地址
- [ ] 浏览器能访问前端页面

功能测试：

- [ ] 场景下拉框显示所有场景
- [ ] 可以选择角色
- [ ] 可以开始谈判
- [ ] AI 能正常回复
- [ ] 轮次计数器正常
- [ ] 能获取反馈
- [ ] 没有 Console 错误

---

## 🎯 总结：你需要做的三件事

### 1️⃣ 本地准备文件
```bash
# 在项目根目录
mkdir scenarios
# 把所有 YAML 文件移动到 scenarios/
# 确保有新的 index.html
```

### 2️⃣ 推送到 GitHub
```bash
git add .
git commit -m "Add scenarios and new frontend"
git push
```

### 3️⃣ 部署前端
- 把 `index.html` 上传到 Vercel 或 Netlify
- 获取 URL，测试功能

---

## 📞 需要帮助？

如果遇到问题：

1. **查看 Render 日志**（最重要！）
   - Dashboard → Your Service → Logs
   - 复制错误消息

2. **检查浏览器 Console**
   - 按 F12 打开开发者工具
   - 查看 Console 标签
   - 查看 Network 标签的请求失败信息

3. **测试各个端点**
   ```
   /health       - 后端健康检查
   /scenarios    - 场景列表
   ```

把错误消息发给我，我可以帮你诊断！

---

## 🎉 完成后

你的学生就可以：
- 访问前端 URL
- 选择不同场景
- 进行谈判练习
- 获得 AI 反馈

祝部署顺利！🚀