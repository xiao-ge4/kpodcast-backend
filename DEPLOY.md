# KPodcast 后端部署指南

## 部署到 Render

### 1. 创建 GitHub 仓库

1. 访问 https://github.com/new
2. 仓库名：`podcast-backend`
3. 设为 Public
4. 点击 "Create repository"

### 2. 推送代码

```bash
cd podcast-backend-deploy
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的用户名/podcast-backend.git
git branch -M main
git push -u origin main
```

### 3. 在 Render 部署

1. 访问 https://render.com/
2. 点击 "New +" → "Web Service"
3. 选择 `podcast-backend` 仓库
4. 配置：
   - **Name**: `kpodcast-backend`
   - **Region**: `Singapore`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt && playwright install chromium`
   - **Start Command**: `uvicorn api_main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (或 Starter)

### 4. 配置环境变量

在 Render Dashboard → Environment 中添加以下变量：

#### 必需的环境变量

| 变量名 | 说明 | 示例值 |
|-------|------|-------|
| `PODCAST_TENCENT_SECRET_ID` | 腾讯云 SecretId | `AKIDxxxxxxxx` |
| `PODCAST_TENCENT_SECRET_KEY` | 腾讯云 SecretKey | `xxxxxxxx` |
| `PODCAST_TENCENT_REGION` | 腾讯云地域 | `ap-beijing` |
| `PODCAST_COS_ENABLED` | 启用 COS 存储 | `true` |
| `PODCAST_COS_SECRET_ID` | COS SecretId（可与上面相同） | `AKIDxxxxxxxx` |
| `PODCAST_COS_SECRET_KEY` | COS SecretKey（可与上面相同） | `xxxxxxxx` |
| `PODCAST_COS_REGION` | COS 地域 | `ap-beijing` |
| `PODCAST_COS_BUCKET` | COS 存储桶名 | `podcast-1308411753` |

#### 可选的环境变量

| 变量名 | 说明 | 默认值 |
|-------|------|-------|
| `PODCAST_BOCHA_BASE_URL` | Bocha 搜索 API | - |
| `PODCAST_BOCHA_API_ID` | Bocha API ID | - |
| `PODCAST_BOCHA_API_KEY` | Bocha API Key | - |
| `PODCAST_HUNYUAN_API_MODEL` | 混元模型 | `hunyuan-turbos-latest` |
| `PODCAST_TENCENT_VOICE_NUMBER` | 音色列表 JSON | `[501006, 601007]` |
| `PODCAST_TENCENT_VOICE_ROLE` | 音色名称 JSON | `["千嶂", "爱小叶"]` |

#### Python 版本

| 变量名 | 值 |
|-------|---|
| `PYTHON_VERSION` | `3.11.0` |

### 5. 等待部署完成

部署完成后，你会得到一个 URL，类似：
```
https://kpodcast-backend.onrender.com
```

### 6. 测试后端

访问 `https://你的后端URL/docs` 应该能看到 FastAPI 文档。

---

## 前端配置

在前端项目中设置 API 地址：

### 方式1：环境变量（推荐）

在 Vercel 中添加环境变量：
```
VITE_PODCAST_API_BASE=https://kpodcast-backend.onrender.com
```

### 方式2：HTML 中设置

在加载 app.js 之前添加：
```html
<script>
  window.PODCAST_API_BASE = 'https://kpodcast-backend.onrender.com';
</script>
```

---

## COS 配置要求

确保腾讯云 COS 存储桶已配置：

1. **访问权限**：公有读私有写
2. **CORS 规则**：
   - 来源 Origin: `*` 或你的域名
   - 操作 Methods: `GET`, `HEAD`, `PUT`
   - Allow-Headers: `*`
   - Expose-Headers: `Content-Length, Content-Type`

---

## 注意事项

1. **Render 免费版**会在 15 分钟无请求后休眠，首次访问需等待
2. **敏感信息**不要提交到 Git，使用环境变量
3. **CORS**：后端已配置允许所有域名，生产环境建议限制
