# 🎙️ AI 播客生成器

> ⚠️ **私人仓库**：包含真实 API 密钥，请勿公开

基于大语言模型的智能播客生成系统，一键生成专业播客内容。

## 必做测试

【腾讯文档】提交流程：复制本表— 根据测试集输入要求 —  批量跑数据 - 提供输出（需每行对应） - 表格上传到AtomGit。

提交截止时间：11月15日2359（如此部分需要延期，请主动联系组委会
https://docs.qq.com/sheet/DTW9oa05NbHpuUkZE?tab=BB08J2

## 项目特色

- 🚀 **一键生成**：输入话题，自动生成完整播客
- 🎯 **多模式输入**：支持搜索、URL、文档、PDF 等多种输入方式
- 🤖 **智能对话**：生成自然流畅的主持人对话
- 🎵 **专业音频**：自动合成语音，添加背景音乐
- 🌐 **双语支持**：支持中英文播客生成

## 快速体验

### 安装

```bash
# 1. 克隆项目
git clone [项目地址]
cd podcast-generator-git

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装浏览器（用于网页提取）
playwright install chromium

# 4. 配置已包含（私人仓库）
# config.ini 文件已包含真实配置
```

### 运行

```bash
python app.py
```

打开浏览器访问 http://localhost:7860

## 使用场景

### 1. 热点新闻播客
- 输入新闻关键词或 URL
- 自动搜索相关资料
- 生成专业新闻播客

### 2. 知识科普播客
- 上传 PDF 文献
- 智能提取核心内容
- 生成通俗易懂的科普播客

### 3. 故事类播客
- 输入故事主题
- 自动创作故事内容
- 生成生动有趣的故事播客

### 4. 企业内容播客
- 输入企业资料
- 自定义播报风格
- 生成企业宣传播客

## 核心功能详解

### 智能搜索
- 使用博查 API 搜索最新资料
- 自动筛选高质量内容
- 智能整合多源信息

### 内容提取
- **网页提取**：支持各类新闻网站、博客
- **PDF 处理**：支持学术论文、报告
- **动态渲染**：支持 JavaScript 渲染的页面

### 脚本生成
- **风格多样**：新闻、故事、知识、娱乐等
- **智能分析**：根据内容自动调整时长
- **对话自然**：模拟真实主持人对话

### 语音合成
- **12 种音色**：适合不同场景
- **情感丰富**：自然语调变化
- **背景音乐**：多种风格可选

## 项目结构

```
podcast-generator-git/
├── app.py              # 主程序入口
├── clients/            # API 客户端
│   ├── hunyuan_api_client.py    # 混元大模型
│   ├── tts_client.py             # 语音合成
│   └── search_client.py         # 搜索服务
├── utils/              # 工具函数
│   ├── config_loader.py         # 配置加载
│   ├── enhanced_url_fetcher.py  # URL 提取
│   └── pdf_extractor.py         # PDF 处理
├── pipeline/           # 生成流程
│   └── podcast_pipeline_new.py  # 播客生成
├── assets/             # 资源文件
│   └── bgm/           # 背景音乐
└── config.ini         # 配置文件
```

## 配置说明

### 必需配置
```ini
[bocha]
api_key = xxx  # 博查搜索 API

[tencent]
secret_id = xxx   # 腾讯云 ID
secret_key = xxx  # 腾讯云密钥
```

### 可选配置
```ini
[hunyuan_api]
temperature = 0.8  # 创造性 (0-1)
max_tokens = 10000 # 输出长度

[web_extract]
render_mode = playwright  # 渲染模式
cookie = xxx              # 网站 Cookie
```

## 常见问题

### Q: 如何获取 API 密钥？
A: 
- 博查 API：访问 [博查官网](https://www.bochaai.com) 注册
- 腾讯云：访问 [腾讯云控制台](https://console.cloud.tencent.com) 申请

### Q: 支持哪些网站的内容提取？
A: 支持大部分新闻网站、博客、百度、知乎等。对于需要登录的网站，可配置 Cookie。

### Q: 生成的播客多长？
A: 根据内容自动调整，通常 5-15 分钟。可通过特殊指令控制长度。

### Q: 如何生成英文播客？
A: 在特殊指令中输入"使用英文"或"Generate in English"。

## 性能优化

### 降低 Token 消耗
- PDF 模式：限制输入长度
- 调整 `max_tokens` 参数
- 使用更精准的搜索关键词

### 提升提取成功率
- 安装 Playwright：`playwright install`
- 配置正确的 User-Agent
- 添加必要的 Cookie

## 开发计划

- [ ] 支持更多语言
- [ ] 添加语音识别输入
- [ ] 支持视频内容提取
- [ ] 批量生成功能
- [ ] API 接口开放

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License

## 联系我们

- Issue: [GitHub Issues]
- Email: [联系邮箱]

---

⭐ 如果这个项目对您有帮助，请给个 Star！
