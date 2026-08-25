# Moa - 多模型协作问答工具

Moa 是一个网页版的多模型协作问答工具，将一个问题分发给多个 AI 模型并发处理，各子模型的回答流式可见，最后由主模型归纳总结形成自己的答案。
> 待办
> 增加支持思考和联网能力
> 增加返回md格式支持

## 特性

- **多模型并发**：同时调用多个 AI 模型，流式输出实时可见
- **主模型汇总**：主模型汲取各子模型视角，形成独立完整答案
- **自适应布局**：模型多时自动换行，主模型独占一行居中显示
- **一键复制**：主模型汇总结果支持一键复制
- **内置预设**：支持智谱、百炼、DeepSeek、Kimi、MiniMax、OpenRouter、硅基流动、Ollama 等主流厂商
- **自定义接入**：支持任何 OpenAI 兼容协议的模型
- **失败重试**：自动重试机制（0.3s/0.6s 间隔）
- **主模型互斥**：选定为主模型后自动从子模型列表排除
- **自定义提示词**：主模型汇总提示词可配置

## 快速开始

### 1. 安装依赖

```bash
pip install flask requests
```

### 2. 配置模型

复制示例配置文件：

```bash
cp config.example.json config.json
```

编辑 `config.json`，填入你的 API Key：

```json
{
  "providers": {
    "dashscope": {
      "enabled": true,
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key": "你的API密钥",
      "model": "qwen-plus",
      "label": "阿里云百炼"
    }
  },
  "main": "dashscope",
  "summary_prompt": "结合各模型的返回，提供的视角和思路，整合后给自己的答案"
}
```

或通过 Web 界面配置：启动服务后点击「⚙ 设置」按钮。

### 3. 启动服务

```bash
python app.py
```

访问 http://127.0.0.1:7819

## 配置说明

### 内置预设

- **智谱 GLM**：`https://open.bigmodel.cn/api/paas/v4`
- **阿里云百炼**：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- **DeepSeek**：`https://api.deepseek.com/v1`
- **月之暗面 Kimi**：`https://api.moonshot.cn/v1`
- **MiniMax**：`https://api.minimax.chat/v1`
- **OpenRouter**：`https://openrouter.ai/api/v1`
- **硅基流动**：`https://api.siliconflow.cn/v1`
- **Ollama 本地**：`http://localhost:11434/v1`

### 自定义 Provider

支持任何 OpenAI 兼容协议的模型，只需填写：
- Base URL
- API Key
- 模型名
- 显示名（可选）

### 主模型汇总提示词

默认提示词：

```
你是主控模型。不要逐个点评或罗列这些回答,而是:
1. 汲取各回答中有价值的视角、论据与洞见;
2. 剔除错误、片面或重复的内容,识别各模型的分歧点并独立判断谁更可信;
3. 在此基础上形成你自己完整、连贯的最终答案——它应当比任何单一子模型的回答都更全面、更准确,直接回应问题本身。
输出格式:直接给出你的最终答案(可分点/分层组织),如有必要可在末尾用一小段简述你采纳与舍弃了哪些观点及原因。用中文回答。
```

可在设置中自定义。

## 项目结构

```
Moa/
├── app.py              # Flask 后端
├── config.json         # 本地配置（包含真实 API Key，不提交）
├── config.example.json # 示例配置（API Key 已脱敏）
├── static/
│   └── index.html      # 前端界面
├── mock_openai.py      # Mock 服务（测试用）
└── README.md
```

## 技术栈

- **后端**：Flask + requests
- **前端**：原生 HTML/CSS/JavaScript
- **协议**：OpenAI Chat Completions API（流式）

## 注意事项

- `config.json` 包含真实 API Key，已在 `.gitignore` 中排除
- 提交前请确保使用 `config.example.json` 作为模板
- 所有模型调用均走 HTTPS，API Key 仅在本地存储

## License

MIT
