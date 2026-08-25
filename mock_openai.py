"""Mock OpenAI-compatible streaming endpoint for testing Moa."""
import json
import time
from flask import Flask, Response, request

app = Flask(__name__)

@app.post("/v1/chat/completions")
def completions():
    body = request.get_json(force=True)
    stream = body.get("stream", False)
    model = body.get("model", "mock")
    tag = model  # use model name as tag in response

    # 根据模型名生成不同风格的回答,便于区分
    texts = {
        "mock-a": f"我是模型 A,针对问题「{body['messages'][0]['content']}」,我的回答是:第一,这个问题可以从多个角度分析。第二,我建议综合考虑各方面因素。第三,最终结论需要权衡利弊。",
        "mock-b": f"我是模型 B,针对问题「{body['messages'][0]['content']}」,我的看法是:首先,需要明确问题的边界。其次,数据支撑很重要。最后,实践是检验真理的唯一标准。",
        "mock-main": f"针对问题「{body['messages'][0]['content']}」,我的综合答案如下:\n\n1. 先界定问题的本质与边界(吸收自模型 B 的思路);\n2. 从多个维度权衡各种可行路径(吸收自模型 A 的思路);\n3. 给出明确的行动建议:小步实践、快速反馈、持续迭代。\n\n以上并非简单拼接,而是我基于两个视角独立形成的结论。",
    }
    text = texts.get(model, f"模型 {model} 的通用回答:这是一个值得深入探讨的问题,需要从多个维度分析。")

    if not stream:
        return {"choices": [{"message": {"content": text}}]}

    def gen():
        # 模拟流式:每次吐 2-4 个字符,间隔 30ms
        i = 0
        while i < len(text):
            chunk_size = min(3, len(text) - i)
            piece = text[i:i+chunk_size]
            i += chunk_size
            data = {"choices": [{"delta": {"content": piece}}]}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            time.sleep(0.03)
        yield "data: [DONE]\n\n"

    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7820, debug=False, threaded=True)
