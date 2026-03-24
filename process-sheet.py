from fastapi import FastAPI, Request
import uvicorn
import json
from typing import Any, List

app = FastAPI()

def collect_key_paths(data: Any, prefix: str = "") -> List[str]:
    paths = []
    if isinstance(data, dict):
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            paths.append(key)
            paths.extend(collect_key_paths(v, key))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            key = f"{prefix}[{i}]"
            paths.append(key)
            paths.extend(collect_key_paths(item, key))
    return paths

@app.post("/process-sheet")
async def process_sheet(request: Request):
    raw_body = await request.body()
    headers = dict(request.headers)
    content_type = headers.get("content-type", "")

    print("\n--- Headers ---")
    print(headers)
    print("\n--- Raw Body ---")
    print(raw_body)

    if not raw_body:
        return {
            "status": "error",
            "message": "Body rỗng",
            "content_type": content_type,
        }

    payload: Any = None
    parse_mode = "unknown"

    try:
        # Ưu tiên parse JSON
        payload = json.loads(raw_body.decode("utf-8"))
        parse_mode = "json"
    except Exception:
        try:
            # Nếu n8n gửi form-data / x-www-form-urlencoded
            form = await request.form()
            payload = dict(form)
            parse_mode = "form"
        except Exception:
            # Fallback text
            payload = raw_body.decode("utf-8", errors="replace")
            parse_mode = "text"

    print("\n--- Parsed Payload ---")
    print(payload)

    key_paths = collect_key_paths(payload) if isinstance(payload, (dict, list)) else []

    return {
        "status": "success",
        "parse_mode": parse_mode,
        "content_type": content_type,
        "payload_type": type(payload).__name__,
        "key_paths": key_paths[:200],  # giới hạn để response không quá lớn
        "payload_preview": payload if isinstance(payload, (dict, list)) else str(payload)[:1000],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)