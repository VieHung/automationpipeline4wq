from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request


# Cho phép import ace_lib và helpful_functions từ thư mục ACE2023_v3
BASE_DIR = Path(__file__).resolve().parent
ACE_DIR = BASE_DIR / "ACE2023_v3"
if str(ACE_DIR) not in sys.path:
    sys.path.append(str(ACE_DIR))

from ace_lib import (  # type: ignore
    DEFAULT_CONFIG,
    check_session_timeout,
    generate_alpha,
    get_alpha_yearly_stats,
    get_prod_corr,
    get_self_corr,
    get_simulation_result_json,
    simulate_alpha_list,
    start_session,
)

app = FastAPI(title="n8n WorldQuant Alpha Simulator")


SESSION_LOCK = threading.Lock()
SESSION = None

JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
WORKERS_STARTED = False


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _reset_session():
    global SESSION
    with SESSION_LOCK:
        SESSION = start_session()
    return SESSION


def _to_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _norm_key(k: str) -> str:
    # Chuẩn hoá key để bắt nhiều biến thể từ n8n/sheet
    return "".join(ch for ch in str(k).strip().lower() if ch.isalnum())


def _pick_value(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    if not isinstance(row, dict):
        return default

    direct_map = {str(k): v for k, v in row.items()}
    for key in keys:
        if key in direct_map:
            return direct_map[key]

    norm_map = {_norm_key(str(k)): v for k, v in row.items()}
    for key in keys:
        nk = _norm_key(key)
        if nk in norm_map:
            return norm_map[nk]

    return default


def _extract_sim_params(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chỉ lấy các trường cần để simulate, bỏ qua toàn bộ cột dư từ n8n/sheet.
    """

    alpha_expr = _pick_value(row, ["Alpha", "alpha", "regular", "expression"], "")
    universe = _pick_value(row, ["Universe", "universe"], "TOP3000")
    neutralization = _pick_value(
        row,
        ["Neutralization", "neutralization", "neutralisation"],
        "SUBINDUSTRY",
    )
    delay = _to_int(_pick_value(row, ["Delay", "delay"], 1), 1)
    decay = _to_int(_pick_value(row, ["Decay", "decay"], 4), 4)
    truncation = _to_float(_pick_value(row, ["Truncation", "truncation"], 0.08), 0.08)
    nan_handling = _pick_value(
        row,
        ["nan_handling", "nanHandling", "nan handling", "Nan Handling"],
        "OFF",
    )

    return {
        "alpha": str(alpha_expr).strip(),
        "universe": str(universe).strip(),
        "neutralization": str(neutralization).strip(),
        "delay": delay,
        "decay": decay,
        "truncation": truncation,
        "nan_handling": str(nan_handling).strip().upper() or "OFF",
    }


def _get_session():
    global SESSION
    with SESSION_LOCK:
        if SESSION is None:
            SESSION = start_session()
        else:
            expiry = check_session_timeout(SESSION)
            if expiry < 1000:
                SESSION = start_session()
    return SESSION


def _records_from_df(df: Any) -> List[Dict[str, Any]]:
    if df is None:
        return []
    try:
        if hasattr(df, "empty") and df.empty:
            return []
        return df.to_dict(orient="records")
    except Exception:
        return []


def _max_from_df(df: Any, col: str) -> Optional[float]:
    try:
        if df is None or getattr(df, "empty", True):
            return None
        if col not in df.columns:
            return None
        value = df[col].max()
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_n8n_payload(payload: Any) -> List[Dict[str, Any]]:
    """
    Hỗ trợ nhiều dạng input từ n8n:
    - 1 dict
    - list[dict]
    - {"items": [...]} hoặc {"data": [...]} hoặc {"records": [...]}.
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("items", "data", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]

    return []


def _json_safe(obj: Any) -> Any:
    """
    Convert object to JSON-safe values (handle NaN/Inf, numpy scalars, datetime).
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # numpy/pandas scalar
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass

    return obj


@app.post("/simulate-alpha")
async def simulate_alpha_endpoint(request: Request):
    raw_body = await request.body()
    if not raw_body:
        return {"status": "error", "message": "Body rỗng"}

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return {
            "status": "error",
            "message": "Body không phải JSON hợp lệ",
            "raw_preview": raw_body.decode("utf-8", errors="replace")[:1000],
        }

    rows = _parse_n8n_payload(payload)
    if not rows:
        return {
            "status": "error",
            "message": "Không đọc được item nào từ payload",
            "payload_type": type(payload).__name__,
        }

    prepared: List[Dict[str, Any]] = []
    simulate_data_list: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        params = _extract_sim_params(row)
        if not params["alpha"]:
            prepared.append(
                {
                    "index": i,
                    "status": "error",
                    "message": "Thiếu trường Alpha/alpha",
                    "input_preview": row,
                }
            )
            continue

        simulate_data = generate_alpha(
            regular=params["alpha"],
            universe=params["universe"],
            neutralization=params["neutralization"],
            delay=params["delay"],
            decay=params["decay"],
            truncation=params["truncation"],
            nan_handling=params["nan_handling"],
        )

        prepared.append(
            {
                "index": i,
                "status": "ready",
                "row_number": _pick_value(row, ["row_number", "rowNumber", "row"], None),
                "params": params,
                "input": row,
            }
        )
        simulate_data_list.append(simulate_data)

    ready_items = [x for x in prepared if x["status"] == "ready"]
    error_items = [x for x in prepared if x["status"] == "error"]

    if not ready_items:
        return {
            "status": "error",
            "message": "Không có alpha hợp lệ để simulate",
            "errors": error_items,
        }

    # Có thể set qua env để thay đổi số luồng khi deploy
    concurrent = _to_int(os.getenv("WQ_CONCURRENT_SIMULATIONS", "3"), 3)
    concurrent = max(1, min(concurrent, 8))

    simulation_config = {
        **DEFAULT_CONFIG,
        "get_pnl": False,
        "get_stats": True,
        "check_submission": False,
        # NOTE: ace_lib dùng DataFrame.append cho 2 check này (không tương thích pandas mới)
        # Mình sẽ tự gọi get_self_corr/get_prod_corr ở phía dưới để lấy đầy đủ dữ liệu.
        "check_self_corr": False,
        "check_prod_corr": False,
    }

    try:
        session = _get_session()
        results = simulate_alpha_list(
            session,
            simulate_data_list,
            limit_of_concurrent_simulations=concurrent,
            simulation_config=simulation_config,
            pre_request_delay=0.3,
            pre_request_jitter=0.5,
        )
    except Exception as e:
        # Fallback phòng trường hợp môi trường vẫn ném lỗi append từ ace_lib
        if "DataFrame" in str(e) and "append" in str(e):
            try:
                fallback_config = {
                    **simulation_config,
                    "check_self_corr": False,
                    "check_prod_corr": False,
                }
                results = simulate_alpha_list(
                    session,
                    simulate_data_list,
                    limit_of_concurrent_simulations=concurrent,
                    simulation_config=fallback_config,
                    pre_request_delay=0.3,
                    pre_request_jitter=0.5,
                )
            except Exception as e2:
                return {
                    "status": "error",
                    "message": f"Lỗi khi simulate (fallback cũng lỗi): {e2}",
                    "ready_count": len(ready_items),
                    "error_count": len(error_items),
                }
        else:
            return {
                "status": "error",
                "message": f"Lỗi khi simulate: {e}",
                "ready_count": len(ready_items),
                "error_count": len(error_items),
            }

    output_items: List[Dict[str, Any]] = []

    for meta, res in zip(ready_items, results):
        alpha_id = res.get("alpha_id")
        params = meta["params"]

        if alpha_id is None:
            output_items.append(
                {
                    "index": meta["index"],
                    "row_number": meta["row_number"],
                    "status": "failed",
                    "params": params,
                    "message": "Simulation không thành công hoặc bị platform từ chối",
                }
            )
            continue

        # Lấy full json alpha để trả được tối đa thông tin đánh giá
        alpha_json = get_simulation_result_json(session, alpha_id)
        is_data = alpha_json.get("is", {}) if isinstance(alpha_json, dict) else {}

        checks = _records_from_df(res.get("is_tests"))
        yearly_stats = _records_from_df(res.get("stats")) or _records_from_df(
            get_alpha_yearly_stats(session, alpha_id)
        )

        self_corr_df = get_self_corr(session, alpha_id)
        prod_corr_df = get_prod_corr(session, alpha_id)

        self_corr_max = _max_from_df(self_corr_df, "correlation")
        prod_corr_max = _max_from_df(prod_corr_df, "max")

        output_items.append(
            {
                "index": meta["index"],
                "row_number": meta["row_number"],
                "status": "success",
                "alpha_id": alpha_id,
                "alpha_url": f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                "params": params,
                # KPI chính
                "fitness": is_data.get("fitness"),
                "sharpe": is_data.get("sharpe"),
                "turnover": is_data.get("turnover"),
                "returns": is_data.get("returns"),
                "drawdown": is_data.get("drawdown"),
                "margin": is_data.get("margin"),
                # Full dữ liệu IS + checks
                "is": {k: v for k, v in is_data.items() if k != "checks"},
                "checks": checks,
                # Correlation chi tiết
                "self_corr_max": self_corr_max,
                "prod_corr_max": prod_corr_max,
                "self_corr_records": _records_from_df(self_corr_df),
                "prod_corr_records": _records_from_df(prod_corr_df),
                # Yearly stats nếu có
                "yearly_stats": yearly_stats,
                # Trả kèm raw alpha json để không bỏ sót field nào
                "alpha_full": alpha_json,
            }
        )

    return _json_safe({
        "status": "success",
        "message": "Đã xử lý simulate alpha từ n8n",
        "input_count": len(rows),
        "ready_count": len(ready_items),
        "error_count": len(error_items),
        "concurrent_simulations": concurrent,
        "errors": error_items,
        "results": output_items,
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
