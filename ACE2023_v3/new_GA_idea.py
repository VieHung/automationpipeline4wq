import importlib
import warm_sart_gp_ace as ws
importlib.reload(ws)

WarmStartGP_ACE         = ws.WarmStartGP_ACE
ParallelWarmStartGP_ACE = ws.ParallelWarmStartGP_ACE
BatchAceLibFitness      = ws.BatchAceLibFitness
build_catalog_from_csv  = ws.build_catalog_from_csv
AlphaParser             = ws.AlphaParser
DataCatalog             = ws.DataCatalog

from ace_lib import (
    start_session, get_self_corr,
    generate_alpha, simulate_single_alpha, get_simulation_result_json
)
import types
import pandas as pd
import re

s = start_session()

# ════════════════════════════════════════════════════════════════════════════
# FIX 2 & 3: Build catalog đúng kiểu từ CSV — nhóm theo (dataset_id + type)
# ════════════════════════════════════════════════════════════════════════════
def build_typed_catalog(csv_path="all_datasets_fields_full.csv"):
    """
    Nhóm biến theo (dataset_id + type) thay vì chỉ dataset_id.
    Đảm bảo mutation chỉ thay biến cùng kiểu dữ liệu.
    """
    catalog = DataCatalog()
    try:
        df = pd.read_csv(csv_path)
        print("✅ CSV columns:", df.columns.tolist())

        groups = {}
        
        # Nhóm theo dataset_id + type nếu có cột type
        group_col = "type" if "type" in df.columns else None

        for dataset_id, grp_df in df.groupby("dataset_id"):
            if group_col and group_col in grp_df.columns:
                # Nhóm nhỏ hơn theo type bên trong dataset
                for type_val, type_df in grp_df.groupby(group_col):
                    fields = type_df["field_id"].dropna().astype(str).drop_duplicates().tolist()
                    if len(fields) > 1:
                        key = f"{dataset_id}__{type_val}"
                        groups[key] = fields
                        print(f"   {key}: {len(fields)} fields")
            else:
                fields = grp_df["field_id"].dropna().astype(str).drop_duplicates().tolist()
                if len(fields) > 1:
                    groups[str(dataset_id)] = fields
                    print(f"   {dataset_id}: {len(fields)} fields")

        # Chỉ thêm biến đã xác nhận tồn tại trên Brain TOP3000/USA
        groups["price_confirmed"] = ["open", "high", "low", "close", "vwap"]
        groups["volume_confirmed"] = ["volume", "adv20"]
        groups["returns_confirmed"] = ["returns"]

        catalog.groups = groups
        total = sum(len(v) for v in groups.values())
        print(f"\n✅ Catalog: {len(groups)} nhóm | {total} biến")

    except FileNotFoundError:
        print(f"⚠ Không tìm thấy {csv_path}")
    return catalog

catalog = build_typed_catalog("all_datasets_fields_full.csv")

# ════════════════════════════════════════════════════════════════════════════
# FIX 1A: Bỏ min_fitness filter, dùng sharpe trực tiếp + self-corr penalty
# ════════════════════════════════════════════════════════════════════════════
fitness = BatchAceLibFitness(
    s                      = s,
    fitness_key            = "sharpe",
    cache_path             = "wsgp_cache.pkl",
    concurrent_simulations = 2,
    pre_request_delay      = 0.4,
    pre_request_jitter     = 0.6,
    region         = "USA",
    universe       = "TOP3000",
    neutralization = "SUBINDUSTRY",
    delay          = 1,
    decay          = 0,
    truncation     = 0.08,
    nan_handling   = "ON",
    unit_handling  = "VERIFY",
)

# Monkey-patch: chỉ self-corr penalty, BỎ min_fitness filter
def _score_with_corr_penalty(self, result):
    alpha_id = result.get("alpha_id")
    if alpha_id is None:
        return 0.0
    
    alpha_json = get_simulation_result_json(self.s, alpha_id)
    is_data    = alpha_json.get("is", {})
    sharpe     = float(is_data.get("sharpe", 0.0) or 0.0)
    
    if sharpe <= 0:
        return 0.0
    
    # Self-corr penalty chỉ khi sharpe > 0
    try:
        self_corr_df = get_self_corr(self.s, alpha_id)
        if not self_corr_df.empty:
            max_corr = float(self_corr_df["correlation"].max())
            if max_corr > 0.7:
                return 0.0
            if max_corr > 0.5:
                sharpe *= (1 - max_corr)
    except Exception as e:
        pass  # self-corr fail → không penalize
    
    return sharpe

fitness._score_from_result = types.MethodType(_score_with_corr_penalty, fitness)

# ════════════════════════════════════════════════════════════════════════════
# FIX 1B: Kiểm tra alphainit trước khi chạy GP
# ════════════════════════════════════════════════════════════════════════════
def check_alphainit(alpha_list, s_session, sim_kwargs):
    """Verify alphainit có sharpe > 0 trên Brain trước khi cho vào GP."""
    confirmed = []
    for i, expr in enumerate(alpha_list):
        try:
            sim = generate_alpha(regular=expr, **sim_kwargs)
            res = simulate_single_alpha(s_session, sim)
            aid = res.get("alpha_id")
            if not aid:
                print(f"  ❌ [{i}] Simulation fail: {expr[:60]}...")
                continue
            
            j = get_simulation_result_json(s_session, aid)
            sharpe = j["is"].get("sharpe", 0)
            brain_fitness = j["is"].get("fitness", 0)
            
            sharpe_val = float(sharpe) if sharpe else 0.0
            fitness_val = float(brain_fitness) if brain_fitness else 0.0
            
            print(f"  [{i}] sharpe={sharpe_val:.4f} | fitness={fitness_val:.4f} | {expr[:60]}...")
            
            if sharpe_val > 0:
                confirmed.append(expr)
                print(f"       ✅ Hợp lệ — thêm vào GP")
            else:
                confirmed.append(expr)
                print(f"       ⚠ Sharpe = 0 — vẫn thêm nhưng GP khó tối ưu")
        
        except Exception as e:
            print(f"  ❌ [{i}] Exception: {e} | {expr[:60]}...")
    
    return confirmed

SIM_KWARGS = dict(
    region="USA", universe="TOP3000",
    neutralization="SUBINDUSTRY", delay=1,
    decay=0, truncation=0.08,
    nan_handling="ON", unit_handling="VERIFY",
)

# ════════════════════════════════════════════════════════════════════════════
# Alpha list
# ════════════════════════════════════════════════════════════════════════════
ALPHA_LIST = [
    "- (days_from_last_change(ts_arg_min(ts_delta(operating_income /operating_expense, 150), 100)))-(rank(ts_sum(kth_element(vec_avg(fnd6_newqeventv110_ivstq), 20, k=1)/ts_arg_min(return_assets, 20),20)) * (ts_mean(close-ts_delay(close, 1), 10)))",

    # Alpha 3: OK
    "add(-ts_mean(days_from_last_change(anl4_netprofit_median / fnd6_newa2v1300_oiadp), 252), group_rank(ts_arg_min(sign(fnd6_newa1v1300_gp),63),sector)+rank(sales/assets))-(rank(ts_arg_min(ts_corr(group_backfill(fnd6_newqv1300_mibnq,subindustry,120),fnd6_newa1v1300_epsfi,240),252)))",
]

# ════════════════════════════════════════════════════════════════════════════
# Validate + Check alphainit
# ════════════════════════════════════════════════════════════════════════════
def split_alpha_statements(expr: str):
    if not expr or not expr.strip():
        return []
    parts, buf, depth = [], [], 0
    for ch in expr:
        if ch == "(": depth += 1
        elif ch == ")": depth = max(0, depth - 1)
        if ch == ";" and depth == 0:
            part = "".join(buf).strip()
            if part: parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail: parts.append(tail)
    return parts

def validate_alpha_list(alpha_list):
    valid, invalid = [], []
    parser = AlphaParser()
    for i, raw in enumerate(alpha_list):
        for j, expr in enumerate(split_alpha_statements(raw)):
            # Chỉ chặn phép gán kiểu: var = expr
            # Không chặn keyword args trong hàm, ví dụ: kth_element(..., k=1)
            if re.match(r"^\s*[A-Za-z_]\w*\s*=", expr):
                invalid.append((i, expr, "Có phép gán biến"))
                print(f"  ❌ [{i}.{j}] Có gán: {expr[:70]}...")
                continue
            try:
                tree = parser.parse(expr)
                leaves = [n.name for _, n in tree.leaves()]
                valid.append(expr)
                print(f"  ✅ [{i}.{j}] Parse OK | {len(leaves)} leaves | {expr[:70]}...")
            except Exception as e:
                invalid.append((i, expr, str(e)))
                print(f"  ❌ [{i}.{j}] Parse error: {e} | {expr[:70]}...")
    print(f"\n  {len(valid)} hợp lệ / {len(invalid)} lỗi\n")
    return valid, invalid

print("🔍 Validate alpha list...")
valid_list, invalid_list = validate_alpha_list(ALPHA_LIST)

print("🔍 Kiểm tra alphainit trên Brain...")
confirmed_list = check_alphainit(valid_list, s, SIM_KWARGS)

# ════════════════════════════════════════════════════════════════════════════
# Chạy GP
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
if confirmed_list:
    print(f"Chạy ParallelWarmStartGP_ACE với {len(confirmed_list)} alphainit")
    pgp = ParallelWarmStartGP_ACE(
        alphainit_exprs = confirmed_list,
        fitness_fn      = fitness,
        catalog         = catalog,
        population_size = 10,
        generations     = 15,
    )
    results = pgp.run()
    print(f"\n{'='*68}")
    print("Kết quả cuối:")
    for rank, (alpha, score) in enumerate(results, 1):
        print(f"  #{rank} sharpe={score:.4f} | {alpha.to_expr()[:80]}")
else:
    print("❌ Không có alphainit nào hợp lệ")