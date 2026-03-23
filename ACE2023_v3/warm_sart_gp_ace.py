"""
warm_start_gp_full.py
────────────────────────────────────────────────────────────────────────────
Warm Start Genetic Programming — All-in-one
Đặt file này vào cùng thư mục với ace_lib.py và helpful_functions.py

    ACE2023_v3/
        ace_lib.py
        helpful_functions.py
        all_datasets_fields.csv
        warm_start_gp_full.py   ← file này

Dùng trong notebook:
    from warm_start_gp_full import WarmStartGP_ACE, BatchAceLibFitness
    from warm_start_gp_full import build_catalog_from_csv
    from ace_lib import start_session
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import copy
import time
import pickle
import random
import hashlib
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

# ── ace_lib (có sẵn trong project) ────────────────────────────────────────
from ace_lib import (
    start_session,
    generate_alpha,
    simulate_single_alpha,
    get_simulation_result_json,
    get_self_corr,
    simulate_alpha_list,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. NODE
# ══════════════════════════════════════════════════════════════════════════════

class Node:
    """
    Nút trong cây biểu thức alpha.
    node_type: "op" | "var" | "num"
    """

    def __init__(
        self,
        name: str,
        children: Optional[List["Node"]] = None,
        node_type: str = "op",
    ):
        self.name      = name
        self.children  = children if children is not None else []
        self.node_type = node_type

    @property
    def is_leaf(self) -> bool:
        return self.node_type in ("var", "num")

    @property
    def is_var(self) -> bool:
        return self.node_type == "var"

    @property
    def is_num(self) -> bool:
        return self.node_type == "num"

    def clone(self) -> "Node":
        return copy.deepcopy(self)

    def to_expr(self) -> str:
        if self.is_leaf:
            return self.name

        # Render arithmetic ops as FASTEXPR infix syntax
        if self.name == "neg" and len(self.children) == 1:
            return f"-({self.children[0].to_expr()})"
        if self.name in ("add", "sub", "mul", "div") and len(self.children) == 2:
            op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            left = self.children[0].to_expr()
            right = self.children[1].to_expr()
            return f"({left} {op_map[self.name]} {right})"

        args = ", ".join(c.to_expr() for c in self.children)
        return f"{self.name}({args})"

    def to_skeleton(self) -> str:
        """Chuỗi cấu trúc — leaf → '?', dùng để so sánh cấu trúc."""
        if self.is_leaf:
            return "?"
        args = ", ".join(c.to_skeleton() for c in self.children)
        return f"{self.name}({args})"

    def leaves(
        self, path: Tuple[int, ...] = ()
    ) -> List[Tuple[Tuple[int, ...], "Node"]]:
        if self.is_leaf:
            return [(path, self)]
        result = []
        for i, child in enumerate(self.children):
            result.extend(child.leaves(path + (i,)))
        return result

    def get(self, path: Tuple[int, ...]) -> "Node":
        node = self
        for idx in path:
            node = node.children[idx]
        return node

    def set(self, path: Tuple[int, ...], new_node: "Node") -> None:
        if not path:
            raise ValueError("Không thể thay thế root")
        node = self
        for idx in path[:-1]:
            node = node.children[idx]
        node.children[path[-1]] = new_node

    def __hash__(self) -> int:
        return hash(self.to_expr())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.to_expr() == other.to_expr()

    def __repr__(self) -> str:
        return self.to_expr()


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARSER
# ══════════════════════════════════════════════════════════════════════════════

class AlphaParser:
    """Recursive-descent parser: chuỗi alpha → cây Node."""

    _TOKEN_RE = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*"
        r"|\d+\.?\d*"
        r"|[(),+\-*/=]"
    )

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._pos: int = 0

    def parse(self, expr: str) -> Node:
        self._tokens = self._TOKEN_RE.findall(expr.replace(" ", ""))
        self._pos    = 0
        node = self._parse_expr()
        if self._pos != len(self._tokens):
            ctx = self._tokens[max(0, self._pos - 2): self._pos + 3]
            raise SyntaxError(f"Unexpected token sequence | context: {ctx}")
        return node

    def _cur(self) -> str:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ""

    def _consume(self, expected: Optional[str] = None) -> str:
        tok = self._cur()
        if expected is not None and tok != expected:
            ctx = self._tokens[max(0, self._pos - 2): self._pos + 3]
            raise SyntaxError(f"Expected '{expected}', got '{tok}' | context: {ctx}")
        self._pos += 1
        return tok

    def _peek(self, offset: int = 1) -> str:
        idx = self._pos + offset
        return self._tokens[idx] if idx < len(self._tokens) else ""

    def _parse_expr(self) -> Node:
        return self._parse_add_sub()

    def _parse_add_sub(self) -> Node:
        node = self._parse_mul_div()
        while self._cur() in ("+", "-"):
            op_tok = self._consume()
            right = self._parse_mul_div()
            node = Node("add" if op_tok == "+" else "sub", [node, right], node_type="op")
        return node

    def _parse_mul_div(self) -> Node:
        node = self._parse_unary()
        while self._cur() in ("*", "/"):
            op_tok = self._consume()
            right = self._parse_unary()
            node = Node("mul" if op_tok == "*" else "div", [node, right], node_type="op")
        return node

    def _parse_unary(self) -> Node:
        if self._cur() == "-":
            self._consume("-")
            return Node("neg", [self._parse_unary()], node_type="op")
        return self._parse_primary()

    def _parse_primary(self) -> Node:
        tok = self._cur()

        if tok == "(":
            self._consume("(")
            node = self._parse_expr()
            self._consume(")")
            return node

        if not tok:
            raise SyntaxError("Unexpected end of expression")

        try:
            float(tok)
            self._consume()
            return Node(tok, node_type="num")
        except ValueError:
            pass

        name = self._consume()
        if self._cur() == "(":
            self._consume("(")
            children: List[Node] = []
            if self._cur() != ")":
                while True:
                    # Hỗ trợ keyword args trong hàm, ví dụ: kth_element(..., k=1)
                    # Parser nội bộ không lưu tên keyword, chỉ lấy giá trị biểu thức.
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self._cur()) and self._peek() == "=":
                        self._consume()     # keyword name
                        self._consume("=")
                    children.append(self._parse_expr())
                    if self._cur() == ",":
                        self._consume(",")
                        continue
                    break
            self._consume(")")
            return Node(name, children, node_type="op")

        return Node(name, node_type="var")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA CATALOG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataCatalog:
    """
    Kho biến theo nhóm — mutation chỉ hoán đổi trong cùng nhóm.
    Mặc định dùng biến WQ Brain chuẩn.
    Ghi đè groups bằng build_catalog_from_csv() để dùng biến thực.
    """

    groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "price"   : ["open", "high", "low", "close", "vwap", "twap"],
        "volume"  : ["volume", "adv5", "adv10", "adv20", "amount", "turnover"],
        "returns" : ["returns", "close_to_open", "open_to_close"],
        "fnd2"    : [
            "fnd2_itxreclstatelocalitxes", "fnd2_revtq", "fnd2_niq",
            "fnd2_oiadpq", "fnd2_atq", "fnd2_ltq", "fnd2_ceqq",
            "fnd2_capxy", "fnd2_oancfy",
        ],
        "fnd6"    : [
            "fnd6_adesinda_curcd", "fnd6_newqeventv110_piq",
            "fnd6_newqv1300_rectoq", "fnd6_eps", "fnd6_bvps",
            "fnd6_sales", "fnd6_ebitda", "fnd6_fcf", "fnd6_roe", "fnd6_roa",
        ],
        "anl4"    : [
            "anl4_netprofita_mean", "anl4_ebit_number",
            "anl4_fsdtlestmtbscqv104_item", "anl4_eps1_mean",
            "anl4_rev1_mean", "anl4_tprice_mean", "anl4_rec_mean",
            "anl4_nup", "anl4_ndown",
        ],
        "rp"      : [
            "rp_nip_inverstor", "rp_css", "rp_mss",
            "rp_en", "rp_er", "rp_nsr",
        ],
        "operating": [
            "operating_income", "operating_expense",
            "net_income", "gross_profit", "ebit", "ebitda",
        ],
    })

    def _reverse_map(self) -> Dict[str, str]:
        rev: Dict[str, str] = {}
        for gname, vars_ in self.groups.items():
            for v in vars_:
                rev[v] = gname
        return rev

    def get_substitutes(self, var_name: str) -> List[str]:
        rev   = self._reverse_map()
        group = rev.get(var_name)
        if group is None:
            return [var_name]
        return [v for v in self.groups[group] if v != var_name] or [var_name]

    def random_substitute(self, var_name: str) -> str:
        return random.choice(self.get_substitutes(var_name))

    def perturb_period(self, val: str, pct: float = 0.2) -> str:
        try:
            n     = int(val)
            delta = max(1, int(n * pct))
            return str(max(2, n + random.randint(-delta, delta)))
        except ValueError:
            try:
                f = float(val)
                return str(round(f * random.uniform(1 - pct, 1 + pct), 4))
            except ValueError:
                return val


def build_catalog_from_csv(csv_path: str = "all_datasets_fields_full.csv") -> DataCatalog:
    """
    Đọc CSV fields và build DataCatalog từ biến thực của WQ Brain.

    Hỗ trợ 2 format:
    1) dataset-based: có cột dataset_id (hoặc _dataset_id) + id/field_id
    2) id-only: chỉ có cột id
    """
    import pandas as pd

    catalog = DataCatalog()
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Đọc {len(df)} fields từ {csv_path}")

        groups: Dict[str, List[str]] = {}

        field_col = None
        for col in ("field_id", "id"):
            if col in df.columns:
                field_col = col
                break

        if field_col is None:
            print(f"⚠ CSV thiếu cột field id (id/field_id) → dùng catalog mặc định")
            return catalog

        # Case 1: Có dataset id rõ ràng
        dataset_col = None
        for col in ("dataset_id", "_dataset_id"):
            if col in df.columns:
                dataset_col = col
                break

        if dataset_col is not None:
            for dataset_id, grp_df in df.groupby(dataset_col):
                fields = (
                    grp_df[field_col]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )
                if fields:
                    groups[str(dataset_id)] = fields
        else:
            # Case 2: id-only CSV → gom nhóm theo prefix (fnd6_, anl4_, ...),
            # phần còn lại vào nhóm wq_fields để mutation vẫn hoạt động.
            def infer_group(field_name: str) -> str:
                if field_name in {"open", "high", "low", "close", "vwap", "twap"}:
                    return "price"
                if field_name in {"volume", "adv5", "adv10", "adv20", "amount", "turnover"}:
                    return "volume"
                if field_name in {"returns", "close_to_open", "open_to_close"}:
                    return "returns"
                m = re.match(r"^([a-z]+\d+)_", field_name.lower())
                if m:
                    return m.group(1)
                return "wq_fields"

            for field_name in df[field_col].dropna().astype(str).drop_duplicates().tolist():
                g = infer_group(field_name)
                groups.setdefault(g, []).append(field_name)

        groups["price"]   = ["open", "high", "low", "close", "vwap", "twap"]
        groups["volume"]  = ["volume", "adv5", "adv10", "adv20", "amount", "turnover"]
        groups["returns"] = ["returns", "close_to_open", "open_to_close"]

        catalog.groups = groups
        total = sum(len(v) for v in groups.values())
        print(f"✅ Catalog: {len(groups)} nhóm | {total} biến")

    except FileNotFoundError:
        print(f"⚠ Không tìm thấy {csv_path} → dùng catalog mặc định")

    return catalog


# ══════════════════════════════════════════════════════════════════════════════
# 4. GENETIC OPERATORS
# ══════════════════════════════════════════════════════════════════════════════

class GeneticOperators:
    """Point mutation + Restricted crossover — cấu trúc cây không đổi."""

    def __init__(self, catalog: DataCatalog) -> None:
        self.catalog = catalog

    def point_mutate(self, tree: Node, num_points: int = 1) -> Node:
        result     = tree.clone()
        all_leaves = result.leaves()
        if not all_leaves:
            return result

        chosen = random.sample(all_leaves, min(num_points, len(all_leaves)))
        for path, leaf in chosen:
            if leaf.is_var:
                new_leaf = Node(self.catalog.random_substitute(leaf.name), node_type="var")
            elif leaf.is_num:
                new_leaf = Node(self.catalog.perturb_period(leaf.name), node_type="num")
            else:
                continue
            if path:
                result.set(path, new_leaf)
        return result

    def restricted_crossover(self, p1: Node, p2: Node) -> Tuple[Node, Node]:
        if p1.to_skeleton() != p2.to_skeleton():
            return p1.clone(), p2.clone()

        leaves = p1.leaves()
        if not leaves:
            return p1.clone(), p2.clone()

        path, _ = random.choice(leaves)
        if not path:
            return p1.clone(), p2.clone()

        c1, c2 = p1.clone(), p2.clone()
        c1.set(path, p2.get(path).clone())
        c2.set(path, p1.get(path).clone())
        return c1, c2

    def same_structure(self, a: Node, b: Node) -> bool:
        return a.to_skeleton() == b.to_skeleton()


# ══════════════════════════════════════════════════════════════════════════════
# 5. FITNESS — dùng ace_lib submit lên WQ Brain
# ══════════════════════════════════════════════════════════════════════════════

class BatchAceLibFitness:
    """
    Fitness function dùng WQ Brain IS Sharpe làm oracle.

    - Evaluate đơn: fitness(node)
    - Evaluate batch: fitness.evaluate_population(nodes)  ← dùng ThreadPool
    - Cache kết quả vào disk → không submit lại alpha đã chạy

    Params
    ------
    s                      : session từ start_session()
    fitness_key            : "sharpe" | "fitness" | "returns" | "drawdown"
    cache_path             : file .pkl lưu cache
    concurrent_simulations : số thread song song (2-3 để tránh rate limit)
    sim_settings           : kwargs cho generate_alpha()
    """

    VALID_KEYS = {"sharpe", "fitness", "returns", "drawdown", "turnover"}

    def __init__(
        self,
        s,
        fitness_key            : str   = "sharpe",
        cache_path             : Optional[str] = "wsgp_cache.pkl",
        concurrent_simulations : int   = 2,
        pre_request_delay      : float = 0.4,
        pre_request_jitter     : float = 0.6,
        min_fitness            : float = 1.0,
        self_corr_soft         : float = 0.5,
        self_corr_hard         : float = 0.7,
        **sim_settings,
    ) -> None:
        if fitness_key not in self.VALID_KEYS:
            raise ValueError(f"fitness_key phải là một trong {self.VALID_KEYS}")

        self.s            = s
        self.fitness_key  = fitness_key
        # Keep only kwargs supported by generate_alpha to avoid passing
        # strategy-specific params (e.g. min_fitness, self_corr_*) downstream.
        allowed = set(inspect.signature(generate_alpha).parameters.keys())
        self.sim_settings = {k: v for k, v in sim_settings.items() if k in allowed}
        self.concurrent   = concurrent_simulations
        self.delay        = pre_request_delay
        self.jitter       = pre_request_jitter
        self.min_fitness  = min_fitness
        self.self_corr_soft = self_corr_soft
        self.self_corr_hard = self_corr_hard

        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, float] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)
            print(f"📦 Cache loaded: {len(self._cache)} entries")

    # ── Evaluate đơn ──────────────────────────────────────────────────────
    def __call__(self, node: Node) -> float:
        key = self._hash(node)
        if key in self._cache:
            return self._cache[key]

        sim_data = generate_alpha(regular=node.to_expr(), **self.sim_settings)
        result   = simulate_single_alpha(
            self.s,
            sim_data,
            pre_request_delay=self.delay,
            pre_request_jitter=self.jitter,
        )
        score    = self._score_from_result(result)

        self._cache[key] = score
        self._save_cache()
        return score

    # ── Evaluate cả quần thể (batch) ──────────────────────────────────────
    def evaluate_population(self, nodes: List[Node]) -> List[float]:
        scores  = [None] * len(nodes)
        to_sim  = []

        for i, node in enumerate(nodes):
            key = self._hash(node)
            if key in self._cache:
                scores[i] = self._cache[key]
            else:
                to_sim.append((i, node))

        print(f"  💾 Cache: {len(nodes)-len(to_sim)}/{len(nodes)} hit | "
              f"Simulate: {len(to_sim)}")

        if to_sim:
            sim_data_list = [
                generate_alpha(regular=node.to_expr(), **self.sim_settings)
                for _, node in to_sim
            ]
            results = simulate_alpha_list(
                self.s, sim_data_list,
                limit_of_concurrent_simulations = self.concurrent,
                pre_request_delay               = self.delay,
                pre_request_jitter              = self.jitter,
            )
            for (orig_idx, node), res in zip(to_sim, results):
                score              = self._score_from_result(res)
                scores[orig_idx]   = score
                self._cache[self._hash(node)] = score
            self._save_cache()

        return scores  # type: ignore

    # ── Helpers ───────────────────────────────────────────────────────────
    def _score_from_result(self, result: dict) -> float:
        alpha_id = result.get("alpha_id")
        if alpha_id is None:
            return 0.0
        
        alpha_json = get_simulation_result_json(self.s, alpha_id)
        is_data    = alpha_json.get("is", {})

        sharpe  = float(is_data.get("sharpe",  0.0) or 0.0)
        fitness = float(is_data.get("fitness", 0.0) or 0.0)

        # Lọc fitness thấp
        if fitness < self.min_fitness:
            return 0.0

        # Penalize bằng self-correlation
        self_corr_df = get_self_corr(self.s, alpha_id)
        if not self_corr_df.empty and "correlation" in self_corr_df.columns:
            max_corr = float(self_corr_df["correlation"].max())

            # Corr quá cao: loại bỏ
            if max_corr > self.self_corr_hard:
                return 0.0

            # Corr trung bình cao: phạt điểm mềm
            if max_corr > self.self_corr_soft:
                sharpe *= max(0.0, 1 - max_corr)

        return max(sharpe, 0.0)

    @staticmethod
    def _hash(node: Node) -> str:
        return hashlib.md5(node.to_expr().encode()).hexdigest()

    def _save_cache(self) -> None:
        if self.cache_path:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)

    def cache_info(self) -> None:
        print(f"Cache: {len(self._cache)} entries | {self.cache_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. WARM START GP
# ══════════════════════════════════════════════════════════════════════════════

class WarmStartGP_ACE:
    """
    Warm Start Genetic Programming tích hợp WQ Brain.

    Khác GP truyền thống:
      1. Khởi tạo từ 1 alphainit đã biết hiệu quả (warm start)
      2. Thế hệ 1: chỉ point mutation → tạo quần thể đa dạng
      3. Thế hệ > 1: restricted crossover + point mutation
      4. Skeleton không bao giờ thay đổi
      5. Không cho cá thể trùng lặp vào quần thể
      6. Batch evaluate qua ThreadPool của ace_lib

    Params
    ------
    alphainit_expr  : chuỗi alpha khởi đầu (đã validate trên WQ Brain)
    fitness_fn      : BatchAceLibFitness
    catalog         : DataCatalog (từ build_catalog_from_csv)
    population_size : kích thước quần thể (khuyên 10-20 vì mỗi = 1 API call)
    generations     : số thế hệ
    crossover_rate  : xác suất crossover vs mutation
    tournament_size : kích thước tournament selection
    elitism         : số cá thể tốt nhất giữ lại mỗi thế hệ
    seed            : random seed
    verbose         : in log mỗi thế hệ
    """

    def __init__(
        self,
        alphainit_expr  : str,
        fitness_fn      : BatchAceLibFitness,
        catalog         : DataCatalog,
        population_size : int   = 10,
        generations     : int   = 15,
        crossover_rate  : float = 0.6,
        tournament_size : int   = 3,
        elitism         : int   = 1,
        seed            : int   = 42,
        verbose         : bool  = True,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)

        self.fitness_fn      = fitness_fn
        self.catalog         = catalog
        self.population_size = population_size
        self.generations     = generations
        self.crossover_rate  = crossover_rate
        self.tournament_size = tournament_size
        self.elitism         = elitism
        self.verbose         = verbose

        self.alphainit = AlphaParser().parse(alphainit_expr)
        self.skeleton  = self.alphainit.to_skeleton()
        self.ops       = GeneticOperators(catalog)

    # ── Evaluate (batch) ──────────────────────────────────────────────────
    def _evaluate(self, population: List[Node]) -> List[float]:
        return self.fitness_fn.evaluate_population(population)

    # ── Tournament selection ──────────────────────────────────────────────
    def _tournament(self, population: List[Node], scores: List[float]) -> Node:
        k   = min(self.tournament_size, len(population))
        idx = random.sample(range(len(population)), k)
        return population[max(idx, key=lambda i: scores[i])].clone()

    # ── Khởi tạo quần thể ────────────────────────────────────────────────
    def _init_population(self) -> List[Node]:
        population: List[Node] = [self.alphainit.clone()]
        seen: Set[str]         = {self.alphainit.to_expr()}
        max_tries = self.population_size * 50

        for _ in range(max_tries):
            if len(population) >= self.population_size:
                break
            candidate = self.ops.point_mutate(self.alphainit)
            key       = candidate.to_expr()
            if key not in seen:
                seen.add(key)
                population.append(candidate)

        if self.verbose:
            print(f"  Khởi tạo: {len(population)}/{self.population_size} cá thể")
        return population

    # ── Một thế hệ tiến hoá ──────────────────────────────────────────────
    def _evolve(self, population: List[Node], scores: List[float]) -> List[Node]:
        paired     = sorted(zip(scores, population), reverse=True, key=lambda x: x[0])
        sorted_pop = [ind for _, ind in paired]

        new_pop: List[Node] = [ind.clone() for ind in sorted_pop[:self.elitism]]
        seen:    Set[str]   = {ind.to_expr() for ind in new_pop}
        max_tries = self.population_size * 40

        for _ in range(max_tries):
            if len(new_pop) >= self.population_size:
                break

            if random.random() < self.crossover_rate:
                p1, p2 = self._tournament(population, scores), self._tournament(population, scores)
                if self.ops.same_structure(p1, p2):
                    candidates = list(self.ops.restricted_crossover(p1, p2))
                else:
                    candidates = [self.ops.point_mutate(p1)]
            else:
                candidates = [self.ops.point_mutate(self._tournament(population, scores))]

            for cand in candidates:
                if len(new_pop) >= self.population_size:
                    break
                key = cand.to_expr()
                if key not in seen:
                    seen.add(key)
                    new_pop.append(cand)

        return new_pop

    # ── Vòng lặp chính ────────────────────────────────────────────────────
    def run(self) -> Tuple[Node, float, List[float]]:
        if self.verbose:
            sep = "=" * 68
            print(f"\n{sep}")
            print("  WARM START GP — WorldQuant Brain")
            print(sep)
            print(f"  alphainit : {self.alphainit.to_expr()[:65]}...")
            print(f"  skeleton  : {self.skeleton[:65]}...")
            print(f"  pop_size  : {self.population_size}  |  generations : {self.generations}")
            print(f"  crossover : {self.crossover_rate:.0%}  |  tournament  : {self.tournament_size}")
            print(f"  elitism   : {self.elitism}")
            print(f"  fitness   : {self.fitness_fn.fitness_key}")
            print(sep + "\n")

        population = self._init_population()
        scores     = self._evaluate(population)
        best_idx   = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_alpha = population[best_idx].clone()
        history: List[float] = []

        for gen in range(self.generations):
            population = self._evolve(population, scores)
            scores     = self._evaluate(population)

            gen_best_idx   = int(np.argmax(scores))
            gen_best_score = scores[gen_best_idx]

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_alpha = population[gen_best_idx].clone()

            history.append(best_score)

            if self.verbose:
                print(
                    f"  Gen {gen+1:4d}/{self.generations} | "
                    f"Best = {best_score:.4f} | "
                    f"Avg  = {float(np.mean(scores)):.4f} | "
                    f"Pop  = {len(population)}"
                )

        if self.verbose:
            sep = "=" * 68
            print(f"\n{sep}")
            print(f"  XONG.  Best {self.fitness_fn.fitness_key} = {best_score:.4f}")
            print(f"  {best_alpha.to_expr()[:100]}")
            print(sep)

        return best_alpha, best_score, history


# ══════════════════════════════════════════════════════════════════════════════
# 7. PARALLEL — Nhiều alphainit chạy song song
# ══════════════════════════════════════════════════════════════════════════════

class ParallelWarmStartGP_ACE:
    """
    Chạy nhiều WarmStartGP_ACE từ nhiều alphainit khác nhau.
    Mỗi luồng tìm trong vùng cấu trúc riêng → giảm tương quan kết quả.
    """

    def __init__(
        self,
        alphainit_exprs : List[str],
        fitness_fn      : BatchAceLibFitness,
        catalog         : DataCatalog,
        **gp_kwargs,
    ) -> None:
        self.alphainit_exprs = alphainit_exprs
        self.fitness_fn      = fitness_fn
        self.catalog         = catalog
        self.gp_kwargs       = gp_kwargs

    def run(self) -> List[Tuple[Node, float]]:
        results: List[Tuple[Node, float]] = []
        n = len(self.alphainit_exprs)
        if n == 0:
            return results

        cache_lock = Lock()

        def run_one(i: int, expr: str) -> Tuple[Node, float]:
            print(f"\n{'#'*68}")
            print(f"  Luồng {i+1}/{n}: {expr[:60]}...")

            # Mỗi luồng dùng session riêng để ổn định request
            s_local = start_session()

            # Mỗi luồng dùng fitness riêng, không ghi cache trực tiếp xuống file
            # để tránh race condition. Gộp cache ở cuối mỗi luồng với lock.
            fitness_local = BatchAceLibFitness(
                s=s_local,
                fitness_key=self.fitness_fn.fitness_key,
                cache_path=None,
                concurrent_simulations=1,
                pre_request_delay=1.0 + i * 0.5,
                pre_request_jitter=self.fitness_fn.jitter,
                min_fitness=self.fitness_fn.min_fitness,
                self_corr_soft=self.fitness_fn.self_corr_soft,
                self_corr_hard=self.fitness_fn.self_corr_hard,
                **self.fitness_fn.sim_settings,
            )

            gp = WarmStartGP_ACE(
                alphainit_expr=expr,
                fitness_fn=fitness_local,
                catalog=self.catalog,
                seed=42 + i * 13,
                **self.gp_kwargs,
            )
            best_alpha, best_score, _ = gp.run()

            # Merge cache thread-safe
            with cache_lock:
                self.fitness_fn._cache.update(fitness_local._cache)
                self.fitness_fn._save_cache()

            return best_alpha, best_score

        max_workers = min(n, 3)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_one, i, expr): i
                for i, expr in enumerate(self.alphainit_exprs)
            }
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{'='*68}  TỔNG HỢP")
        for rank, (alpha, score) in enumerate(results, 1):
            print(f"  #{rank}  {self.fitness_fn.fitness_key}={score:.4f} | {alpha.to_expr()[:60]}...")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Bước 1: Đăng nhập ────────────────────────────────────────────────
    s = start_session()

    # ── Bước 2: Catalog từ CSV có sẵn ────────────────────────────────────
    catalog = build_catalog_from_csv("all_datasets_fields_full.csv")

    # ── Bước 3: Fitness ───────────────────────────────────────────────────
    fitness = BatchAceLibFitness(
        s                      = s,
        fitness_key            = "sharpe",
        cache_path             = "wsgp_cache.pkl",
        concurrent_simulations = 2,
        pre_request_delay      = 0.4,
        pre_request_jitter     = 0.6,
        # ── settings cho generate_alpha() ─────────────────────────────
        region         = "USA",
        universe       = "TOP3000",
        neutralization = "SUBINDUSTRY",
        delay          = 1,
        decay          = 0,
        truncation     = 0.08,
        nan_handling   = "ON",
        unit_handling  = "VERIFY",
    )

    # ── Bước 4: Alphainit ─────────────────────────────────────────────────
    # Chọn 1 alpha đã chạy tốt từ danh sách của bạn làm điểm khởi đầu
    ALPHAINIT = (
        "rank(bookvalue_ps / (close + 0.0001)) * "
        "rank(net_income_adjusted / (total_assets_reported_value + 0.0001))"
    )

    # ── Bước 5a: Đơn luồng ───────────────────────────────────────────────
    gp = WarmStartGP_ACE(
        alphainit_expr  = ALPHAINIT,
        fitness_fn      = fitness,
        catalog         = catalog,
        population_size = 10,
        generations     = 15,
        crossover_rate  = 0.6,
        tournament_size = 3,
        elitism         = 1,
        seed            = 42,
        verbose         = True,
    )

    best_alpha, best_sharpe, history = gp.run()

    print(f"\nBest alpha : {best_alpha.to_expr()}")
    print(f"IS Sharpe  : {best_sharpe:.4f}")
    fitness.cache_info()

    # ── Bước 5b: Song song nhiều alphainit ───────────────────────────────
    # MULTI_ALPHAS = [ALPHAINIT, "rank(eps / (close + 0.0001))", ...]
    # pgp = ParallelWarmStartGP_ACE(
    #     alphainit_exprs = MULTI_ALPHAS,
    #     fitness_fn      = fitness,
    #     catalog         = catalog,
    #     population_size = 10,
    #     generations     = 10,
    # )
    # all_results = pgp.run()