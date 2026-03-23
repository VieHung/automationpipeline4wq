from typing import Optional

import requests
from urllib.parse import urljoin
import time
import json
import os
import getpass
import random
from pathlib import Path

import pandas as pd

from multiprocessing.pool import ThreadPool
from functools import partial

import tqdm
from helpful_functions import save_simulation_result, set_alpha_properties, save_pnl, save_yearly_stats, get_alpha_pnl, get_alpha_yearly_stats

DEFAULT_CONFIG = {
    "get_pnl": False,
    "get_stats": False,
    "save_pnl_file": False,
    "save_stats_file": False,
    "save_result_file": False,
    "check_submission": False,
    "check_self_corr": False,
    "check_prod_corr": False,
}


def get_credentials():
    """
    Function gets json with platform credentials if exists,
    or asks to enter new ones
    """

    credential_email = os.environ.get('ACE_CREDENTIAL_EMAIL')
    credential_password = os.environ.get('ACE_CREDENTIAL_PASSWORD')

    credentials_folder_path = os.path.join(os.path.expanduser("~"), "secrets")
    credentials_file_path = os.path.join(credentials_folder_path, "platform-brain.json")

    if (
        Path(credentials_file_path).exists()
        and os.path.getsize(credentials_file_path) > 2
    ):
        with open(credentials_file_path) as file:
            data = json.loads(file.read())
    else:
        os.makedirs(credentials_folder_path, exist_ok=True)
        if credential_email and credential_password:
            email = credential_email
            password = credential_password
        else:
            email = input("Email:\n")
            password = getpass.getpass(prompt="Password:")
        data = {"email": email, "password": password}
        with open(credentials_file_path, "w") as file:
            json.dump(data, file)
    return (data["email"], data["password"])


def start_session(): 

    """
    Function sign in to platform
    and checks credentials
    and returns session object
    """

    s = requests.Session()
    s.auth = get_credentials()
    r = s.post("https://api.worldquantbrain.com/authentication")

    if r.status_code == requests.status_codes.codes.unauthorized:
        if r.headers["WWW-Authenticate"] == "persona":
            print(
                "Complete biometrics authentication and press any key to continue: \n"
                + urljoin(r.url, r.headers["Location"]) + "\n"
            )
            input()
            s.post(urljoin(r.url, r.headers["Location"]))
            
            while True:
                if s.post(urljoin(r.url, r.headers["Location"])).status_code != 201:
                    input("Biometrics authentication is not complete. Please try again and press any key when completed \n")
                else:
                    break
        else:
            print("\nIncorrect email or password\n")
            with open(
                os.path.join(os.path.expanduser("~"), "secrets/platform-brain.json"),
                "w",
            ) as file:
                json.dump({}, file)
            return start_session()
    return s

def check_session_timeout(s):
    """
    Function checks session time out
    """

    authentication_url = "https://api.worldquantbrain.com/authentication"
    try:
        result = s.get(authentication_url).json()["token"]["expiry"]
        return result
    except:
        return 0


def generate_alpha(
    regular: str,
    region: str = "USA",
    universe: str = "TOP500",
    neutralization: str = "NONE",
    delay: int = 1,
    decay: int = 2,
    truncation: float = 0.08,
    nan_handling: str = "OFF",
    unit_handling: str = "VERIFY",
    pasteurization: str = "ON",
    visualization: bool = False,
): 
    """
    Function generates data to use in simulation
    has default parameters
    """

    simulation_data = {
        "type": "REGULAR",
        "settings": {
            "nanHandling": nan_handling,
            "instrumentType": "EQUITY",
            "delay": delay,
            "universe": universe,
            "truncation": truncation,
            "unitHandling": unit_handling,
            "pasteurization": pasteurization,
            "region": region,
            "language": "FASTEXPR",
            "decay": decay,
            "neutralization": neutralization,
            "visualization": visualization,
        },
        "regular": regular,
    }
    return simulation_data


def start_simulation(
    s, simulate_data
):  
    simulate_response = s.post(
        "https://api.worldquantbrain.com/simulations", json=simulate_data
    )
    return simulate_response


def simulation_progress(s,
    simulate_response,
):  

    if simulate_response.status_code // 100 != 2:
        print(simulate_response.text)
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    while True:
        simulation_progress = s.get(simulation_progress_url)
        if simulation_progress.headers.get("Retry-After", 0) == 0:
            if simulation_progress.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break

        time.sleep(float(simulation_progress.headers["Retry-After"]))

    if error_flag:

        print("An error occurred")
        if "message" in simulation_progress.json():
            print(simulation_progress.json()["message"])
        return {"completed": False, "result": {}}

    alpha = simulation_progress.json().get("alpha", 0)
    if alpha == 0:
        return {"completed": False, "result": {}}
    simulation_result = get_simulation_result_json(s, alpha)
    return {"completed": True, "result": simulation_result}



def multisimulation_progress(s,
    simulate_response,
): 
    
    if simulate_response.status_code // 100 != 2:
        print(simulate_response.text)
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    while True:
        simulation_progress = s.get(simulation_progress_url)
        if simulation_progress.headers.get("Retry-After", 0) == 0:
            if simulation_progress.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break

        time.sleep(float(simulation_progress.headers["Retry-After"]))

    if error_flag:
        print("An error occurred")
        if "message" in simulation_progress.json():
            print(simulation_progress.json()["message"])
        return {"completed": False, "result": {}}


    children = simulation_progress.json().get("children", 0)
    if len(children) == 0:
        return {"completed": False, "result": {}}
    children_list = []
    for child in children:
        child_progress = s.get("https://api.worldquantbrain.com/simulations/" + child)
        alpha = child_progress.json()["alpha"]
        child_result = get_simulation_result_json(s, alpha)
        children_list.append(child_result)
    return {"completed": True, "result": children_list}


def get_prod_corr(s, alpha_id):
    """
    Function gets alpha's prod correlation
    and save result to dataframe
    """

    while True:
        result = s.get(
            "https://api.worldquantbrain.com/alphas/" + alpha_id + "/correlations/prod"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    prod_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)

    return prod_corr_df


def check_prod_corr_test(s, alpha_id, threshold: float = 0.7):
    """
    Function checks if alpha's prod_corr test passed
    Saves result to dataframe
    """

    prod_corr_df = get_prod_corr(s, alpha_id)
    value = prod_corr_df[prod_corr_df.alphas > 0]["max"].max()
    result = [
        {"test": "PROD_CORRELATION", "result": "PASS" if value <= threshold else "FAIL", "limit": threshold, "value": value, "alpha_id": alpha_id}
    ]
    return pd.DataFrame(result)


def get_self_corr(s, alpha_id):
    """
    Function gets alpha's self correlation
    and save result to dataframe
    """

    while True:

        result = s.get(
            "https://api.worldquantbrain.com/alphas/" + alpha_id + "/correlations/self"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        return pd.DataFrame()
    
    records_len = len(result.json()["records"])
    if records_len == 0:
        return pd.DataFrame()

    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    self_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)

    return self_corr_df


def check_self_corr_test(s, alpha_id, threshold: float = 0.7):
    """
    Function checks if alpha's self_corr test passed
    Saves result to dataframe
    """

    self_corr_df = get_self_corr(s, alpha_id)
    if self_corr_df.empty:
        result = [{"test": "SELF_CORRELATION", "result": "PASS", "limit": threshold, "value": 0, "alpha_id": alpha_id}]
    else:
        value = self_corr_df["correlation"].max()
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS" if value < threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id
            }
        ]
    return pd.DataFrame(result)



def get_check_submission(s, alpha_id):
    """
    Function gets alpha's check submission checks
    and returns result in dataframe
    """

    while True:
        result = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/check")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("is", 0) == 0:
        return pd.DataFrame()
    
    checks_df = pd.DataFrame(
            result.json()["is"]["checks"]
    ).assign(alpha_id=alpha_id)
    
    if 'year' in checks_df:
        ladder_dict = [checks_df.loc[checks_df.index[checks_df['name']=='IS_LADDER_SHARPE']][['value', 'year']].iloc[0].to_dict()]
        checks_df.at[checks_df.index[checks_df['name']=='IS_LADDER_SHARPE'], 'value'] = ladder_dict
        checks_df.drop(['endDate', 'startDate', 'year'], axis=1, inplace=True)

    return checks_df


def performance_comparison(s, alpha_id, team_id:Optional[str] = None, competition:Optional[str] = 'ACE2023'):
    """
    Returns performance comparison for merged performance check
    """
    if competition is not None:
        part_url = f'competitions/{competition}'
    elif team_id is not None:
        part_url = f'teams/{team_id}'
    else:
        part_url = 'users/self'
    while True:
        result = s.get(
            f"https://api.worldquantbrain.com/{part_url}/alphas/" + alpha_id + "/before-and-after-performance"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("stats", 0) == 0:
        return {}
    if result.status_code != 200:
        return {}

    return result.json()


def submit_alpha(s, alpha_id):
    """
    Function submits an alpha
    This function is not used anywhere
    """
    result = s.post("https://api.worldquantbrain.com/alphas/" + alpha_id + "/submit")
    while True:
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
            result = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/submit")
        else:
            break
    return result.status_code == 200


def get_simulation_result_json(s, alpha_id):
    return s.get("https://api.worldquantbrain.com/alphas/" + alpha_id).json()


def simulate_single_alpha(
    s,
    simulate_data,
    pre_request_delay: float = 0.0,
    pre_request_jitter: float = 0.0,
):
    """
    To simulate single alpha
    """
    
    if check_session_timeout(s) < 1000:
        s = start_session()

    # Stagger requests from concurrent workers to avoid burst traffic / 429
    sleep_seconds = max(0.0, pre_request_delay)
    if pre_request_jitter > 0:
        sleep_seconds += random.uniform(0, pre_request_jitter)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    simulate_response = start_simulation(s, simulate_data)
    simulation_result = simulation_progress(s, simulate_response)
    
    if not simulation_result["completed"]:
        return {'alpha_id': None, 'simulate_data': simulate_data}
    set_alpha_properties(s, simulation_result["result"]["id"])
    return {'alpha_id': simulation_result["result"]["id"], 'simulate_data': simulate_data}


def simulate_multi_alpha(
    s,
    simulate_data_list,
    pre_request_delay: float = 0.0,
    pre_request_jitter: float = 0.0,
):
    """
    To simulate single alpha
    """
    
    if check_session_timeout(s) < 1000:
        s = start_session()

    # Stagger requests from concurrent workers to avoid burst traffic / 429
    sleep_seconds = max(0.0, pre_request_delay)
    if pre_request_jitter > 0:
        sleep_seconds += random.uniform(0, pre_request_jitter)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    if len(simulate_data_list) == 1:
        return [
            simulate_single_alpha(
                s,
                simulate_data_list[0],
                pre_request_delay=pre_request_delay,
                pre_request_jitter=pre_request_jitter,
            )
        ]
    simulate_response = start_simulation(s, simulate_data_list)
    simulation_result = multisimulation_progress(s, simulate_response)
    
    if not simulation_result["completed"]:
        return [{'alpha_id': None, 'simulate_data': x} for x in simulate_data_list]
    result = [{"alpha_id": x["id"], "simulate_data": {"type": x["type"], "settings": x["settings"], "regular": x["regular"]["code"]}} for x in simulation_result["result"]]
    _ = [set_alpha_properties(s, x["id"]) for x in simulation_result["result"]]
    return result
    

def get_specified_alpha_stats(
    s,
    alpha_id,
    simulate_data,
    get_pnl: bool = False,
    get_stats: bool = False,
    save_pnl_file: bool = False,
    save_stats_file: bool = False,
    save_result_file: bool = False,
    check_submission: bool = False,
    check_self_corr: bool = False,
    check_prod_corr: bool = False,
):
    """
    Master-Function to get specified in config statistics

    """
    pnl = None
    stats = None

    if alpha_id is None:
        return {'alpha_id': None, 'simulate_data': simulate_data, 'is_stats': None, 'pnl': pnl, 'stats': stats, 'is_tests': None}

    result = get_simulation_result_json(s, alpha_id)
    region = result["settings"]["region"]
    is_stats = pd.DataFrame([{key: value for key, value in result['is'].items() if key!='checks'}]).assign(alpha_id=alpha_id)
    
    if get_pnl:
        pnl = get_alpha_pnl(s, alpha_id)
    if get_stats:
        stats = get_alpha_yearly_stats(s, alpha_id)
    
    if save_result_file:
        save_simulation_result(result)
    if save_pnl_file and get_pnl:
        save_pnl(pnl, alpha_id, region)
    if save_stats_file and get_stats:
        save_yearly_stats(stats, alpha_id, region)

    is_tests = pd.DataFrame(
        result["is"]["checks"]
    ).assign(alpha_id=alpha_id)

    if check_submission:
        is_tests = get_check_submission(s, alpha_id)

        return {'alpha_id': alpha_id, 'simulate_data': simulate_data, 'is_stats': is_stats, 'pnl': pnl, 'stats': stats, 'is_tests': is_tests}

    if check_self_corr and not check_submission:
        self_corr_test = check_self_corr_test(s, alpha_id)
        is_tests = (
            is_tests.append(self_corr_test, ignore_index=True, sort=False)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )
    if check_prod_corr and not check_submission:
        prod_corr_test = check_prod_corr_test(s, alpha_id)
        is_tests = (
            is_tests.append(prod_corr_test, ignore_index=True, sort=False)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )

    return {'alpha_id': alpha_id, 'simulate_data': simulate_data, 'is_stats': is_stats, 'pnl': pnl, 'stats': stats, 'is_tests': is_tests}


def simulate_alpha_list(
    s,
    alpha_list,
    limit_of_concurrent_simulations=3,
    simulation_config=DEFAULT_CONFIG,
    pre_request_delay: float = 0.0,
    pre_request_jitter: float = 0.0,
):
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        
        with tqdm.tqdm(total=len(alpha_list)) as pbar:
            
            for result in pool.imap_unordered(
                partial(
                    simulate_single_alpha,
                    s,
                    pre_request_delay=pre_request_delay,
                    pre_request_jitter=pre_request_jitter,
                ),
                alpha_list,
            ):
                result_list.append(result)
                pbar.update()

    stats_list_result = []
    func = lambda x: get_specified_alpha_stats(s, x['alpha_id'], x['simulate_data'], **simulation_config)
    with ThreadPool(3) as pool:
        for result in pool.map(
            func, result_list
        ):
            stats_list_result.append(result)
    
    return stats_list_result 


def simulate_alpha_list_multi(
    s,
    alpha_list,
    limit_of_concurrent_simulations=3,
    limit_of_multi_simulations=3,
    simulation_config=DEFAULT_CONFIG,
    pre_request_delay: float = 0.0,
    pre_request_jitter: float = 0.0,
):
    if (limit_of_multi_simulations<2) or (limit_of_multi_simulations>10):
        print('Warning, limit of multi-simulation should be 2..10')
        limit_of_multi_simulations = 3
    if len(alpha_list)<10:
        print('Warning, list of alphas too short, single concurrent simulations will be used instead of multisimulations')
        return simulate_alpha_list(
            s,
            alpha_list,
            simulation_config=simulation_config,
            pre_request_delay=pre_request_delay,
            pre_request_jitter=pre_request_jitter,
        )
    
    tasks = [alpha_list[i:i + limit_of_multi_simulations] for i in range(0, len(alpha_list), limit_of_multi_simulations)]
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        
        with tqdm.tqdm(total=len(tasks)) as pbar:
                
            for result in pool.imap_unordered(
                partial(
                    simulate_multi_alpha,
                    s,
                    pre_request_delay=pre_request_delay,
                    pre_request_jitter=pre_request_jitter,
                ),
                tasks,
            ):
                result_list.append(result)
                pbar.update()
    result_list_flat = [item for sublist in result_list for item in sublist]
    
    stats_list_result = []
    func = lambda x: get_specified_alpha_stats(s, x['alpha_id'], x['simulate_data'], **simulation_config)
    with ThreadPool(3) as pool:
        for result in pool.map(
            func, result_list_flat
        ):
            stats_list_result.append(result)
            
    return stats_list_result 


def main():
    """
    Main function
    """

    s = start_session()

    k = [
        # 1. B/M ratio: high = cheap vs book value
    "rank(bookvalue_ps / (close + 0.0001))",
 
    # 2. Inverse P/B (same signal, different form)
    "-rank(close / (bookvalue_ps + 0.0001))",
 
    # 3. B/M within SUBINDUSTRY group
    "group_neutralize(rank(bookvalue_ps / (close + 0.0001)), subindustry)",
 
    # 4. P/B momentum reversal: P/B rising fast → future underperformance
    "-rank(ts_delta(close / (bookvalue_ps + 0.0001), 60))",
 
    # 5. P/B ts_rank: relatively expensive vs own history
    "-rank(ts_rank(close / (bookvalue_ps + 0.0001), 252))",
 
    # 6. P/B z-score vs own 1-year history
    "-rank(ts_zscore(close / (bookvalue_ps + 0.0001), 252))",
 
    # 7. B/M group rank within subindustry (cheapest-in-group signal)
    "group_rank(bookvalue_ps / (close + 0.0001), subindustry)",
 
    # 8. Value + no recent price rally: B/M high AND price flat
    "rank(bookvalue_ps / (close + 0.0001)) - rank(ts_delta(close, 60))",
 
    # 9. Smoothed B/M (20-day average to cut noise from stale book data)
    "rank(ts_mean(bookvalue_ps / (close + 0.0001), 20))",
 
    # 10. Equity / cap: total equity relative to market cap
    "rank(equity / (cap + 0.0001))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 2  ▸  EARNINGS YIELD / E/P / EPS  (Paper Rank #14 & #16)
    # ═══════════════════════════════════════════════════════════════════════
 
    # 11. Earnings yield = E/P
    "rank(eps / (close + 0.0001))",
 
    # 12. Negative P/E (lower = cheaper)
    "-rank(close / (eps + 0.0001))",
 
    # 13. EPS time-series momentum (improving earnings)
    "rank(ts_delta(eps, 60))",
 
    # 14. EPS ts_rank vs own history
    "rank(ts_rank(eps, 252))",
 
    # 15. Smoothed earnings yield to reduce quarterly jump noise
    "rank(ts_mean(eps / (close + 0.0001), 60))",
 
    # 16. EPS acceleration: second derivative of earnings
    "rank(ts_delta(ts_delta(eps, 20), 20))",
 
    # 17. EPS z-score: how far current EPS is from 1-year average
    "rank(ts_zscore(eps, 252))",
 
    # 18. P/E compression: P/E shrinking means stock getting cheaper
    "-rank(ts_delta(close / (eps + 0.0001), 60))",
 
    # 19. Earnings yield group rank within subindustry
    "group_rank(eps / (close + 0.0001), subindustry)",
 
    # 20. Earnings yield × positive EPS trend (quality filter)
    "rank(eps / (close + 0.0001)) * sign(ts_delta(eps, 60))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 3  ▸  PRICE-TO-SALES  (Paper Rank #5 — TOP RATIO)
    # sales_ps = sales per share; P/S = close / sales_ps
    # ═══════════════════════════════════════════════════════════════════════
 
    # 21. Revenue yield: sales_ps / close (high = cheap vs revenue)
    "rank(sales_ps / (close + 0.0001))",
 
    # 22. Negative P/S (lower P/S = more value)
    "-rank(close / (sales_ps + 0.0001))",
 
    # 23. Revenue yield within subindustry
    "group_neutralize(-rank(close / (sales_ps + 0.0001)), subindustry)",
 
    # 24. Revenue growth: year-over-year sales growth rate
    "rank(ts_delta(revenue, 252) / (abs(ts_delay(revenue, 252)) + 0.0001))",
 
    # 25. Smoothed revenue yield (60-day)
    "rank(ts_mean(sales_ps / (close + 0.0001), 60))",
 
    # 26. P/S ts_rank vs own history
    "-rank(ts_rank(close / (sales_ps + 0.0001), 252))",
 
    # 27. Revenue yield group rank within subindustry
    "group_rank(sales_ps / (close + 0.0001), subindustry)",
 
    # 28. P/S z-score
    "-rank(ts_zscore(close / (sales_ps + 0.0001), 252))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 4  ▸  ROA / OROA / GPM — Profitability  (Paper Rank #6, #13, #15)
    # ═══════════════════════════════════════════════════════════════════════
 
    # 29. ROA: net_income / total_assets
    "rank(net_income_adjusted / (total_assets_reported_value + 0.0001))",
 
    # 30. OROA: operating_income / total_assets
    "rank(operating_income / (total_assets_reported_value + 0.0001))",
 
    # 31. GPM: gross_income / revenue (gross profit margin)
    "rank(gross_income_total / (revenue + 0.0001))",
 
    # 32. ROA within subindustry
    "group_neutralize(rank(net_income_adjusted / (total_assets_reported_value + 0.0001)), subindustry)",
 
    # 33. GPM within subindustry
    "group_neutralize(rank(gross_income_total / (revenue + 0.0001)), subindustry)",
 
    # 34. ROA improvement momentum
    "rank(ts_delta(net_income_adjusted / (total_assets_reported_value + 0.0001), 60))",
 
    # 35. Smoothed ROA (60-day to bridge quarterly reporting gaps)
    "rank(ts_mean(net_income_adjusted / (total_assets_reported_value + 0.0001), 60))",
 
    # 36. ROA ts_rank vs own history
    "rank(ts_rank(net_income_adjusted / (total_assets_reported_value + 0.0001), 252))",
 
    # 37. ROA z-score
    "rank(ts_zscore(net_income_adjusted / (total_assets_reported_value + 0.0001), 252))",
 
    # 38. Novy-Marx quality: gross_income / total_assets
    "rank(gross_income_total / (total_assets_reported_value + 0.0001))",
 
    # 39. OROA group rank within subindustry
    "group_rank(operating_income / (total_assets_reported_value + 0.0001), subindustry)",
 
    # 40. GPM z-score
    "rank(ts_zscore(gross_income_total / (revenue + 0.0001), 252))",
 
    # 41. OROA improvement trend
    "rank(ts_delta(operating_income / (total_assets_reported_value + 0.0001), 60))",
 
    # 42. GPM improvement trend
    "rank(ts_delta(gross_income_total / (revenue + 0.0001), 60))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 5  ▸  EFFICIENCY: ITR & WCTR  (Paper Rank #7 & #18)
    # inventory_turnover is pre-computed in BRAIN
    # ═══════════════════════════════════════════════════════════════════════
 
    # 43. ITR: pre-computed inventory turnover
    "rank(inventory_turnover)",
 
    # 44. WCTR: sales / working_capital
    "rank(revenue / (working_capital + 0.0001))",
 
    # 45. ITR within subindustry
    "group_neutralize(rank(inventory_turnover), subindustry)",
 
    # 46. WCTR within subindustry
    "group_neutralize(rank(revenue / (working_capital + 0.0001)), subindustry)",
 
    # 47. ITR improvement momentum
    "rank(ts_delta(inventory_turnover, 60))",
 
    # 48. Asset turnover: revenue / total_assets
    "rank(revenue / (total_assets_reported_value + 0.0001))",
 
    # 49. ITR z-score vs own history
    "rank(ts_zscore(inventory_turnover, 252))",
 
    # 50. ITR group rank
    "group_rank(inventory_turnover, subindustry)",
 
    # 51. WCTR momentum
    "rank(ts_delta(revenue / (working_capital + 0.0001), 60))",
 
    # 52. Smoothed ITR
    "rank(ts_mean(inventory_turnover, 60))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 6  ▸  MARKET CAP / SIZE  (Paper Rank #11)
    # cap = market capitalization (confirmed field)
    # ═══════════════════════════════════════════════════════════════════════
 
    # 53. Small-cap premium: buy small
    "-rank(cap)",
 
    # 54. Market cap growth momentum (size expansion = winning)
    "rank(ts_delta(cap, 20))",
 
    # 55. Smallest-in-subindustry signal
    "group_rank(-cap, subindustry)",
 
    # 56. Log market cap (smoother small-cap signal)
    "-rank(log(cap + 1))",
 
    # 57. Market cap z-score (relatively small vs own history)
    "-rank(ts_zscore(cap, 252))",
 
    # 58. Small cap neutralized within subindustry
    "group_neutralize(-rank(cap), subindustry)",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 7  ▸  PRICE-TO-CASH-FLOW  (Paper Rank #17)
    # CF/P = cash_flow_from_operations / cap
    # ═══════════════════════════════════════════════════════════════════════
 
    # 59. Cash flow yield: CF / market cap
    "rank(cash_flow_from_operations / (cap + 0.0001))",
 
    # 60. Negative P/CF equivalent
    "-rank(cap / (cash_flow_from_operations + 0.0001))",
 
    # 61. CF yield within subindustry
    "group_neutralize(rank(cash_flow_from_operations / (cap + 0.0001)), subindustry)",
 
    # 62. Cash flow momentum: growing operating CF
    "rank(ts_delta(cash_flow_from_operations, 60))",
 
    # 63. Free cash flow yield
    "rank(free_cash_flow_total / (cap + 0.0001))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 8  ▸  NET INCOME GROWTH / NIGR  (Paper Rank #19)
    # ═══════════════════════════════════════════════════════════════════════
 
    # 64. YoY net income growth rate
    "rank(ts_delta(net_income_adjusted, 252) / (abs(ts_delay(net_income_adjusted, 252)) + 0.0001))",
 
    # 65. Earnings growth relative to own history
    "rank(ts_rank(ts_delta(net_income_adjusted, 60), 252))",
 
    # 66. EPS growth momentum (smoothed 120-day)
    "rank(ts_mean(ts_delta(eps, 60), 120))",
 
    # 67. Earnings growth within subindustry
    "group_neutralize(rank(ts_delta(net_income_adjusted, 252) / (abs(net_income_adjusted) + 0.0001)), subindustry)",
 
    # 68. Revenue growth momentum
    "rank(ts_delta(sales, 252) / (abs(ts_delay(sales, 252)) + 0.0001))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 9  ▸  MULTI-FACTOR COMPOSITES  (XGBoost non-linear combos)
    # Paper: XGBoost 83.5% accuracy via non-linear feature interactions
    # Strategy: multiply orthogonal signals from paper's top features
    # ═══════════════════════════════════════════════════════════════════════
 
    # 69. Value × Profitability: low P/B + high ROA (Piotroski F-score logic)
    "rank(bookvalue_ps / (close + 0.0001)) * rank(net_income_adjusted / (total_assets_reported_value + 0.0001))",
 
    # 70. Double value: earnings yield × revenue yield
    "rank(eps / (close + 0.0001)) * rank(sales_ps / (close + 0.0001))",
 
    # 71. Margin × Asset productivity (DuPont decomposition)
    "rank(net_income_adjusted / (revenue + 0.0001)) * rank(revenue / (total_assets_reported_value + 0.0001))",
 
    # 72. Efficiency × Profitability: ITR × ROA
    "rank(inventory_turnover) * rank(net_income_adjusted / (total_assets_reported_value + 0.0001))",
 
    # 73. Quality minus Price: ROA minus P/S (want cheap AND profitable)
    "rank(net_income_adjusted / (total_assets_reported_value + 0.0001)) - rank(close / (sales_ps + 0.0001))",
 
    # 74. Revenue yield × Operating margin
    "rank(sales_ps / (close + 0.0001)) * rank(operating_income / (revenue + 0.0001))",
 
    # 75. B/M × GPM: cheap + high gross margin (value + quality)
    "rank(bookvalue_ps / (close + 0.0001)) * rank(gross_income_total / (revenue + 0.0001))",
 
    # 76. Cash flow yield × ROA (cash quality + profitability)
    "rank(cash_flow_from_operations / (cap + 0.0001)) * rank(net_income_adjusted / (total_assets_reported_value + 0.0001))",
 
    # 77. Gross profit/assets (Novy-Marx) × Revenue yield
    "rank(gross_income_total / (total_assets_reported_value + 0.0001)) * rank(sales_ps / (close + 0.0001))",
 
    # 78. Triple factor: E/P + CF/P + ROA
    "rank(eps / (close + 0.0001)) + rank(cash_flow_from_operations / (cap + 0.0001)) + rank(net_income_adjusted / (total_assets_reported_value + 0.0001))",
 
    # 79. Efficiency composite: ITR + WCTR
    "rank(inventory_turnover) + rank(revenue / (working_capital + 0.0001))",
 
    # 80. GPM minus valuation multiple (quality at a discount)
    "rank(gross_income_total / (revenue + 0.0001)) - rank(close / (eps + 0.0001))",
 
    # 81. Three-way: B/M × E/P × ITR (value × earnings × efficiency)
    "rank(bookvalue_ps / (close + 0.0001)) * rank(eps / (close + 0.0001)) * rank(inventory_turnover)",
 
    # 82. Earnings growth × GPM (growing earnings in high-margin companies)
    "rank(ts_delta(net_income_adjusted, 252) / (abs(net_income_adjusted) + 0.0001)) * rank(gross_income_total / (revenue + 0.0001))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 10  ▸  SUBINDUSTRY GROUP COMPOSITES
    # Designed to survive SUBINDUSTRY neutralization
    # ═══════════════════════════════════════════════════════════════════════
 
    # 83. Quality composite: E/P + ROA within subindustry
    "group_neutralize(rank(eps / (close + 0.0001)) + rank(net_income_adjusted / (total_assets_reported_value + 0.0001)), subindustry)",
 
    # 84. Value × quality: B/M × GPM within subindustry
    "group_neutralize(rank(bookvalue_ps / (close + 0.0001)) * rank(gross_income_total / (revenue + 0.0001)), subindustry)",
 
    # 85. Cash × profitability within subindustry
    "group_neutralize(rank(cash_flow_from_operations / (cap + 0.0001)) * rank(net_income_adjusted / (total_assets_reported_value + 0.0001)), subindustry)",
 
    # 86. Efficiency × profitability within subindustry
    "group_neutralize(rank(inventory_turnover) * rank(net_income_adjusted / (total_assets_reported_value + 0.0001)), subindustry)",
 
    # 87. Combined ROA + margin within subindustry
    "group_rank(net_income_adjusted / (total_assets_reported_value + 0.0001) + net_income_adjusted / (revenue + 0.0001), subindustry)",
 
    # 88. EPS group rank (earnings relative to peers)
    "group_rank(eps, subindustry)",
 
    # 89. Cash flow yield group rank
    "group_rank(cash_flow_from_operations / (cap + 0.0001), subindustry)",
 
    # 90. B/M group rank
    "group_rank(bookvalue_ps / (close + 0.0001), subindustry)",
 
    # 91. EPS ts_rank within subindustry
    "group_neutralize(rank(ts_rank(eps, 252)), subindustry)",
 
    # 92. Revenue yield + ROA composite within subindustry
    "group_neutralize(rank(sales_ps / (close + 0.0001)) + rank(net_income_adjusted / (total_assets_reported_value + 0.0001)), subindustry)",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 11  ▸  TIME-SERIES RELATIVE FUNDAMENTALS
    # Fundamental improvement relative to own history
    # ═══════════════════════════════════════════════════════════════════════
 
    # 93. ROA vs own trailing year
    "rank(ts_rank(net_income_adjusted / (total_assets_reported_value + 0.0001), 252))",
 
    # 94. Earnings yield vs own history
    "rank(ts_rank(eps / (close + 0.0001), 252))",
 
    # 95. Earnings yield acceleration (speed of improvement)
    "rank(ts_delta(ts_rank(eps, 252), 60))",
 
    # 96. GPM vs own history
    "rank(ts_rank(gross_income_total / (revenue + 0.0001), 252))",
 
    # 97. B/M vs own history (currently cheap relative to own norm)
    "-rank(ts_rank(close / (bookvalue_ps + 0.0001), 252))",
 
    # 98. CF yield vs own history
    "rank(ts_rank(cash_flow_from_operations / (cap + 0.0001), 252))",
 
    # 99. ITR vs own trailing year
    "rank(ts_rank(inventory_turnover, 252))",
 
    # 100. ROA acceleration: is profitability improvement speeding up?
    "rank(ts_delta(ts_rank(net_income_adjusted / (total_assets_reported_value + 0.0001), 252), 60))",
 
    ]

    alpha_list = [generate_alpha(x) for x in k]

    # Anti-burst settings: stagger each worker request to avoid 429 / DDOS-like spikes
    simulate_alpha_list(
        s,
        alpha_list,
        limit_of_concurrent_simulations=2,
        pre_request_delay=0.4,
        pre_request_jitter=0.6,
    )


if __name__ == "__main__":
    main()
