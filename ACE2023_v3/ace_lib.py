from typing import Optional

import requests
from urllib.parse import urljoin
import time
import json
import os
import getpass
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
    universe: str = "TOP3000",
    neutralization: str = "SUBINDUSTRY",
    delay: int = 1,
    decay: int = 4,
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
):
    """
    To simulate single alpha
    """
    
    if check_session_timeout(s) < 1000:
        s = start_session()

    simulate_response = start_simulation(s, simulate_data)
    simulation_result = simulation_progress(s, simulate_response)
    
    if not simulation_result["completed"]:
        return {'alpha_id': None, 'simulate_data': simulate_data}
    set_alpha_properties(s, simulation_result["result"]["id"])
    return {'alpha_id': simulation_result["result"]["id"], 'simulate_data': simulate_data}


def simulate_multi_alpha(
    s,
    simulate_data_list,
):
    """
    To simulate single alpha
    """
    
    if check_session_timeout(s) < 1000:
        s = start_session()
    if len(simulate_data_list) == 1:
        return [simulate_single_alpha(s, simulate_data_list[0])]
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
):
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        
        with tqdm.tqdm(total=len(alpha_list)) as pbar:
            
            for result in pool.imap_unordered(
                partial(simulate_single_alpha, s), alpha_list
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
):
    if (limit_of_multi_simulations<2) or (limit_of_multi_simulations>10):
        print('Warning, limit of multi-simulation should be 2..10')
        limit_of_multi_simulations = 3
    if len(alpha_list)<10:
        print('Warning, list of alphas too short, single concurrent simulations will be used instead of multisimulations')
        return simulate_alpha_list(s, alpha_list, simulation_config=simulation_config)
    
    tasks = [alpha_list[i:i + limit_of_multi_simulations] for i in range(0, len(alpha_list), limit_of_multi_simulations)]
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        
        with tqdm.tqdm(total=len(tasks)) as pbar:
                
            for result in pool.imap_unordered(
                partial(simulate_multi_alpha, s), tasks
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
    "rank(book_value / (close * shares))",
 
    # 2. Inverse P/B: low price-to-book is value
    "-rank(close / (book_value_per_share_reported_value + 0.0001))",
 
    # 3. B/M within SUBINDUSTRY (boosts signal after neutralization)
    "group_neutralize(rank(book_value_per_share_reported_value / (close + 0.0001)), subindustry)",
 
    # 4. Time-series momentum on B/M ratio
    "rank(ts_delta(book_value_per_share_reported_value / (close + 0.0001), 60))",
 
    # 5. B/M ranked over past year (relative to own history)
    "rank(ts_rank(book_value_per_share_reported_value / (close + 0.0001), 252))",
 
    # 6. B/M z-score — how far from its own historical mean
    "rank(ts_zscore(book_value_per_share_reported_value / (close + 0.0001), 252))",
 
    # 7. Group rank of B/M within subindustry
    "group_rank(book_value_per_share_reported_value / (close + 0.0001), subindustry)",
 
    # 8. Combined: buy high B/M + selling when P/B increasing (momentum reversal)
    "rank(book_value_per_share_reported_value / (close  + 0.0001)) - rank(ts_delta(close / (book_value_per_share_reported_value + 0.0001), 60))",
 
    # 9. Smoothed B/M to remove noise (20-day average)
    "rank(ts_mean(book_value_per_share_reported_value (close + 0.0001), 20))",
 
    # 10. B/M changes: companies growing book value faster than price
    "rank(ts_delta(book_value_per_share_reported_value, 252) / (ts_delay(book_value_per_share_reported_value, 252) + 0.0001) - ts_delta(close , 252) / (ts_delay(close , 252) + 0.0001))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 2 — EARNINGS YIELD / E/P / EPS (Paper Rank #14, #16)
    # Theory: High earnings relative to price = undervalued + quality
    # Paper: EPS and E/P rank 14th and 16th in importance
    # ═══════════════════════════════════════════════════════════════════════
 
    # 11. Earnings yield = E/P (inverse of P/E)
    "rank(eps / (close + 0.0001))",
 
    # 12. Within-subindustry earnings yield
    "group_neutralize(rank(eps / (close + 0.0001)), subindustry)",
 
    # 13. EPS time-series momentum (rising earnings)
    "rank(ts_delta(eps, 60))",
 
    # 14. EPS relative to its own history
    "rank(ts_rank(eps, 252))",
 
    # 15. Smoothed earnings yield to reduce quarterly noise
    "rank(ts_mean(eps, 60) / (close + 0.0001))",
 
    # 16. Earnings acceleration: change in earnings growth rate
    "rank(ts_delta(ts_delta(eps, 20), 20))",
 
    # 17. EPS zscore — standardized relative to history
    "rank(ts_zscore(eps, 252))",
 
    # 18. P/E compression (P/E ratio shrinking = becoming cheaper over time)
    "-rank(ts_delta(close / (eps + 0.0001), 60))",
 
    # 19. Earnings yield group rank within subindustry
    "group_rank(eps / (close + 0.0001), subindustry)",
 
    # 20. Earnings yield combined with earnings trend
    "rank(eps / (close + 0.0001)) * sign(ts_delta(eps, 60))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 3 — PRICE-TO-SALES / P/S RATIO (Paper Rank #5 — MOST IMPORTANT RATIO)
    # Theory: Low P/S stocks have been the single most predictive ratio in Iran market
    # Consistent with global findings: revenue yield predicts returns
    # ═══════════════════════════════════════════════════════════════════════
 
    # 21. Revenue yield (inverse P/S): more revenue per dollar of market cap
    "rank(revenue / (close * shares + 0.0001))",
 
    # 22. Negative P/S: lower is better (value)
    "-rank(close * shares / (revenue + 0.0001))",
 
    # 23. Revenue yield within subindustry
    "group_neutralize(-rank(close * shares / (revenue + 0.0001)), subindustry)",
 
    # 24. Revenue growth momentum (top-line acceleration)
    "rank(ts_delta(revenue, 252) / (abs(ts_delay(revenue, 252)) + 0.0001))",
 
    # 25. Smoothed revenue yield
    "rank(ts_mean(revenue, 60) / (close * shares + 0.0001))",
 
    # 26. P/S ratio ts_rank (relative to own history)
    "-rank(ts_rank(close * shares / (revenue + 0.0001), 252))",
 
    # 27. Revenue yield group rank
    "group_rank(revenue / (close * shares + 0.0001), subindustry)",
 
    # 28. Revenue yield z-score
    "rank(ts_zscore(revenue / (close * shares + 0.0001), 252))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 4 — RETURN ON ASSETS / PROFITABILITY (Paper Rank #6, #13, #15)
    # Theory: ROA, OROA, GPM — profitability quality factor
    # Paper: These 3 metrics together capture operational efficiency
    # ═══════════════════════════════════════════════════════════════════════
 
    # 29. ROA: net income / assets
    "rank(net_income_adjusted / (total_assets + 0.0001))",
 
    # 30. OROA: operating income / assets (cleaner than ROA)
    "rank(operating_income / (total_assets + 0.0001))",
 
    # 31. GPM: gross profit margin (high = pricing power)
    "rank(gross_profit / (revenue + 0.0001))",
 
    # 32. ROA within subindustry (compensates for SUBINDUSTRY neutralization)
    "group_neutralize(rank(net_income_adjusted / (total_assets + 0.0001)), subindustry)",
 
    # 33. GPM within subindustry
    "group_neutralize(rank(gross_profit / (revenue + 0.0001)), subindustry)",
 
    # 34. ROA momentum: improving profitability
    "rank(ts_delta(net_income_adjusted / (total_assets + 0.0001), 60))",
 
    # 35. Smoothed ROA to reduce quarterly noise
    "rank(ts_mean(net_income_adjusted / (total_assets + 0.0001), 60))",
 
    # 36. ROA relative to own history
    "rank(ts_rank(net_income_adjusted / (total_assets + 0.0001), 252))",
 
    # 37. ROA z-score
    "rank(ts_zscore(net_income_adjusted / (total_assets + 0.0001), 252))",
 
    # 38. Gross profit to assets (Novy-Marx quality factor)
    "rank(gross_profit / (total_assets + 0.0001))",
 
    # 39. OROA group rank
    "group_rank(operating_income / (total_assets + 0.0001), subindustry)",
 
    # 40. GPM z-score over time
    "rank(ts_zscore(gross_profit / (revenue + 0.0001), 252))",
 
    # 41. Operating income growth (OROA change)
    "rank(ts_delta(operating_income / (total_assets + 0.0001), 60))",
 
    # 42. Gross profit margin trend
    "rank(ts_delta(gross_profit / (revenue + 0.0001), 60))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 5 — EFFICIENCY RATIOS: ITR & WCTR (Paper Rank #7, #18)
    # Theory: High inventory turnover = operational efficiency + demand strength
    # WCTR captures working capital efficiency
    # ═══════════════════════════════════════════════════════════════════════
 
    # 43. Inventory Turnover Ratio (ITR): revenue / inventory
    "rank(revenue / (inventory + 0.0001))",
 
    # 44. Working Capital Turnover (WCTR): revenue / working_capital
    "rank(revenue / (working_capital + 0.0001))",
 
    # 45. ITR within subindustry
    "group_neutralize(rank(revenue / (inventory + 0.0001)), subindustry)",
 
    # 46. WCTR within subindustry
    "group_neutralize(rank(revenue / (working_capital + 0.0001)), subindustry)",
 
    # 47. ITR momentum
    "rank(ts_delta(revenue / (inventory + 0.0001), 60))",
 
    # 48. Asset Turnover Ratio (related efficiency metric)
    "rank(revenue / (total_assets + 0.0001))",
 
    # 49. ITR z-score
    "rank(ts_zscore(revenue / (inventory + 0.0001), 252))",
 
    # 50. ITR group rank
    "group_rank(revenue / (inventory + 0.0001), subindustry)",
 
    # 51. WCTR momentum
    "rank(ts_delta(revenue / (working_capital + 0.0001), 60))",
 
    # 52. Smoothed ITR
    "rank(ts_mean(revenue / (inventory + 0.0001), 60))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 6 — MARKET CAP / SIZE (Paper Rank #11)
    # Theory: Small-cap premium (Fama-French) — smaller MC = higher expected returns
    # Paper: MC is a significant predictor in both directions
    # ═══════════════════════════════════════════════════════════════════════
 
    # 53. Small-cap premium: buy small
    "-rank(close * shares)",
 
    # 54. Market cap change (momentum: growing caps are winning stocks)
    "rank(ts_delta(close * shares, 20))",
 
    # 55. Relative size rank within subindustry (bet against giants)
    "group_rank(-close * shares, subindustry)",
 
    # 56. Log market cap (smoother size signal)
    "-rank(log(close * shares + 1))",
 
    # 57. Market cap z-score
    "-rank(ts_zscore(close * shares, 252))",
 
    # 58. Small cap within group: smallest in subindustry
    "group_neutralize(-rank(close * shares), subindustry)",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 7 — PRICE-TO-CASH-FLOW (Paper Rank #17)
    # Theory: P/CF is cleaner than P/E (harder to manipulate)
    # Operating cash flow quality signal
    # ═══════════════════════════════════════════════════════════════════════
 
    # 59. Cash flow yield: operating CF / market cap
    "rank(operating_cashflow_reported_value / (close * shares + 0.0001))",
 
    # 60. Negative P/CF: low P/CF = value
    "-rank(close * shares / (operating_cf + 0.0001))",
 
    # 61. CF yield within subindustry
    "group_neutralize(rank(operating_cf / (close * shares + 0.0001)), subindustry)",
 
    # 62. Cash flow momentum
    "rank(ts_delta(operating_cf, 60))",
 
    # 63. Cash flow yield z-score
    "rank(ts_zscore(operating_cf / (close * shares + 0.0001), 252))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 8 — NET INCOME GROWTH (Paper Rank #19: NIGR)
    # Theory: Earnings growth predicts future returns — momentum in fundamentals
    # Paper: NIGR ranked 19th, growth category has mild but real impact
    # ═══════════════════════════════════════════════════════════════════════
 
    # 64. Year-over-year net income growth
    "rank(ts_delta(net_income_adjusted, 252) / (abs(ts_delay(net_income_adjusted, 252)) + 0.0001))",
 
    # 65. Relative earnings growth vs own history
    "rank(ts_rank(ts_delta(net_income_adjusted, 60), 252))",
 
    # 66. EPS growth momentum (smoothed)
    "rank(ts_mean(ts_delta(eps, 60), 120))",
 
    # 67. Earnings growth within subindustry
    "group_neutralize(rank(ts_delta(net_income_adjusted, 252) / (abs(net_income_adjusted) + 0.0001)), subindustry)",
 
    # 68. Revenue growth (top-line + bottom-line combo)
    "rank(ts_delta(revenue, 252) / (abs(ts_delay(revenue, 252)) + 0.0001))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 9 — MULTI-FACTOR COMPOSITES (XGBoost-Inspired Non-linear Combos)
    # Theory: Paper shows XGBoost (non-linear combos) beats single-factor models
    # Strategy: multiply or add orthogonal signals → diversified alpha
    # Combinations follow paper's feature importance hierarchy:
    #   market_value × profitability × efficiency
    # ═══════════════════════════════════════════════════════════════════════
 
    # 69. ROA × B/M (profitability + value — Piotroski-like)
    "rank(net_income_adjusted / (total_assets + 0.0001)) * rank(book_value / (close * shares + 0.0001))",
 
    # 70. Earnings yield × Revenue yield (double value)
    "rank(eps / (close + 0.0001)) * rank(revenue / (close * shares + 0.0001))",
 
    # 71. GPM × ROA (two profitability signals — margins + asset productivity)
    "rank(gross_profit / (revenue + 0.0001)) * rank(net_income_adjusted / (total_assets + 0.0001))",
 
    # 72. ITR × ROA (efficiency + profitability: operationally excellent firms)
    "rank(revenue / (inventory + 0.0001)) * rank(net_income_adjusted / (total_assets + 0.0001))",
 
    # 73. Quality minus price (ROA – P/S): cheap AND profitable
    "rank(net_income_adjusted / (total_assets + 0.0001)) - rank(close * shares / (revenue + 0.0001))",
 
    # 74. OROA × Revenue yield
    "rank(operating_income / (total_assets + 0.0001)) * rank(revenue / (close * shares + 0.0001))",
 
    # 75. B/M × GPM (cheap + high margins = quality value)
    "rank(book_value / (close * shares + 0.0001)) * rank(gross_profit / (revenue + 0.0001))",
 
    # 76. Cash flow yield × ROA (cash quality + profitability)
    "rank(operating_cf / (close * shares + 0.0001)) * rank(net_income_adjusted / (total_assets + 0.0001))",
 
    # 77. Gross profit to assets × Revenue yield (Novy-Marx + value)
    "rank(gross_profit / (total_assets + 0.0001)) * rank(revenue / (close * shares + 0.0001))",
 
    # 78. Triple composite: E/P + CF/P + ROA (Fama-French quality)
    "rank(eps / (close + 0.0001)) + rank(operating_cf / (close * shares + 0.0001)) + rank(net_income_adjusted / (total_assets + 0.0001))",
 
    # 79. Efficiency composite: ITR + WCTR (operational speed)
    "rank(revenue / (inventory + 0.0001)) + rank(revenue / (working_capital + 0.0001))",
 
    # 80. GPM – P/E (quality discount signal: good margins but cheap)
    "rank(gross_profit / (revenue + 0.0001)) - rank(close / (eps + 0.0001))",
 
    # 81. B/M × E/P × ITR (value × earnings yield × efficiency — 3-factor)
    "rank(book_value / (close * shares + 0.0001)) * rank(eps / (close + 0.0001)) * rank(revenue / (inventory + 0.0001))",
 
    # 82. Earnings growth × GPM (growing earnings in high-margin companies)
    "rank(ts_delta(net_income_adjusted, 252) / (abs(net_income_adjusted) + 0.0001)) * rank(gross_profit / (revenue + 0.0001))",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 10 — CROSS-SECTIONAL GROUP COMPOSITES (SUBINDUSTRY-Aware)
    # Theory: Paper uses SUBINDUSTRY neutralization — group operators maximize
    # what survives after neutralization. group_neutralize already does it.
    # ═══════════════════════════════════════════════════════════════════════
 
    # 83. Quality composite within subindustry
    "group_neutralize(rank(eps / (close + 0.0001)) + rank(net_income_adjusted / (total_assets + 0.0001)), subindustry)",
 
    # 84. Value × quality within subindustry
    "group_neutralize(rank(book_value / (close * shares + 0.0001)) * rank(gross_profit / (revenue + 0.0001)), subindustry)",
 
    # 85. Cash quality within subindustry
    "group_neutralize(rank(operating_cf / (close * shares + 0.0001)) * rank(net_income_adjusted / (total_assets + 0.0001)), subindustry)",
 
    # 86. Efficiency × profitability within subindustry
    "group_neutralize(rank(revenue / (inventory + 0.0001)) * rank(net_income_adjusted / (total_assets + 0.0001)), subindustry)",
 
    # 87. Combined quality rank within group
    "group_rank(net_income_adjusted / (total_assets + 0.0001) + gross_profit / (revenue + 0.0001), subindustry)",
 
    # 88. EPS group rank (earnings per share relative to peers)
    "group_rank(eps / (close + 0.0001), subindustry)",
 
    # 89. Cash flow yield group rank
    "group_rank(operating_cf / (close * shares + 0.0001), subindustry)",
 
    # 90. B/M group rank
    "group_rank(book_value / (close * shares + 0.0001), subindustry)",
 
    # 91. Earnings history rank within subindustry
    "group_neutralize(rank(ts_rank(eps, 252)), subindustry)",
 
    # 92. Revenue yield within subindustry
    "group_neutralize(-rank(close * shares / (revenue + 0.0001)), subindustry)",
 
 
    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 11 — TIME-SERIES RANKS ON FUNDAMENTALS (Relative to Own History)
    # Theory: Fundamental improvement relative to own history = positive signal
    # Paper: Gradient boosting captures non-linear temporal patterns in ratios
    # ═══════════════════════════════════════════════════════════════════════
 
    # 93. ROA vs own trailing year
    "rank(ts_rank(net_income_adjusted / (total_assets + 0.0001), 252))",
 
    # 94. Earnings yield vs own trailing year
    "rank(ts_rank(eps / (close + 0.0001), 252))",
 
    # 95. Earnings yield acceleration (change in ts_rank over 2 months)
    "rank(ts_delta(ts_rank(eps, 252), 60))",
 
    # 96. GPM vs own history
    "rank(ts_rank(gross_profit / (revenue + 0.0001), 252))",
 
    # 97. B/M vs own history
    "rank(ts_rank(book_value / (close * shares + 0.0001), 252))",
 
    # 98. Cash flow yield vs own history
    "rank(ts_rank(operating_cf / (close * shares + 0.0001), 252))",
 
    # 99. ITR vs own trailing year
    "rank(ts_rank(revenue / (inventory + 0.0001), 252))",
 
    # 100. ROA acceleration: is profitability improvement speeding up?
    "rank(ts_delta(ts_rank(net_income_adjusted / (total_assets + 0.0001), 252), 60))",
    ]

    alpha_list = [generate_alpha(x) for x in k]

    simulate_alpha_list(s, alpha_list)


if __name__ == "__main__":
    main()
