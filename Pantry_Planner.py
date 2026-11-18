"""ADAPT-D initial planner: non-retail (pantry) coverage model for Detroit.

Run:  streamlit run app.py
To add it to an existing multi-page Streamlit app, copy this file into pages/
and keep the data/ folder next to it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

DATA = Path(__file__).parent / "data"
SCENARIOS = ["Low demand / lean supply", "Low demand / abundant supply",
             "High demand / lean supply", "High demand / abundant supply"]

st.set_page_config(page_title="ADAPT-D pantry coverage planner", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    tracts = pd.read_csv(DATA / "tracts.csv", dtype={"geoid": str})
    existing = pd.read_csv(DATA / "existing_sites.csv")
    cand = pd.read_csv(DATA / "candidate_sites.csv")
    return tracts, existing, cand


def miles(lat1, lon1, lat2, lon2):
    """Straight-line (haversine) distance matrix in miles."""
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------ model
@st.cache_data(show_spinner=False)
def solve(demand_lo, demand_hi, supply_lo, supply_hi, lbs_per_visit, reach, circuity,
          hh_per_hour, hours_cap, new_hours, n_min, n_max, priority, walkup_only,
          include_hot_meal, cand_types, allow_new):
    tracts, existing, cand = load()
    ex = existing.copy()
    if not include_hot_meal:
        ex = ex[ex.subtype.str.strip() != "Hot Meal"]
    hrs = ex.weekly_hours_walkup if walkup_only else ex.weekly_hours
    ex = ex.assign(capacity=np.minimum(hrs, hours_cap) * hh_per_hour)
    ca = cand[cand.subtype.isin(cand_types)].assign(capacity=new_hours * hh_per_hour) if allow_new else cand.iloc[0:0].assign(capacity=0.0)
    sites = pd.concat([ex, ca], ignore_index=True)
    nE, nC, nS, nT = len(ex), len(ca), len(ex) + len(ca), len(tracts)

    dist = miles(tracts.lat.values, tracts.lon.values, sites.lat.values, sites.lon.values) * circuity
    ti, sj = np.nonzero((dist <= reach) & (sites.capacity.values[None, :] > 0))
    nP = len(ti)                                   # reachable tract-site pairs

    base = tracts.food_insecure_hh.values
    dem = {0: base * demand_lo, 1: base * demand_lo, 2: base * demand_hi, 3: base * demand_hi}
    sup = {0: supply_lo, 1: supply_hi, 2: supply_lo, 3: supply_hi}
    w = {"Coverage: most households served": np.ones(nT),
         "Need: highest food-insecurity rate first": 1 + tracts.food_insecure_rate.values / tracts.food_insecure_rate.max(),
         "Car-limited: fewest vehicles first": 1 + tracts.no_vehicle_share.values / tracts.no_vehicle_share.max()}[priority]

    def run(open_new, enforce_min=True):
        """open_new: None -> optimise candidate choice; array -> fixed choice."""
        nY = 4 * nP                                # y[s, pair] households served per week
        nV = nY + nC                               # + x[candidate] open or not
        c = np.zeros(nV)
        for s in range(4):
            c[s * nP:(s + 1) * nP] = -w[ti] / 4.0  # maximise average over the four scenarios
        c[nY:] = 1e-3                              # tiny cost so only useful sites open
        rows, cols, vals, ub, r = [], [], [], [], 0
        for s in range(4):
            off = s * nP
            # demand: each tract cannot receive more than it seeks
            rows += list(r + ti); cols += list(off + np.arange(nP)); vals += [1.0] * nP
            ub += list(dem[s]); r += nT
            # capacity: each site serves at most capacity (x capacity if candidate)
            rows += list(r + sj); cols += list(off + np.arange(nP)); vals += [1.0] * nP
            for k in range(nC):
                rows.append(r + nE + k); cols.append(nY + k); vals.append(-ca.capacity.values[k])
            ub += list(ex.capacity.values) + [0.0] * nC; r += nS
            # supply: total households served limited by pounds available
            rows += [r] * nP; cols += list(off + np.arange(nP)); vals += [1.0] * nP
            ub.append(sup[s] / lbs_per_visit); r += 1
        lb_con = [-np.inf] * r
        if nC:                                     # number of new locations between min and max
            rows += [r] * nC; cols += list(nY + np.arange(nC)); vals += [1.0] * nC
            lb_con.append(min(n_min, nC) if enforce_min else 0); ub.append(min(n_max, nC)); r += 1
        A = coo_matrix((vals, (rows, cols)), shape=(r, nV)).tocsr()
        hi = np.concatenate([np.full(nY, np.inf), np.ones(nC)])
        lo = np.zeros(nV)
        if open_new is not None:
            lo[nY:] = hi[nY:] = open_new
        integ = np.concatenate([np.zeros(nY), np.ones(nC)])
        res = milp(c, constraints=LinearConstraint(A, lb_con, ub), bounds=Bounds(lo, hi),
                   integrality=integ, options={"time_limit": 60, "mip_rel_gap": 0.005})
        if res.x is None:
            return None
        y = res.x[:nY].reshape(4, nP)
        return y, np.round(res.x[nY:]).astype(int)

    if nC and n_max > 0:
        b = run(np.zeros(nC), enforce_min=False)   # baseline: existing network only
        o = run(None)
    else:
        b = o = run(None, enforce_min=False)
    if b is None or o is None:
        return None
    (yb, _), (yo, xo) = b, o

    def per_tract(y):
        return np.array([np.bincount(ti, weights=y[s], minlength=nT) for s in range(4)])

    def per_site(y):
        return np.array([np.bincount(sj, weights=y[s], minlength=nS) for s in range(4)])

    reachable_now = np.zeros(nT, bool); reachable_now[ti[sj < nE]] = True
    open_idx = np.concatenate([np.arange(nE), nE + np.nonzero(xo)[0]])
    reachable_new = np.zeros(nT, bool); reachable_new[ti[np.isin(sj, open_idx)]] = True
    return dict(tracts=tracts, sites=sites, nE=nE, x=xo, dem=np.array([dem[s] for s in range(4)]),
                tb=per_tract(yb), to=per_tract(yo), sb=per_site(yb), so=per_site(yo),
                reach_now=reachable_now, reach_new=reachable_new, pairs=nP)


# ------------------------------------------------------------------ sidebar
tracts, existing, cand = load()
st.title("ADAPT-D pantry coverage planner")
st.caption("Non-retail food network only (pantries, distributions, hot meals). Chooses new distribution "
           "locations to maximise food-insecure households served within reach. Initial version: "
           "inputs are public data and stated assumptions, not observed pantry use.")

sb = st.sidebar
sb.header("Locations")
allow_new = sb.checkbox("Allow new locations", True)
n_min, n_max = sb.slider("New locations to open (min, max)", 0, 60, (0, 15), disabled=not allow_new)
types = sorted(cand.subtype.unique())
cand_types = sb.multiselect("Candidate host types", types, default=types, disabled=not allow_new)
new_hours = sb.slider("Hours per week at each new location", 1.0, 12.0, 3.0, 0.5, disabled=not allow_new)

sb.header("Demand")
d_lo, d_hi = sb.slider("Food-insecure households seeking pantry food each week (%), low and high", 1, 40, (6, 21))
sb.caption("Default 6% = 26% use a pantry, once a month. 21% = 45.5% use one, twice a month (USDA ERS use rates; visit frequency assumed).")

sb.header("Supply")
s_lo, s_hi = sb.slider("Donated food available (thousand lb per week), lean and abundant", 25, 1000, (150, 450), 25)
lbs = sb.slider("Pounds per household visit", 15, 60, 32)
sb.caption("Supply range and pounds per visit are placeholders. Replace with operator figures.")

sb.header("Reach and capacity")
reach = sb.slider("Maximum travel distance (miles)", 0.25, 3.0, 0.5, 0.25)
circ = sb.slider("Street-network detour factor", 1.0, 1.6, 1.3, 0.05,
                 help="Straight-line distance is multiplied by this to approximate walking distance on streets.")
hh_hr = sb.slider("Households served per site-hour", 5, 80, 25, 5)
h_cap = sb.slider("Cap on weekly hours counted per existing site", 2, 45, 12,
                  help="Stops appointment-only offices open all week from counting as 40 hours of distribution.")
walkup = sb.checkbox("Count only walk-up / open hours at existing sites", False,
                     help="Excludes drive-up-only, appointment-only and age-restricted sessions.")
hot = sb.checkbox("Include hot-meal-only sites", False)

sb.header("Priority")
priority = sb.radio("Objective", ["Coverage: most households served",
                                  "Need: highest food-insecurity rate first",
                                  "Car-limited: fewest vehicles first"])

with st.spinner("Solving..."):
    R = solve(d_lo / 100, d_hi / 100, s_lo * 1000, s_hi * 1000, lbs, reach, circ, hh_hr, h_cap, new_hours,
              n_min, n_max, priority, walkup, hot, tuple(cand_types), allow_new)
if R is None:
    st.error("No feasible plan. The minimum number of new locations may exceed the candidates selected.")
    st.stop()

T, S, nE = R["tracts"], R["sites"], R["nE"]
scen = st.radio("Scenario shown on the map and tables", SCENARIOS, index=2, horizontal=True)
k = SCENARIOS.index(scen)

# ------------------------------------------------------------------ headline
dem_k, base_k, opt_k = R["dem"][k].sum(), R["tb"][k].sum(), R["to"][k].sum()
fi_total = T.food_insecure_hh.sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Households seeking food / week", f"{dem_k:,.0f}")
c2.metric("Served by current network", f"{base_k:,.0f}", f"{base_k / dem_k:.0%} of demand", delta_color="off")
c3.metric("Served with new locations", f"{opt_k:,.0f}", f"+{max(opt_k - base_k, 0):,.0f} households")
c4.metric("New locations opened", int(R["x"].sum()), f"of {n_max} allowed" if allow_new else "none allowed", delta_color="off")
c1, c2 = st.columns(2)
c1.metric("Food-insecure households with any site in reach (current)",
          f"{T.food_insecure_hh[R['reach_now']].sum() / fi_total:.0%}")
c2.metric("With new locations", f"{T.food_insecure_hh[R['reach_new']].sum() / fi_total:.0%}")

st.subheader("All four scenarios, one set of locations")
tab = pd.DataFrame({"Scenario": SCENARIOS,
                    "Demand (households/week)": R["dem"].sum(1),
                    "Supply limit (households/week)": [s_lo * 1000 / lbs, s_hi * 1000 / lbs, s_lo * 1000 / lbs, s_hi * 1000 / lbs],
                    "Served, current": R["tb"].sum(1), "Served, with new": R["to"].sum(1)})
tab["Gain"] = tab["Served, with new"] - tab["Served, current"]
tab["% of demand met, current"] = tab["Served, current"] / tab["Demand (households/week)"]
tab["% of demand met, with new"] = tab["Served, with new"] / tab["Demand (households/week)"]
st.dataframe(tab.style.format({c: "{:,.0f}" for c in tab.columns[1:6]} | {c: "{:.0%}" for c in tab.columns[6:]}),
             hide_index=True)
st.caption("The model picks one set of new locations that does best on average across the four scenarios. "
           "If 'served' equals the supply limit, food is the binding constraint and more sites will not help.")

# ------------------------------------------------------------------ map
tr = T.assign(demand=R["dem"][k], served=R["to"][k])
tr["unmet"] = (tr.demand - tr.served).clip(lower=0)
tr["unmet_share"] = np.where(tr.demand > 0, tr.unmet / tr.demand, 0)
tr["radius"] = 60 + 22 * np.sqrt(tr.demand)
tr["color"] = tr.unmet_share.map(lambda u: [int(60 + 180 * u), int(150 - 110 * u), int(170 - 130 * u), 150])
tr["tip"] = tr.apply(lambda r: f"Tract {r.geoid}: {r.demand:.0f} seeking, {r.served:.0f} served, {r.unmet_share:.0%} unmet", axis=1)
ex_map = S.iloc[:nE].assign(served=R["so"][k][:nE])
ex_map["tip"] = ex_map.apply(lambda r: f"{r['name']} (existing): {r.served:.0f} of {r.capacity:.0f} households/week", axis=1)
new_map = S.iloc[nE:][R["x"] == 1].assign(served=R["so"][k][nE:][R["x"] == 1])
new_map["tip"] = new_map.apply(lambda r: f"NEW: {r['name']} ({r.subtype}): {r.served:.0f} households/week", axis=1)
layers = [
    pdk.Layer("ScatterplotLayer", tr, get_position="[lon, lat]", get_radius="radius", get_fill_color="color", pickable=True),
    pdk.Layer("ScatterplotLayer", ex_map, get_position="[lon, lat]", get_radius=90, get_fill_color=[20, 20, 20, 230],
              get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=1, pickable=True),
    pdk.Layer("ScatterplotLayer", new_map, get_position="[lon, lat]", get_radius=170, get_fill_color=[255, 200, 0, 240],
              get_line_color=[0, 0, 0], stroked=True, line_width_min_pixels=2, pickable=True),
]
st.subheader("Map")
st.pydeck_chart(pdk.Deck(layers=layers, map_style="light", tooltip={"text": "{tip}"},
                         initial_view_state=pdk.ViewState(latitude=42.375, longitude=-83.10, zoom=10.3)))
st.caption("Circles = census tracts, sized by households seeking food; teal = demand met, red = unmet. "
           "Black dots = existing sites. Yellow dots = new locations chosen by the model.")

# ------------------------------------------------------------------ tables
a, b = st.columns(2)
with a:
    st.subheader("New locations chosen")
    out = new_map[["name", "subtype", "address", "zip", "served", "capacity"]].rename(columns={
        "name": "Location", "subtype": "Type", "address": "Address", "zip": "ZIP",
        "served": "Households/week", "capacity": "Capacity"}).sort_values("Households/week", ascending=False)
    st.dataframe(out.style.format({"Households/week": "{:,.0f}", "Capacity": "{:,.0f}", "ZIP": "{:.0f}"}, na_rep=""),
                 hide_index=True)
    st.download_button("Download new locations (CSV)", out.to_csv(index=False), "adaptd_new_locations.csv")
with b:
    st.subheader("Tracts with the most unmet demand")
    gap = tr.sort_values("unmet", ascending=False).head(25)[
        ["geoid", "food_insecure_rate", "no_vehicle_share", "demand", "served", "unmet"]].rename(columns={
        "geoid": "Tract", "food_insecure_rate": "Food insecure", "no_vehicle_share": "No vehicle",
        "demand": "Seeking/week", "served": "Served", "unmet": "Unmet"})
    st.dataframe(gap.style.format({"Food insecure": "{:.0%}", "No vehicle": "{:.0%}", "Seeking/week": "{:,.0f}",
                                   "Served": "{:,.0f}", "Unmet": "{:,.0f}"}), hide_index=True)
    st.download_button("Download all tract results (CSV)", tr.drop(columns=["radius", "color", "tip"]).to_csv(index=False),
                       "adaptd_tract_results.csv")

with st.expander("Data, model and limits"):
    st.markdown(f"""
**Model.** Mixed-integer program (HiGHS). Chooses which candidate hosts to open and assigns households in each
tract to sites within reach, subject to site capacity, total food supply and tract demand. Existing sites stay open.
{R['pairs']:,} reachable tract-site pairs in this run.

**Data in the model**
- Demand points: {len(T)} Detroit census tracts. Households from ACS 2020-24; adult food-insecurity rate from CDC PLACES (2025 release), applied to households.
- Existing sites: {len(existing)} locations on the City of Detroit Food Pantry Finder (layer last edited Nov 2025), hours hand-coded.
- Candidate hosts: {len(cand)} sites: open schools (City of Detroit, Nov 2026), public libraries (2024), recreation centers (2016 layer, verify).

**Assumptions to replace with real figures**
- Share of food-insecure households seeking food per week, pounds per visit, total supply, households served per site-hour.
- Reach is straight-line distance times a detour factor from the tract centre, not walk or bus time by hour of day.

**Not included yet**
- Gleaners and Forgotten Harvest mobile stops and partner pantries missing from the City map.
- Days and hours of new sessions, host availability, opening cost, and eligibility rules.
""")
