# FLUXA parameter origins

Every number in `system_6bus.json` is listed here with its origin. The
classification is deliberately blunt:

- **[ASSUMED]** — chosen by the model author to be physically plausible for a
  small island system. Not taken from any measurement or published dataset.
- **[CONVENTIONAL]** — a standard value or standard practice in power-system
  engineering, used because it is the usual choice, not because it was
  measured here.
- **[DERIVED]** — computed from other parameters in this file.

**No parameter in this file is an observation of a real power system.** The
6-bus system is synthetic. It is sized and shaped so that the experiments
exercise real mechanisms (ramp limits binding, network congestion, reserve
adequacy, load shedding), not so that it resembles any particular grid.

## System scale

| Parameter | Value | Origin |
|---|---|---|
| `peak_load_mw` | 420.0 | [ASSUMED] Island-scale system, large enough that a single 220 MW unit tripping is a material contingency. |
| `S_BASE_MVA` (`fluxa/units.py`) | 100.0 | [CONVENTIONAL] Standard per-unit base. |
| `F_NOMINAL_HZ` | 50.0 | [CONVENTIONAL] 50 Hz system. |

## Buses

| Parameter | Origin |
|---|---|
| `load_share` (0.10 / 0 / 0 / 0.35 / 0.40 / 0.15) | [ASSUMED] Load concentrated at an industrial park and an urban centre; the two renewable buses carry no load, which is typical of remote generation siting. Shares sum to 1.0 (validated). |
| `is_slack` on B1 | [CONVENTIONAL] The angle reference is placed at the largest synchronous unit's bus. |

## Lines

| Parameter | Origin |
|---|---|
| `reactance_pu` (0.04–0.09) | [ASSUMED] Plausible per-unit reactances for sub-transmission corridors on a 100 MVA base. Chosen so that solved angle spreads stay under ~0.35 rad, which keeps the DC linearisation self-consistent (asserted by `test_angles_are_small_so_the_dc_linearisation_is_self_consistent`). |
| `capacity_mw` (110–200) | [ASSUMED] Sized so the baseline runs at 0.84 maximum utilisation — tight enough that congestion is reachable, loose enough that the baseline is secure. |
| Resistance | Not modelled. [ASSUMED] r ≪ x; see the DC assumptions in `fluxa/network.py`. |

## Generators

| Parameter | Value | Origin |
|---|---|---|
| `GAS_CCGT_1.p_max_mw` | 220.0 | [ASSUMED] ~52% of peak load, so its loss is a severe but survivable contingency. |
| `GAS_CCGT_1.p_min_mw` | 40.0 | [CONVENTIONAL] CCGTs have a minimum stable load, typically 30–50% of rating; 40 MW (18%) is on the low side, chosen because FLUXA has no commitment binary and a higher floor would force uneconomic must-run output. |
| `GAS_CCGT_1.marginal_cost_usd_per_mwh` | 48.0 | [ASSUMED] Fuel plus variable O&M for an efficient CCGT at a mid-range gas price. |
| `GAS_CCGT_1.ramp_mw_per_min` | 4.0 | [CONVENTIONAL] ≈1.8%/min of rating, a realistic sustained CCGT ramp. Yields 20 MW per 5-minute step, which binds in practice (see `test_ramp_limit_binds_when_unit_is_available`). |
| `GAS_CCGT_1.co2_tonnes_per_mwh` | 0.36 | [CONVENTIONAL] Typical CCGT direct combustion intensity. |
| `GAS_GT_2.p_max_mw` | 90.0 | [ASSUMED] Peaking capacity. |
| `GAS_GT_2.p_min_mw` | 0.0 | [ASSUMED] Set to zero specifically so the peaker can sit at zero without a commitment binary. This is a modelling convenience, stated as such. |
| `GAS_GT_2.marginal_cost_usd_per_mwh` | 92.0 | [ASSUMED] Open-cycle heat rate, roughly 1.9× the CCGT. |
| `GAS_GT_2.ramp_mw_per_min` | 9.0 | [CONVENTIONAL] 10%/min, a fast-start aeroderivative figure. |
| `GAS_GT_2.co2_tonnes_per_mwh` | 0.55 | [CONVENTIONAL] Open-cycle intensity. |
| `SOLAR_PV_1.p_max_mw` | 180.0 | [ASSUMED] 43% of peak load as PV nameplate. |
| `WIND_1.p_max_mw` | 150.0 | [ASSUMED] 36% of peak load as wind nameplate. |
| Variable-unit `marginal_cost_usd_per_mwh` (1.0 / 2.0) | [ASSUMED] Non-zero variable O&M so the merit order is strictly ordered and the LP has a unique dispatch preference among zero-fuel units. |

Combined VRE nameplate (330 MW) is 79% of peak load. This is deliberately
*not* an overbuild: on this mix, curtailment from pure energy surplus is
structurally near zero, which the experiments report rather than conceal
(scenarios G and H produce curtailment through network congestion instead).

## Battery

| Parameter | Value | Origin |
|---|---|---|
| `energy_capacity_mwh` | 320.0 | [ASSUMED] 4-hour duration at full discharge power. |
| `p_charge_max_mw` / `p_discharge_max_mw` | 80.0 | [ASSUMED] 19% of peak load. |
| `charge_efficiency` / `discharge_efficiency` | 0.95 / 0.95 | [CONVENTIONAL] One-way efficiencies giving a 0.9025 round trip, in the normal range for a modern Li-ion BESS. |
| `soc_min` / `soc_max` | 0.10 / 0.95 | [CONVENTIONAL] Depth-of-discharge limits used to protect cycle life. |
| `soc_initial` | 0.50 | [ASSUMED] Mid-window start so the first hours are not dominated by the initial condition. |
| `cycle_cost_usd_per_mwh` | 4.0 | [ASSUMED] Degradation plus O&M charged on throughput. |

## Interconnection

| Parameter | Value | Origin |
|---|---|---|
| `import_max_mw` / `export_max_mw` | 100.0 | [ASSUMED] 24% of peak load of firm tie capacity. |
| `import_price_usd_per_mwh` | 75.0 | [ASSUMED] Above the CCGT's cost and below the peaker's, so imports sit between them in merit order. |
| `export_price_usd_per_mwh` | 25.0 | [ASSUMED] Below every internal generator's cost, so export is a surplus sink rather than a profit centre. |

## Economics

| Parameter | Value | Origin |
|---|---|---|
| `value_of_lost_load_usd_per_mwh` | 10000.0 | [CONVENTIONAL] VOLL estimates in the literature span roughly 3,000–30,000 USD/MWh by customer class; 10,000 is a common mid-range planning figure. Its role here is to make shedding the last resort, and the dispatch is insensitive to its exact value as long as it exceeds every other cost by orders of magnitude. |
| `curtailment_cost_usd_per_mwh` | 15.0 | [ASSUMED] Stands in for a PPA curtailment payment or lost renewable certificate value. |
| `overload_penalty_usd_per_mwh` | 2000.0 | [ASSUMED] A **soft-constraint penalty, not a price.** It exists so a network violation is measured rather than making the LP infeasible. Set above every generation cost and below VOLL, so the dispatch prefers overloading a line to shedding load — which matches operator behaviour for a short-duration emergency. Excluded from reported operating cost. |
| `reserve_shortfall_penalty_usd_per_mwh` | 500.0 | [ASSUMED] Soft-constraint penalty, as above. Set above the peaker's cost so reserve is procured when it can be, and below the overload penalty so reserve is given up before a line is overloaded. |
| `stored_energy_value_usd_per_mwh` | 48.0 | [DERIVED] Equal to the CCGT marginal cost. This is the terminal value of stored energy at the end of each receding horizon; without it a finite-horizon LP empties the battery at the window edge as an artefact. Setting it to the mid-merit cost means the battery will not arbitrage against the CCGT and acts only as a ramp-bridging and peak-shaving resource — a consequence that is visible in the results and discussed in the report. |
| `co2_price_usd_per_tonne` | 0.0 | [ASSUMED] No carbon price in the base case; emissions are accounted but not priced. |

## Reserve policy

| Parameter | Value | Origin |
|---|---|---|
| `load_fraction` | 0.08 | [CONVENTIONAL] A load-proportional contingency reserve term. |
| `renewable_fraction` | 0.15 | [CONVENTIONAL] A variability term proportional to instantaneous VRE availability. |
| `duration_h` | 0.25 | [ASSUMED] Reserve must be sustainable for 15 minutes. This is what turns a battery's power headroom into an energy-tested contribution. |

The requirement is a deterministic linear proxy, not a probabilistic
reserve-adequacy calculation.

## Frequency proxy

| Parameter | Value | Origin |
|---|---|---|
| `nominal_hz` | 50.0 | [CONVENTIONAL] |
| `response_characteristic_mw_per_hz` | 30.0 | [ASSUMED] A single aggregate frequency-response characteristic for a ~420 MW island. Plausible order of magnitude; not fitted to anything. |
| `alarm_deviation_hz` | 0.20 | [CONVENTIONAL] A typical operational frequency alarm band. |
| `FREQUENCY_PROXY_VALID_DEVIATION_HZ` (`fluxa/state.py`) | 2.0 | [ASSUMED] Beyond ±2 Hz the linear steady-state proxy is outside any regime a real system would survive without under-frequency load shedding and probable cascading. States beyond this are flagged `frequency_proxy_in_range=False` and counted separately; they indicate severity ordering, not predicted frequency. |

## Exogenous profiles (`fluxa/profiles.py`)

All profile shape parameters are **[ASSUMED]**. The load curve is a
double-peaked daily shape with a weekday/weekend factor; solar follows a
clear-sky sinusoid between fixed sunrise and sunset hours modulated by a
seeded AR(1) cloud process; wind is a mean plus diurnal and synoptic
sinusoids plus a seeded AR(1) term. AR(1) is used rather than white noise
because white noise would be averaged away by the dispatch and would
understate ramping stress.

These are synthetic profiles chosen to be physically plausible. **They are
not measurements of any real grid's load, irradiance or wind resource**, and
no statistic derived from them should be read as an empirical grid statistic.
