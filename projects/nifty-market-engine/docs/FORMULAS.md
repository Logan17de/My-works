# Signal formulas

This engine treats support/resistance as a **context**, then classifies the interaction as breakout, reversal, or uncertain. It does not assume that a touched level must reverse.

All default constants are hypotheses for paper testing, not claims of profitability. They are centralized in `StrategyParams` and should be calibrated only on out-of-sample results.

## 1. Constituent activity and direction

For constituent `i` over sampling interval `dt`:

```text
volume_rate_i = max(V_i(t) - V_i(t-1), 0) / dt
RVOL_i        = volume_rate_i / baseline_volume_rate_i
activity_i    = log(1 + clip(RVOL_i, 0, RVOL_cap)) / log(1 + RVOL_cap)
return_bps_i  = (P_i(t) / P_i(t-1) - 1) * 10,000
direction_i   = tanh(return_bps_i / direction_scale_bps)
```

`baseline_volume_rate_i` is the median expected rate for that stock and time-of-day bucket computed from **prior sessions only**:

```text
baseline_volume_rate_i(minute) = median(volume_rate_i for the same minute over prior sessions)
```

This avoids comparing raw share counts across companies and prevents current-session future leakage. The helper in `baselines.py` implements this robust median baseline.

## 2. Weighted cash-market pressure

Let `w_i` be the normalized NIFTY weight (or equal weights only for early testing):

```text
pressure = sum(w_i * activity_i * direction_i) / sum(w_i * activity_i)
breadth  = (advancers - decliners) / N
```

Participation is a normalized inverse Herfindahl concentration measure over weighted activity:

```text
share_i       = (w_i * activity_i) / sum(w_j * activity_j)
HHI           = sum(share_i^2)
participation = (1 - HHI) / (1 - 1/N)
```

This deliberately down-rates a move driven by only one or two constituents.

```text
cash_raw   = 0.75 * pressure + 0.25 * breadth
cash_score = cash_raw * (0.55 + 0.45 * participation)
```

## 3. Signed volume acceleration

```text
accel_i = tanh((volume_rate_i - previous_volume_rate_i) / previous_volume_rate_i)
signed_volume_accel = sum(w_i * accel_i * direction_i)
```

A sudden increase in volume only becomes bullish/bearish after being combined with price direction. High volume alone is not labelled "institutional buying" because aggregate market data does not identify the trader.

## 4. NIFTY futures confirmation

```text
future_direction = tanh(future_return_bps / future_direction_scale_bps)
future_RVOL      = future_volume_rate / future_baseline_volume_rate
future_activity  = activity(future_RVOL)
OI_change_pct    = (OI_t / OI_t-1 - 1) * 100
OI_confirmation  = future_direction * tanh(OI_change_pct / OI_scale_pct)

basis_t          = future_price_t - spot_price_t
basis_change_bps = ((basis_t - basis_t-1) / spot_price_t) * 10,000
basis_component  = tanh(basis_change_bps / basis_scale_bps)
```

The OI term is a confirmation term, not a rigid "long buildup/short buildup" rule. Rising OI amplifies the current futures direction; falling OI reduces that confirmation.

```text
directional_with_activity = future_direction * (0.50 + 0.50 * future_activity)
futures_score =
    0.45 * directional_with_activity
  + 0.30 * OI_confirmation
  + 0.25 * basis_component
```

## 5. Combined directional state

```text
combined_direction = 0.60 * cash_score + 0.40 * futures_score
```

The result is clipped to `[-1, +1]`.

## 6. Level-event classifier

For resistance, breakout direction is `+1`; for support it is `-1`.

```text
signed_distance_bps = breakout_direction * ((spot / level) - 1) * 10,000
penetration = clip(max(signed_distance_bps, 0) / breakout_penetration_bps, 0, 1)
rejection   = clip(max(-signed_distance_bps, 0) / rejection_depth_bps, 0, 1)
persistence = clip(seconds_beyond_level / persistence_target_seconds, 0, 1)
```

Directional alignment:

```text
breakout_direction_score = max(breakout_direction * combined_direction, 0)
reversal_direction_score = max(-breakout_direction * combined_direction, 0)
```

Acceleration uses both the change in combined score and signed constituent volume acceleration. For each event direction, the engine takes the stronger positive confirmation:

```text
breakout_acceleration = max(clip(breakout_dir * Δcombined, 0, 1),
                            clip(breakout_dir * signed_volume_accel, 0, 1))
reversal_acceleration = max(clip(-breakout_dir * Δcombined, 0, 1),
                            clip(-breakout_dir * signed_volume_accel, 0, 1))
```

```text
breakout_score =
    0.40 * breakout_direction_score
  + 0.20 * penetration
  + 0.15 * persistence
  + 0.15 * participation
  + 0.10 * breakout_acceleration

reversal_score =
    0.40 * reversal_direction_score
  + 0.20 * rejection
  + 0.15 * (1 - persistence)
  + 0.15 * participation
  + 0.10 * reversal_acceleration
```

Classification is intentionally three-way:

```text
BREAKOUT if breakout_score >= threshold and leads reversal_score by decision_margin
REVERSAL if reversal_score >= threshold and leads breakout_score by decision_margin
UNCERTAIN otherwise
```

`UNCERTAIN` is a first-class state and means no order.

## 7. Option contract selection

Greeks do **not** decide market direction. They select the contract after direction has been decided.

Candidates are first filtered to the requested CE/PE type and a configurable absolute delta band, default `0.48-0.68` with target `0.58`.

Within the eligible set, volume and OI are log-scaled then min-max normalized. Liquidity is:

```text
liquidity = 0.55 * normalized_log_volume + 0.45 * normalized_log_OI
```

Each surviving contract receives normalized components:

```text
contract_score =
    0.35 * delta_fit
  + 0.30 * liquidity(volume, OI)
  + 0.15 * low_theta_cost
  + 0.10 * lower_relative_IV
  + 0.10 * gamma_score
```

If bid/ask is available, contracts above the maximum spread percentage are rejected.

## 8. Position sizing and circuit breakers

For long options, the conservative maximum premium-at-risk budget is:

```text
max_premium_risk = account_equity * risk_per_trade_pct
cost_per_lot     = option_ltp * lot_size
lots             = floor(max_premium_risk / cost_per_lot)
quantity         = lots * lot_size
```

The risk engine can veto a technically valid signal for stale data, insufficient confidence, an existing position, cooldown, trade-count limit, consecutive-loss limit, or daily-loss limit.

## 9. Data-quality gate

The default live risk gate requires at least **45 of 50** NIFTY constituents and rejects a snapshot older than **30 seconds**. These are configurable. A high score computed from incomplete or stale inputs is not allowed to become an order.

## 10. Parameter ownership

Every tunable coefficient is stored in `StrategyParams`, including the subweights used for option volume/OI liquidity and the level-touch memory window. Parameter groups are validated to sum to 1 where they form a weighted score.
