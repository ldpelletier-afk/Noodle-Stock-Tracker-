"""Portfolio risk analytics.

Self-contained numerical module — no Streamlit UI code, no ollama, no Chroma.
Everything here is pure pandas/numpy + a thin yfinance fetcher wrapped in
Streamlit's cache so the UI layer can re-render without re-downloading.

Public surface consumed by the UI:
    fetch_price_history(tickers, period)    -> DataFrame of close prices
    compute_returns(prices)                 -> DataFrame of log returns
    build_portfolio_weights(holdings, px)   -> dict[ticker -> weight]
    portfolio_return_series(returns, w)     -> Series of portfolio returns
    historical_var(series, conf)            -> (var, cvar)  (positive = loss)
    parametric_var(series, conf)            -> (var, cvar)
    max_drawdown(series)                    -> dict{max_dd, peak, trough, days}
    annualized_volatility(series)           -> float
    sharpe_ratio / sortino_ratio / calmar_ratio
    beta_alpha(asset, market)               -> (beta, alpha_ann, r2)
    rolling_beta(asset, market, window)     -> Series
    correlation_matrix(returns)             -> DataFrame
    factor_exposure(port_ret, factor_df)    -> dict{loadings, r2, ...}
    concentration(weights)                  -> dict{hhi, top, effective_n}
    portfolio_risk_report(holdings, prices) -> dict — one-shot aggregator

All loss-based metrics (VaR, CVaR, drawdown) are returned as **positive
magnitudes representing downside**, matching common risk-dashboard
convention. Sharpe/Sortino/Calmar are annualized assuming 252 trading days.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

TRADING_DAYS = 252

# Benchmark ETFs used for factor decomposition. Long/short pairs are
# constructed inside build_factor_returns() to approximate the Fama-French
# style factors without pulling Ken French's zip file:
#   MKT  = SPY returns                          (market)
#   SMB  = IWM - SPY                            (small minus big)
#   HML  = VTV - VUG                            (value minus growth)
#   MOM  = MTUM - SPY                           (momentum premium, rough)
#   RATES = TLT                                 (long-duration rates beta)
_FACTOR_ETFS = ["SPY", "IWM", "VTV", "VUG", "MTUM", "TLT"]


# ---------- Price fetch ----------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    """Batched adjusted-close download. Returns wide DataFrame; missing
    tickers silently dropped. Cache key uses the tuple form so the same set
    of tickers hits cache regardless of iteration order when callers sort."""
    real = [t for t in tickers if t and t.upper() != "CASH"]
    if not real:
        return pd.DataFrame()
    try:
        data = yf.download(
            " ".join(real),
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    # yfinance returns either a MultiIndex (many tickers) or flat (one ticker).
    closes: dict[str, pd.Series] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in real:
            if t in data.columns.get_level_values(0):
                sub = data[t]
                if "Close" in sub.columns:
                    s = sub["Close"].dropna()
                    if len(s) > 5:
                        closes[t] = s
    else:
        if "Close" in data.columns:
            s = data["Close"].dropna()
            if len(s) > 5 and len(real) == 1:
                closes[real[0]] = s

    if not closes:
        return pd.DataFrame()
    df = pd.DataFrame(closes).sort_index().dropna(how="all")
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. Rows with any NaN dropped so every column aligns
    to the same trading calendar (needed for covariance/regression)."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    r = np.log(prices / prices.shift(1))
    return r.dropna(how="any")


# ---------- Weights & portfolio series ----------

def build_portfolio_weights(
    holdings: dict, live_prices: dict[str, dict]
) -> dict[str, float]:
    """Market-value weights over *risky* assets only. CASH (or any ticker
    with unknown price) is excluded from the risk series — it contributes
    zero variance but distorts the denominator if included at 1.0."""
    mv: dict[str, float] = {}
    for ticker, pos in (holdings or {}).items():
        if not ticker or ticker.upper() == "CASH":
            continue
        qty = float(pos.get("quantity", 0) or 0)
        px = (live_prices or {}).get(ticker, {}).get("price")
        if qty <= 0 or not px:
            continue
        mv[ticker] = qty * float(px)
    total = sum(mv.values())
    if total <= 0:
        return {}
    return {t: v / total for t, v in mv.items()}


def portfolio_return_series(
    returns: pd.DataFrame, weights: dict[str, float]
) -> pd.Series:
    """Weighted sum over the columns present in both `returns` and
    `weights`. Any weight on a ticker missing from returns is dropped and
    the remaining weights are re-normalized so the series is still a
    proper weighted average."""
    if returns is None or returns.empty or not weights:
        return pd.Series(dtype=float)
    common = [t for t in weights if t in returns.columns]
    if not common:
        return pd.Series(dtype=float)
    w = np.array([weights[t] for t in common], dtype=float)
    s = w.sum()
    if s <= 0:
        return pd.Series(dtype=float)
    w = w / s
    return (returns[common] * w).sum(axis=1)


# ---------- Loss metrics ----------

def historical_var(
    series: pd.Series, confidence: float = 0.95
) -> tuple[float, float]:
    """Non-parametric VaR + Expected Shortfall.

    Returns (VaR, CVaR) as **positive loss magnitudes**. Daily horizon —
    scale with sqrt(h) if the caller wants an N-day figure.
    """
    if series is None or series.empty:
        return (float("nan"), float("nan"))
    alpha = 1.0 - confidence
    q = float(np.quantile(series, alpha))
    tail = series[series <= q]
    cvar = float(tail.mean()) if len(tail) else q
    return (abs(q), abs(cvar))


def parametric_var(
    series: pd.Series, confidence: float = 0.95
) -> tuple[float, float]:
    """Gaussian VaR + ES.  Closed-form, so useful as a sanity cross-check
    against the historical number (and it extrapolates cleanly beyond the
    sample's worst observed day, which historical VaR cannot)."""
    if series is None or series.empty:
        return (float("nan"), float("nan"))
    from scipy.stats import norm
    mu = float(series.mean())
    sigma = float(series.std(ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1.0 - confidence)
    var = -(mu + sigma * z)
    # Closed-form ES for a normal distribution:  mu - sigma * phi(z) / (1-c)
    es = -(mu - sigma * norm.pdf(z) / (1.0 - confidence))
    return (max(var, 0.0), max(es, 0.0))


def max_drawdown(series: pd.Series) -> dict:
    """Peak-to-trough on the *cumulative* return path. Input is daily
    log returns — we exponentiate before computing drawdown so the peak
    is a real equity-curve peak, not a log-path peak."""
    if series is None or series.empty:
        return {"max_dd": float("nan"), "peak": None, "trough": None, "days": 0}
    equity = np.exp(series.cumsum())
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax() if trough is not None else None
    duration = (trough - peak).days if (peak is not None and trough is not None) else 0
    return {
        "max_dd": float(abs(dd.min())),
        "peak": peak,
        "trough": trough,
        "days": int(duration),
    }


# ---------- Return metrics ----------

def annualized_volatility(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    return float(series.std(ddof=1) * math.sqrt(TRADING_DAYS))


def annualized_return(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    return float(series.mean() * TRADING_DAYS)


def sharpe_ratio(series: pd.Series, rf_annual: float = 0.0) -> float:
    if series is None or series.empty:
        return float("nan")
    rf_daily = rf_annual / TRADING_DAYS
    excess = series - rf_daily
    sd = excess.std(ddof=1)
    if not math.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(TRADING_DAYS))


def sortino_ratio(series: pd.Series, rf_annual: float = 0.0) -> float:
    """Downside-deviation Sharpe. Upside vol doesn't hurt — only pain
    below the minimum acceptable return (here = rf) matters."""
    if series is None or series.empty:
        return float("nan")
    rf_daily = rf_annual / TRADING_DAYS
    excess = series - rf_daily
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")
    dd_std = math.sqrt((downside ** 2).mean())
    if dd_std <= 0:
        return float("nan")
    return float(excess.mean() / dd_std * math.sqrt(TRADING_DAYS))


def calmar_ratio(series: pd.Series) -> float:
    """Annual return / max drawdown. Highly sensitive to sample length —
    a 2y window with no crisis will flatter this vs a 10y one."""
    if series is None or series.empty:
        return float("nan")
    mdd = max_drawdown(series)["max_dd"]
    if not mdd or not math.isfinite(mdd) or mdd <= 0:
        return float("nan")
    return float(annualized_return(series) / mdd)


# ---------- Beta ----------

def beta_alpha(
    asset: pd.Series, market: pd.Series
) -> tuple[float, float, float]:
    """OLS of asset on market (single-factor CAPM). Returns
    (beta, annualized alpha, r^2). Requires both series indexed the same
    way — caller is expected to align before passing."""
    df = pd.concat([asset, market], axis=1, join="inner").dropna()
    df.columns = ["a", "m"]
    if len(df) < 20:
        return (float("nan"), float("nan"), float("nan"))
    var_m = df["m"].var(ddof=1)
    if var_m <= 0:
        return (float("nan"), float("nan"), float("nan"))
    cov = df["a"].cov(df["m"])
    beta = cov / var_m
    alpha_daily = df["a"].mean() - beta * df["m"].mean()
    # r^2
    resid = df["a"] - (alpha_daily + beta * df["m"])
    ss_res = (resid ** 2).sum()
    ss_tot = ((df["a"] - df["a"].mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return (float(beta), float(alpha_daily * TRADING_DAYS), r2)


def rolling_beta(
    asset: pd.Series, market: pd.Series, window: int = 63
) -> pd.Series:
    df = pd.concat([asset, market], axis=1, join="inner").dropna()
    df.columns = ["a", "m"]
    if len(df) < window + 5:
        return pd.Series(dtype=float)
    cov = df["a"].rolling(window).cov(df["m"])
    var_m = df["m"].rolling(window).var()
    beta = cov / var_m
    return beta.dropna()


# ---------- Correlation & concentration ----------

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    if returns is None or returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()
    return returns.corr()


def concentration(weights: dict[str, float]) -> dict:
    """HHI + effective-N from weights. Effective-N = 1/HHI — a portfolio
    with HHI=0.1 behaves roughly like 10 equally-weighted names for
    diversification purposes."""
    if not weights:
        return {"hhi": float("nan"), "top": float("nan"), "effective_n": float("nan"),
                "n_positions": 0}
    w = np.array(list(weights.values()), dtype=float)
    s = w.sum()
    if s <= 0:
        return {"hhi": float("nan"), "top": float("nan"), "effective_n": float("nan"),
                "n_positions": int(len(w))}
    w = w / s
    hhi = float((w ** 2).sum())
    return {
        "hhi": hhi,
        "top": float(w.max()),
        "effective_n": float(1.0 / hhi) if hhi > 0 else float("nan"),
        "n_positions": int(len(w)),
    }


# ---------- Factor exposure ----------

@st.cache_data(ttl=3600, show_spinner=False)
def build_factor_returns(period: str = "2y") -> pd.DataFrame:
    """Compute factor return series using ETF proxies. See module docstring
    for the factor definitions. Returns a DataFrame with columns
    [MKT, SMB, HML, MOM, RATES] indexed by date."""
    px = fetch_price_history(tuple(sorted(_FACTOR_ETFS)), period=period)
    if px.empty:
        return pd.DataFrame()
    r = compute_returns(px)
    if r.empty:
        return pd.DataFrame()

    factors = pd.DataFrame(index=r.index)
    if "SPY" in r.columns:
        factors["MKT"] = r["SPY"]
    if "IWM" in r.columns and "SPY" in r.columns:
        factors["SMB"] = r["IWM"] - r["SPY"]
    if "VTV" in r.columns and "VUG" in r.columns:
        factors["HML"] = r["VTV"] - r["VUG"]
    if "MTUM" in r.columns and "SPY" in r.columns:
        factors["MOM"] = r["MTUM"] - r["SPY"]
    if "TLT" in r.columns:
        factors["RATES"] = r["TLT"]
    return factors.dropna(how="any")


def factor_exposure(
    portfolio_returns: pd.Series, factor_returns: pd.DataFrame
) -> dict:
    """Multiple regression of portfolio excess returns on factor returns.
    No intercept constraint (we let alpha absorb whatever isn't spanned).
    Returns loadings dict + annualized alpha + r^2."""
    if (
        portfolio_returns is None or portfolio_returns.empty
        or factor_returns is None or factor_returns.empty
    ):
        return {}

    df = factor_returns.join(portfolio_returns.rename("port"), how="inner").dropna()
    if len(df) < 30 or df.shape[1] < 2:
        return {}

    y = df["port"].values
    X = df.drop(columns=["port"]).values
    # Add intercept column
    X_aug = np.column_stack([np.ones(len(X)), X])

    try:
        coefs, resid, rank, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    except Exception:
        return {}

    alpha_daily = float(coefs[0])
    loadings = {
        name: float(coefs[i + 1])
        for i, name in enumerate(df.drop(columns=["port"]).columns)
    }
    y_hat = X_aug @ coefs
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha_annual": alpha_daily * TRADING_DAYS,
        "loadings": loadings,
        "r2": float(r2),
        "n_obs": int(len(df)),
    }


# ---------- Master aggregator ----------

def portfolio_risk_report(
    holdings: dict,
    live_prices: dict[str, dict],
    period: str = "2y",
    rf_annual: float = 0.04,
    var_confidence: float = 0.95,
    benchmark: str = "SPY",
) -> dict:
    """One-call convenience: pull prices, build weights, compute everything.

    Returns a dict with raw numeric metrics and a few DataFrames that the
    UI can chart directly. Designed to be safe against partial data — any
    metric that can't be computed comes back as NaN or empty rather than
    raising, so the dashboard still renders something useful.
    """
    weights = build_portfolio_weights(holdings, live_prices)
    tickers = tuple(sorted(weights.keys()))

    # Fetch the risky tickers + benchmark + factor ETFs in one pull so the
    # cache holds a consistent snapshot.
    all_needed = tuple(sorted(set(tickers) | {benchmark} | set(_FACTOR_ETFS)))
    prices = fetch_price_history(all_needed, period=period)
    returns = compute_returns(prices)

    # Subset for portfolio construction.
    if returns.empty or not weights:
        return {
            "error": "No usable price history for current holdings.",
            "weights": weights,
            "n_positions": len(weights),
        }

    port_r = portfolio_return_series(returns[[c for c in returns.columns if c in weights]], weights)
    if port_r.empty:
        return {
            "error": "Could not build portfolio return series.",
            "weights": weights,
        }

    hv_1d, hcvar_1d = historical_var(port_r, var_confidence)
    pv_1d, pcvar_1d = parametric_var(port_r, var_confidence)
    dd = max_drawdown(port_r)
    vol = annualized_volatility(port_r)
    ann_ret = annualized_return(port_r)
    sharpe = sharpe_ratio(port_r, rf_annual)
    sortino = sortino_ratio(port_r, rf_annual)
    calmar = calmar_ratio(port_r)

    beta = alpha = r2 = float("nan")
    roll_b = pd.Series(dtype=float)
    if benchmark in returns.columns:
        beta, alpha, r2 = beta_alpha(port_r, returns[benchmark])
        roll_b = rolling_beta(port_r, returns[benchmark], window=63)

    # Correlation matrix over the *holdings* only.
    corr_df = correlation_matrix(returns[[t for t in weights if t in returns.columns]])

    # Factor exposure.
    factors_df = build_factor_returns(period=period)
    fx = factor_exposure(port_r, factors_df) if not factors_df.empty else {}

    conc = concentration(weights)

    # Position-level metrics — per-ticker vol, beta, and contribution to risk.
    per_pos = []
    port_var = float(port_r.var(ddof=1))
    for t in weights:
        if t not in returns.columns:
            per_pos.append({
                "Ticker": t, "Weight": weights[t],
                "Vol (ann.)": float("nan"), "Beta": float("nan"),
                "Contribution to Risk": float("nan"),
            })
            continue
        ti = returns[t]
        t_vol = annualized_volatility(ti)
        t_beta = float("nan")
        if benchmark in returns.columns:
            t_beta, _, _ = beta_alpha(ti, returns[benchmark])
        # Marginal contribution to variance = w_i * cov(r_i, r_p) / var(r_p)
        if port_var > 0:
            mctr = weights[t] * ti.cov(port_r) / port_var
        else:
            mctr = float("nan")
        per_pos.append({
            "Ticker": t,
            "Weight": weights[t],
            "Vol (ann.)": t_vol,
            "Beta": t_beta,
            "Contribution to Risk": float(mctr) if mctr is not None else float("nan"),
        })
    per_pos_df = pd.DataFrame(per_pos).sort_values("Weight", ascending=False)

    return {
        "weights": weights,
        "n_positions": len(weights),
        "period": period,
        "n_obs": int(len(port_r)),
        "var_confidence": var_confidence,
        "hist_var_1d": hv_1d,
        "hist_cvar_1d": hcvar_1d,
        "param_var_1d": pv_1d,
        "param_cvar_1d": pcvar_1d,
        "ann_return": ann_ret,
        "ann_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": dd["max_dd"],
        "dd_peak": dd["peak"],
        "dd_trough": dd["trough"],
        "dd_days": dd["days"],
        "beta": beta,
        "alpha_annual": alpha,
        "r_squared": r2,
        "concentration": conc,
        "factor_exposure": fx,
        "portfolio_returns": port_r,
        "rolling_beta": roll_b,
        "correlation": corr_df,
        "per_position": per_pos_df,
    }
