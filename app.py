import math
from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="📈",
    layout="wide",
)


PORTFOLIO_COLUMNS = ["Ticker", "Shares", "Purchase Price"]
NUMERIC_COLUMNS = ("Shares", "Purchase Price")
DEFAULT_PORTFOLIO_ROWS = [
    {"Ticker": "AAPL", "Shares": 10.0, "Purchase Price": 120.50},
    {"Ticker": "MSFT", "Shares": 5.0, "Purchase Price": 250.75},
    {"Ticker": "GOOGL", "Shares": 4.0, "Purchase Price": 150.10},
    {"Ticker": "AMZN", "Shares": 3.0, "Purchase Price": 110.30},
    {"Ticker": "TSLA", "Shares": 2.0, "Purchase Price": 200.95},
]
PLOTLY_CONFIG = {"responsive": True, "displaylogo": False}


def default_portfolio_table() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_PORTFOLIO_ROWS, columns=PORTFOLIO_COLUMNS).copy()


def coerce_numeric_columns(df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result:
            result[column] = (
                pd.to_numeric(result[column], errors="coerce")
                .astype(float)
            )
    return result


def ensure_portfolio_table(raw_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if raw_df is None:
        base = pd.DataFrame(columns=PORTFOLIO_COLUMNS)
    else:
        base = raw_df.copy()
    for column in PORTFOLIO_COLUMNS:
        if column not in base:
            base[column] = np.nan
    base = base[PORTFOLIO_COLUMNS]
    return coerce_numeric_columns(base, NUMERIC_COLUMNS)


@st.cache_data(ttl=600, show_spinner=False)
def validate_ticker(symbol: str) -> bool:
    cleaned = symbol.strip().upper()
    if not cleaned:
        return False
    try:
        history = yf.Ticker(cleaned).history(period="5d", auto_adjust=False)
    except Exception:
        return False
    return not history.empty


@st.cache_data(ttl=900, show_spinner=False)
def download_price_history(tickers: Tuple[str, ...], start: date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=list(tickers),
            start=start,
            progress=False,
            group_by="column",
            auto_adjust=False,
            actions=False,
        )
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            try:
                adj_close = data.xs("Adj Close", axis=1, level=0)
            except (KeyError, IndexError):
                adj_close = data.xs("Adj Close", axis=1, level=1)
        else:
            adj_close = data["Adj Close"]
    except (KeyError, IndexError):
        return pd.DataFrame()

    if isinstance(adj_close, pd.Series):
        symbol = tickers[0]
        adj_close = adj_close.to_frame(name=symbol)

    adj_close = adj_close.rename(columns=str.upper)
    adj_close.index = adj_close.index.tz_localize(None)
    adj_close.index.name = "Date"
    return adj_close.sort_index()


def clean_portfolio_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["Ticker", "Shares", "Purchase Price"])

    df = raw_df.copy()
    df["Ticker"] = (
        df.get("Ticker", "")
        .astype(str)
        .str.upper()
        .str.strip()
        .replace("", pd.NA)
    )

    for column in ["Shares", "Purchase Price", "Target Allocation %"]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Ticker", "Shares"])
    df = df[df["Shares"] > 0]
    df = df.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
    return df


def build_portfolio_snapshot(
    portfolio: pd.DataFrame,
    prices: pd.Series,
    previous_prices: Optional[pd.Series],
) -> pd.DataFrame:
    if portfolio.empty or prices.empty:
        return pd.DataFrame()

    df = portfolio.set_index("Ticker").copy()
    df["Current Price"] = prices.reindex(df.index)

    if "Purchase Price" not in df:
        df["Purchase Price"] = np.nan
    df["Purchase Price"] = df["Purchase Price"].fillna(df["Current Price"])

    df["Cost Basis"] = df["Shares"] * df["Purchase Price"]
    df["Market Value"] = df["Shares"] * df["Current Price"]
    df["Unrealized P/L"] = df["Market Value"] - df["Cost Basis"]
    df["Return %"] = np.where(
        df["Cost Basis"] > 0,
        df["Unrealized P/L"] / df["Cost Basis"] * 100,
        np.nan,
    )

    if previous_prices is not None and not previous_prices.empty:
        prev = previous_prices.reindex(df.index)
        df["Prev Close"] = prev
        df["Daily Change"] = df["Shares"] * (df["Current Price"] - prev)
        df["Daily Change %"] = np.where(
            prev > 0,
            (df["Current Price"] - prev) / prev * 100,
            np.nan,
        )
    else:
        df["Prev Close"] = np.nan
        df["Daily Change"] = np.nan
        df["Daily Change %"] = np.nan

    return df


def compute_portfolio_timeseries(
    price_history: pd.DataFrame, shares_by_ticker: pd.Series
) -> pd.Series:
    if price_history.empty or shares_by_ticker.empty:
        return pd.Series(dtype=float)

    aligned_shares = shares_by_ticker.reindex(price_history.columns).fillna(0)
    position_values = price_history.mul(aligned_shares, axis=1)
    portfolio_series = position_values.sum(axis=1)
    portfolio_series = portfolio_series[portfolio_series > 0]
    portfolio_series.index.name = "Date"
    return portfolio_series


def render_summary_metrics(snapshot: pd.DataFrame, risk_free_rate: float) -> None:
    total_value = snapshot["Market Value"].sum()
    total_cost = snapshot["Cost Basis"].sum()
    total_pl = snapshot["Unrealized P/L"].sum()
    day_change = snapshot["Daily Change"].sum(min_count=1)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Portfolio Value", f"${total_value:,.2f}")
    if not math.isfinite(total_pl):
        col_b.metric("Unrealized P/L", "—")
    else:
        col_b.metric("Unrealized P/L", f"${total_pl:,.2f}")
    if not math.isfinite(total_cost) or total_cost == 0:
        total_return_pct = np.nan
    else:
        total_return_pct = total_pl / total_cost * 100
    if math.isfinite(total_return_pct):
        col_c.metric("Total Return %", f"{total_return_pct:,.2f}%")
    else:
        col_c.metric("Total Return %", "—")
    if math.isfinite(day_change):
        prev_value = total_value - day_change
        day_pct = (day_change / prev_value * 100) if prev_value else np.nan
        delta_pct_display = f"{day_pct:,.2f}%" if math.isfinite(day_pct) else "—"
        col_d.metric("Daily Change", f"${day_change:,.2f}", delta_pct_display)
    else:
        col_d.metric("Daily Change", "—")

    st.markdown("---")

    price_history = st.session_state.get("portfolio_history", pd.Series(dtype=float))
    if price_history.empty or price_history.size < 3:
        st.info("More price history is required to compute risk metrics.")
        return

    returns = price_history.pct_change().dropna()
    if returns.empty:
        st.info("Not enough return data to compute risk metrics.")
        return

    avg_daily = returns.mean()
    volatility = returns.std() * np.sqrt(252)
    annual_return = (1 + avg_daily) ** 252 - 1
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility else np.nan

    drawdowns = price_history / price_history.cummax() - 1
    max_drawdown = drawdowns.min()

    risk_cols = st.columns(4)
    risk_cols[0].metric("Annual Return", f"{annual_return*100:,.2f}%")
    risk_cols[1].metric("Annual Volatility", f"{volatility*100:,.2f}%")
    risk_cols[2].metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:,.2f}" if math.isfinite(sharpe_ratio) else "—",
    )
    risk_cols[3].metric(
        "Max Drawdown",
        f"{max_drawdown*100:,.2f}%" if math.isfinite(max_drawdown) else "—",
    )


def render_portfolio_table(snapshot: pd.DataFrame) -> None:
    if snapshot.empty:
        st.warning("Add tickers and share counts to see your portfolio analysis.")
        return

    display_df = snapshot.copy()
    display_df.index.name = "Ticker"
    display_df = display_df[
        [
            "Shares",
            "Purchase Price",
            "Current Price",
            "Market Value",
            "Cost Basis",
            "Unrealized P/L",
            "Return %",
            "Daily Change",
            "Daily Change %",
        ]
    ]

    st.dataframe(
        display_df.style.format(
            {
                "Shares": "{:,.2f}",
                "Purchase Price": "${:,.2f}",
                "Current Price": "${:,.2f}",
                "Market Value": "${:,.2f}",
                "Cost Basis": "${:,.2f}",
                "Unrealized P/L": "${:,.2f}",
                "Return %": "{:,.2f}%",
                "Daily Change": "${:,.2f}",
                "Daily Change %": "{:,.2f}%",
            }
        ),
        width="stretch",
    )


def render_allocation_chart(snapshot: pd.DataFrame) -> None:
    if snapshot.empty:
        return

    breakdown = snapshot["Market Value"].reset_index(name="Market Value")
    breakdown = breakdown[breakdown["Market Value"] > 0]
    if breakdown.empty:
        return

    fig = px.pie(
        breakdown,
        names="Ticker",
        values="Market Value",
        title="Allocation by Market Value",
        hole=0.4,
    )
    st.plotly_chart(fig, config=PLOTLY_CONFIG)


def render_history_chart(portfolio_history: pd.Series) -> None:
    if portfolio_history.empty:
        return

    history_df = (
        portfolio_history.to_frame(name="Portfolio Value")
        .reset_index()
        .rename(columns={"index": "Date"})
    )
    fig = px.line(
        history_df,
        x="Date",
        y="Portfolio Value",
        title="Portfolio Value Over Time",
    )
    st.plotly_chart(fig, config=PLOTLY_CONFIG)


def forecast_portfolio_returns(
    price_history: pd.DataFrame,
    snapshot: pd.DataFrame,
    horizon_days: int,
    simulations: int,
    lookback_days: int,
    method: str,
) -> Optional[dict]:
    if (
        price_history.empty
        or snapshot.empty
        or horizon_days <= 0
        or simulations <= 0
        or len(price_history.columns) == 0
    ):
        return None

    if lookback_days > 0 and len(price_history) > lookback_days:
        price_history = price_history.iloc[-lookback_days:]

    daily_returns = price_history.pct_change().dropna()
    if daily_returns.empty or len(daily_returns) < 5:
        return None

    log_returns = np.log1p(daily_returns)
    mu = log_returns.mean().to_numpy()
    cov = log_returns.cov().to_numpy()
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(cov)):
        return None

    if "Market Value" not in snapshot.columns:
        return None

    if "Ticker" in snapshot.columns:
        market_values = snapshot.set_index("Ticker")["Market Value"]
    else:
        market_values = snapshot["Market Value"]
        market_values.index = snapshot.index

    weights = (
        market_values.reindex(price_history.columns)
        .fillna(0)
    )
    total_value = weights.sum()
    if total_value <= 0:
        return None

    weights = (weights / total_value).to_numpy()
    simulations = min(int(simulations), 10000)
    horizon_days = int(horizon_days)

    rng = np.random.default_rng()
    draws: np.ndarray
    if method == "Bootstrap" and len(log_returns) >= horizon_days:
        history = log_returns.to_numpy()
        random_idx = rng.integers(0, history.shape[0], size=(simulations, horizon_days))
        draws = history[random_idx]
    else:
        try:
            draws = rng.multivariate_normal(mu, cov, size=(simulations, horizon_days))
        except np.linalg.LinAlgError:
            variances = np.clip(np.diag(cov), a_min=1e-10, a_max=None)
            std_dev = np.sqrt(variances)
            draws = rng.normal(
                loc=mu,
                scale=std_dev,
                size=(simulations, horizon_days, len(mu)),
            )

    cumulative_log_returns = draws.sum(axis=1)
    asset_returns = np.expm1(cumulative_log_returns)
    portfolio_returns = asset_returns @ weights
    if not np.all(np.isfinite(portfolio_returns)):
        return None

    expected = float(np.mean(portfolio_returns))
    median = float(np.median(portfolio_returns))
    downside = float(np.percentile(portfolio_returns, 5))
    upside = float(np.percentile(portfolio_returns, 95))

    return {
        "expected_return": expected,
        "median_return": median,
        "downside": downside,
        "upside": upside,
        "simulated_returns": portfolio_returns,
        "current_value": float(total_value),
    }


def render_forecast_section(
    price_history: pd.DataFrame,
    snapshot: pd.DataFrame,
    horizon_days: int,
    simulations: int,
    lookback_days: int,
    method: str,
) -> None:
    results = forecast_portfolio_returns(
        price_history,
        snapshot,
        horizon_days,
        simulations,
        lookback_days,
        method,
    )
    if not results:
        st.info("More price history is required to produce a near-term forecast.")
        return

    current_value = results["current_value"]
    expected_value = current_value * (1 + results["expected_return"])
    downside_value = current_value * (1 + results["downside"])
    upside_value = current_value * (1 + results["upside"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        f"Expected {horizon_days}-day Return",
        f"{results['expected_return']*100:,.2f}%",
        delta=f"${expected_value - current_value:,.0f}",
    )
    col2.metric(
        f"Median {horizon_days}-day Return",
        f"{results['median_return']*100:,.2f}%",
        delta=f"${current_value * (1 + results['median_return']) - current_value:,.0f}",
    )
    col3.metric(
        "Downside (5th pct)",
        f"{results['downside']*100:,.2f}%",
        delta=f"${downside_value - current_value:,.0f}",
    )
    col4.metric(
        "Upside (95th pct)",
        f"{results['upside']*100:,.2f}%",
        delta=f"${upside_value - current_value:,.0f}",
    )

    st.caption(
        "Monte Carlo forecast using recent daily log returns. "
        f"Method: {method}. Downside and upside show the 5th and 95th percentile outcomes."
    )
    histogram_df = pd.DataFrame(
        {"Simulated Return %": results["simulated_returns"] * 100}
    )
    hist_fig = px.histogram(
        histogram_df,
        x="Simulated Return %",
        nbins=40,
        title="Distribution of Simulated Portfolio Returns",
    )
    st.plotly_chart(hist_fig, config=PLOTLY_CONFIG)


def render_download_button(snapshot: pd.DataFrame) -> None:
    if snapshot.empty:
        return

    csv_bytes = snapshot.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download snapshot as CSV",
        data=csv_bytes,
        file_name="portfolio_snapshot.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Live Portfolio Tracker")
    st.caption(
        "Enter your holdings to monitor market value, daily performance, and risk metrics using live Yahoo Finance data."
    )
    sidebar = st.sidebar
    sidebar.header("Configuration")
    default_start = date.today() - timedelta(days=365)
    history_start = sidebar.date_input(
        "Price history start date",
        value=default_start,
        max_value=date.today(),
    )
    risk_free_rate_pct = sidebar.number_input(
        "Risk-free rate (annual, %)",
        min_value=-5.0,
        max_value=15.0,
        value=2.0,
        step=0.25,
    )
    sidebar.caption("Live quotes are fetched from Yahoo Finance when you run or update the dashboard.")
    sidebar.subheader("Forecast settings")
    forecast_horizon_days = sidebar.number_input(
        "Forecast horizon (days)",
        min_value=1,
        max_value=90,
        value=7,
        step=1,
    )
    simulation_count = sidebar.slider(
        "Monte Carlo simulations",
        min_value=200,
        max_value=5000,
        value=2000,
        step=100,
    )
    lookback_days = sidebar.slider(
        "Historical lookback (days)",
        min_value=30,
        max_value=720,
        value=252,
        step=30,
    )
    forecast_method = sidebar.selectbox(
        "Simulation method",
        options=("Bootstrap", "Gaussian"),
        index=0,
    )

    if "portfolio_table" not in st.session_state:
        st.session_state["portfolio_table"] = default_portfolio_table()

    sidebar.subheader("Add ticker from Yahoo Finance")
    with sidebar.form("add_ticker_form"):
        new_ticker = st.text_input("Ticker symbol", placeholder="e.g. NVDA")
        new_shares = st.number_input(
            "Shares",
            min_value=0.0,
            value=1.0,
            step=0.01,
        )
        new_price = st.number_input(
            "Purchase price (optional)",
            min_value=0.0,
            value=0.0,
            step=0.01,
        )
        submitted = st.form_submit_button("Verify & add")

    if submitted:
        symbol = new_ticker.strip().upper()
        existing_symbols = (
            st.session_state["portfolio_table"]["Ticker"].astype(str).str.upper()
        )
        if not symbol:
            sidebar.warning("Enter a ticker symbol to add.")
        elif symbol in existing_symbols.values:
            sidebar.warning(f"{symbol} is already in the portfolio.")
        elif new_shares <= 0:
            sidebar.warning("Shares must be greater than zero.")
        elif not validate_ticker(symbol):
            sidebar.error(
                f"Yahoo Finance did not recognise {symbol}. Check the ticker and try again."
            )
        else:
            new_row = {
                "Ticker": symbol,
                "Shares": float(new_shares),
                "Purchase Price": float(new_price) if new_price > 0 else np.nan,
            }
            st.session_state["portfolio_table"] = ensure_portfolio_table(
                pd.concat(
                    [st.session_state["portfolio_table"], pd.DataFrame([new_row])],
                    ignore_index=True,
                )
            )
            sidebar.success(f"Added {symbol} from Yahoo Finance.")

    editable_table = ensure_portfolio_table(st.session_state["portfolio_table"])
    portfolio_input = st.data_editor(
        editable_table,
        num_rows="dynamic",
        width="stretch",
        key="portfolio_editor",
    )

    st.session_state["portfolio_table"] = ensure_portfolio_table(portfolio_input)

    portfolio = clean_portfolio_input(st.session_state["portfolio_table"])
    if portfolio.empty:
        st.info("Add at least one position to build the dashboard.")
        return

    requested_tickers = tuple(portfolio["Ticker"].tolist())
    price_history = download_price_history(requested_tickers, history_start)
    missing = set(requested_tickers) - set(price_history.columns)
    if missing:
        st.warning(
            f"No data was returned for: {', '.join(sorted(missing))}. "
            "Check the ticker symbols and try again."
        )

    if price_history.empty:
        st.error("Unable to load price data. Please adjust the tickers or try later.")
        return

    st.session_state["portfolio_history"] = compute_portfolio_timeseries(
        price_history, portfolio.set_index("Ticker")["Shares"]
    )

    latest_prices = price_history.iloc[-1]
    previous_prices = price_history.iloc[-2] if len(price_history) > 1 else None
    snapshot = build_portfolio_snapshot(portfolio, latest_prices, previous_prices)
    if snapshot.empty:
        st.error("Portfolio data could not be computed. Please review your inputs.")
        return

    render_summary_metrics(snapshot, risk_free_rate_pct / 100)

    tab_overview, tab_history, tab_forecast, tab_download = st.tabs(
        ["Overview", "Performance", "Forecast", "Export"]
    )
    with tab_overview:
        render_portfolio_table(snapshot)
        render_allocation_chart(snapshot)

    with tab_history:
        render_history_chart(st.session_state["portfolio_history"])
        normalized = price_history / price_history.iloc[0] * 100
        normalized = normalized.reset_index().melt(
            id_vars="Date",
            var_name="Ticker",
            value_name="Indexed Price",
        )
        asset_fig = px.line(
            normalized,
            x="Date",
            y="Indexed Price",
            color="Ticker",
            title="Indexed Price Performance (Start = 100)",
        )
        st.plotly_chart(asset_fig, config=PLOTLY_CONFIG)

        portfolio_returns = (
            st.session_state["portfolio_history"].pct_change().dropna().mul(100)
        )
        if not portfolio_returns.empty:
            st.bar_chart(portfolio_returns, width="stretch")

    with tab_forecast:
        render_forecast_section(
            price_history,
            snapshot,
            forecast_horizon_days,
            simulation_count,
            lookback_days,
            forecast_method,
        )

    with tab_download:
        render_download_button(snapshot)
        st.dataframe(price_history, width="stretch")


if __name__ == "__main__":
    main()
