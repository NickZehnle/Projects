## Regime-Adaptive PCA Statistical Arbitrage Strategy

#### Backtest Results
[Strategy Performance Report 2023-2025](PCA_Stat_Arb_Report_23-25.pdf)
[Strategy Performance Report 2022-2024](PCA_Stat_Arb_Report_22-24.pdf)
[Strategy Performance Statistics](PCA_Stat_Arb_Stats.pdf)

#### Code
~~~python
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from AlgorithmImports import *
from sklearn.decomposition import PCA
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from QuantConnect.Data.Fundamental import MorningstarSectorCode

class PCAStatArbitrageAlgorithm(QCAlgorithm):

    def Initialize(self):
        # Backtest 
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(1_000_000)
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0

        # Parameters
        self._pca_lookback = 20
        self._regime_lookback = 20
        self._universe_size = 60
        
        # Setup
        self.indicators = {}
        self.SetWarmUp(timedelta(days=44))
        self.industry = MorningstarSectorCode.TECHNOLOGY
        self.AddUniverse(self.CoarseSelection, self.FineSelection)
        self.benchmark_etf = self.AddEquity("QQQ", Resolution.Daily).Symbol
        self.spy_symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Trading schedule
        self.Schedule.On(
            self.DateRules.WeekStart(self.spy_symbol),
            self.TimeRules.AfterMarketOpen(self.spy_symbol, 5),
            self._trade
        )

        self.Schedule.On(
            self.DateRules.WeekEnd(self.spy_symbol),
            self.TimeRules.BeforeMarketClose(self.spy_symbol, 5),
            lambda: self.Liquidate()
        )

        # Additional Charts
        evr_chart = Chart('Explained Variance Ratio')
        self.AddChart(evr_chart)
        for i in range(2):
            evr_chart.AddSeries(Series(f"Component {i}", SeriesType.LINE, ""))

        z_chart = Chart('Volatility Adjusted Threshold')
        z_chart.AddSeries(Series('Z-Score Threshold', SeriesType.LINE, ''))
        self.AddChart(z_chart)

    def CoarseSelection(self, coarse):
        # Universe by dollar volume
        filtered = [c for c in coarse if c.HasFundamentalData and c.Price > 5]
        top = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)[:500]
        return [x.Symbol for x in top]

    def FineSelection(self, fine):
        # Refine universe to satisfy intended size and industry
        same_industry = [
            f for f in fine
            if f.HasFundamentalData
            and f.AssetClassification is not None
            and getattr(f.AssetClassification, "MorningstarSectorCode", None) == self.industry
        ]

        selected = sorted(
            same_industry,
            key=lambda f: getattr(f, "MarketCap", 0) or 0,
            reverse=True
        )[:self._universe_size]

        self.selected = [x.Symbol for x in selected]
        return self.selected

    def OnSecuritiesChanged(self, changes):
        # Compute RSI for additional robustness of signal
        for added in changes.AddedSecurities:
            symbol = added.Symbol
            if symbol not in self.indicators:
                rsi = self.RSI(symbol, 14, MovingAverageType.Wilders, Resolution.Daily)
                self.indicators[symbol] = {"RSI": rsi}

    def _trade(self):
        # Trading logic
        if self.IsWarmingUp or not hasattr(self, "selected") or not self.selected:
            return

        tradeable_assets = [
            s for s in self.selected
            if s in self.Securities and self.Securities[s].HasData
        ]

        history = self.History(
            tradeable_assets, self._pca_lookback + 10, Resolution.Daily,
            dataNormalizationMode=DataNormalizationMode.ScaledRaw
        ).close.unstack(level=0)
        
        history = history.loc[:, history.count() >= self._pca_lookback]
        if history.empty or history.shape[1] < 5:
            return

        etf_history = self.History(self.benchmark_etf, 504, Resolution.Daily).close
        if etf_history.empty:
            return

        etf_returns = np.log(etf_history).diff().dropna()
        etf_vol = self._egarch_vol(etf_returns)
        z_score_threshold = self._adjust_z_score_threshold(etf_vol)
        self.Plot('Volatility Adjusted Threshold', 'Z-Score Threshold', z_score_threshold)

        regime, direction = self._arima_regime(etf_returns)

        if regime == "trending":
            self.SetHoldings(self.benchmark_etf, direction)
            return

        weights = self._get_weights(history, regime, z_score_threshold)
        self.SetHoldings([PortfolioTarget(sym, -w) for sym, w in weights.items()], True)

    def _arima_regime(self, returns):
        # Detect industry regime using ARIMA 
        fit = ARIMA(returns[-self._regime_lookback:], order=(4, 0, 4)).fit()
        trend_strength = np.mean(fit.arparams) + np.mean(fit.maparams)
        bias = np.std(fit.resid)
        if trend_strength > bias:
            regime = "trending"
            direction = np.sign(fit.fittedvalues.iloc[-1] + fit.resid.iloc[-1])
        else:
            regime = "reverting"
            direction = 0
        self.Debug(f"ARIMA Regime detected: {regime}, direction: {direction}")
        return regime, direction

    def _egarch_vol(self, returns):
        # Determine conditional volatility of industry ETF using EGARCH
        model = arch_model(returns, vol="EGARCH", p=1, q=1)
        fit = model.fit(disp="off")
        return fit.conditional_volatility.iloc[-1]

    def _adjust_z_score_threshold(self, vol):
        # Adjust threshold for signal according to current conditional volatility
        if not hasattr(self, "_vol_history"):
            self._vol_history = []

        self._vol_history.append(vol)
        self._vol_history = self._vol_history[-20:]
        vol_percentile = np.mean(np.array(self._vol_history) < vol)
        return .5 + vol_percentile

    def _get_weights(self, history, regime, z_score_threshold):
        # Perform PCA on log prices, compute Z-scores on Kalman filtered residuals
        # Weight positions according to composite PCA-RSI Z-scores above the threshold 
        sample = np.log(history[-self._pca_lookback:].dropna(axis=1))
        sample -= sample.mean()

        model = PCA().fit(sample)
        cum_var = np.cumsum(model.explained_variance_ratio_)
        num_components = np.searchsorted(cum_var, 0.8) + 1

        for i in range(min(num_components, len(model.explained_variance_ratio_))):
            self.Plot('Explained Variance Ratio', f"Component {i}", model.explained_variance_ratio_[i])

        factors = np.dot(sample, model.components_.T)[:, :num_components]
        factors = sm.add_constant(factors)
        model_by_ticker = {t: sm.OLS(sample[t], factors).fit() for t in sample.columns}
        resids = pd.DataFrame({t: self._apply_kalman_filter(m.resid) for t, m in model_by_ticker.items()})
        zscores = ((resids - resids.mean()) / resids.std()).iloc[-1]

        rsi_vals = {
            s: ind["RSI"].Current.Value
            for s, ind in self.indicators.items()
            if s in zscores.index and all(i.IsReady for i in ind.values())
        }

        rsi_vals = pd.Series(rsi_vals)
        rsi_z = (rsi_vals - rsi_vals.mean()) / rsi_vals.std()

        trend_factor = 1 if regime == "trending" else 0
        adj = {s: (1 - trend_factor) * (0.8 * zscores[s] + 0.2 * rsi_z.get(s,0)) for s in zscores.index}
        adj = pd.Series(adj).dropna()

        selected = adj[abs(adj) > z_score_threshold]

        if not selected.empty:
            weights = selected * (1 / selected.abs().sum())
            return weights.sort_values()

        return pd.Series(dtype=float)

    def _apply_kalman_filter(self, series):
        # Specify Kalman filter to reduce noise in residuals
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=series.iloc[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01
        )
        state_means, _ = kf.filter(series.values)
        return pd.Series(state_means.flatten(), index=series.index)
~~~
