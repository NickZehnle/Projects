## Regime-Adaptive PCA Mean Reversion Strategy

#### Backtest Results
[Strategy Performance Report](PCA_Stat_Arb_Report.pdf)

#### Code
~~~python
import pandas as pd
from pykalman import KalmanFilter
from AlgorithmImports import *
from sklearn.decomposition import PCA
import statsmodels.api as sm
from arch import arch_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class PCAStatArbitrageAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 4, 1)
        self.SetCash(1_000_000)
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0

        self._num_components = self.get_parameter("num_components", 2)
        self._lookback = self.get_parameter("lookback_days", 504)
        self._z_score_threshold = self.get_parameter("z_score_threshold", 1)
        self._universe_size = self.get_parameter("universe_size", 30)
        self.mu = self.get_parameter("mu", 200)
        self.theta = self.get_parameter("theta", .03)
        self._pca_lookback = self.get_parameter("pca_lookback", 10)

        self.indicators = {}
        self.indicator_period = 14
        self.SetWarmUp(timedelta(days=44))

        schedule_symbol = Symbol.Create("SPY", SecurityType.EQUITY, Market.USA)
        date_rule = self.DateRules.WeekStart(schedule_symbol)
        self.UniverseSettings.Schedule.On(date_rule)
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.RAW
        self._universe = self.AddUniverse(self._select_assets)

        self.spy_symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        self.Schedule.On(
            date_rule, 
            self.TimeRules.AfterMarketOpen(schedule_symbol, 1), 
            self._trade
        )

        chart = Chart('Explained Variance Ratio')
        self.AddChart(chart)
        for i in range(self._num_components):
            chart.AddSeries(Series(f"Component {i}", SeriesType.LINE, ""))

    def OnSecuritiesChanged(self, changes):
        for added in changes.AddedSecurities:
            symbol = added.Symbol
            if symbol not in self.indicators:
                rsi = self.RSI(symbol, self.indicator_period, MovingAverageType.Wilders, Resolution.Daily)
                macd = self.MACD(symbol, 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
                self.indicators[symbol] = {"RSI": rsi, "MACD": macd}

    def _select_assets(self, fundamental):
        return [
            f.Symbol 
            for f in sorted(
                [f for f in fundamental if f.Price > 5], 
                key=lambda f: f.DollarVolume
            )[-self._universe_size:]
        ]
    
    def _trade(self):
        tradeable_assets = [
            symbol 
            for symbol in self._universe.Selected 
            if self.Securities[symbol].Price and symbol in self.CurrentSlice.QuoteBars
        ]

        history = self.History(
            tradeable_assets, self._lookback, Resolution.Daily, 
            dataNormalizationMode=DataNormalizationMode.ScaledRaw
        ).close.unstack(level=0)
        history = history.loc[:, history.count() >= 500]
        self.Debug(f'Number of assets: {len(history.columns)}')

        if history.empty:
            return

        avg_vol = self._egarch_vol(history)
        self._z_score_threshold = self._adjust_z_score_threshold(avg_vol)
        regime = self._knn_regime()
        weights = self._get_weights(history, regime)

        self.SetHoldings(
            [
                PortfolioTarget(symbol, -weight) 
                for symbol, weight in weights.items()
            ], 
            True
        )

    def _knn_regime(self):
        spy_history = self.History(self.spy_symbol, 252, Resolution.Daily).close.unstack(level=0)
        if spy_history.empty:
            return 0  # Default to reverting

        returns = np.log(spy_history).diff().dropna()

        features = pd.DataFrame(index=returns.index)
        features['volatility'] = returns.rolling(20).std().mean(axis=1)
        features['autocorr'] = returns.rolling(20).apply(lambda x: x.autocorr(), raw=False).mean(axis=1)
        features['trend'] = returns.mean(axis=1).rolling(20).mean()

        features = features.dropna()

        features['label'] = ((features['autocorr'] > 0) & (features['trend'].abs() > 0.001)).astype(int)

        scaler = StandardScaler()
        X = scaler.fit_transform(features[['volatility', 'autocorr', 'trend']])
        y = features['label']
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, y)

        current_features = pd.DataFrame({
            'volatility': [returns[-20:].std().mean()],
            'autocorr': [returns[-20:].apply(lambda x: x.autocorr()).mean()],
            'trend': [returns[-20:].mean().mean()]
        })

        X_current = scaler.transform(current_features)
        regime = knn.predict(X_current)[0]  

        self.Debug(f"Regime detected: {'Trending' if regime == 1 else 'Reverting'}")
        return regime
    
    def _egarch_vol(self, history):
        history = history.dropna()
        returns = (np.log(history)).diff().dropna()
        vol_ests = {}

        for asset in returns.columns:
            model = arch_model(returns[asset], vol='EGARCH', p=1, q=1)
            fit = model.fit(disp="off")
            vol_ests[asset] = fit.conditional_volatility[-1] 

        avg_vol = pd.Series(vol_ests).mean()
        self.Debug(f'avg_vol: {avg_vol}')
        return avg_vol

    def _adjust_z_score_threshold(self, avg_vol):
        sigmoid = 1 / (1 + np.exp(-self.mu * (avg_vol - self.theta)))
        z_score_threshold = 0.5 + (sigmoid * (1.5 - 0.5)) 
        self.Debug(f'z_score_threshold: {z_score_threshold}')
        return z_score_threshold

    def _get_weights(self, history, regime):
        sample = np.log(history[-self._pca_lookback:].dropna(axis=1))
        sample -= sample.mean()
        model = PCA().fit(sample)

        for i in range(self._num_components):
            self.Plot(
                'Explained Variance Ratio', f"Component {i}", 
                model.explained_variance_ratio_[i]
            )

        factors = np.dot(sample, model.components_.T)[:, :self._num_components]
        factors = sm.add_constant(factors)

        model_by_ticker = {
            ticker: sm.OLS(sample[ticker], factors).fit() 
            for ticker in sample.columns
        }

        resids = pd.DataFrame(
            {ticker: self._apply_kalman_filter(model.resid) for ticker, model in model_by_ticker.items()}
        )

        zscores = ((resids - resids.mean()) / resids.std()).iloc[-1]

        rsi_vals = {}
        macd_vals = {}

        for symbol in zscores.index:
            indicators = self.indicators.get(symbol)
            if indicators is None or not all(indicator.IsReady for indicator in indicators.values()):
                continue

            rsi_vals[symbol] = indicators["RSI"].Current.Value
            macd_vals[symbol] = indicators["MACD"].Current.Value - indicators["MACD"].Signal.Current.Value

        def compute_z_scores(val_dict):
            series = pd.Series(val_dict)
            return (series - series.mean()) / series.std()

        rsi_z = compute_z_scores(rsi_vals)
        macd_z = compute_z_scores(macd_vals)

        adj_scores = {}
        for symbol in zscores.index:
            if symbol not in rsi_z or symbol not in macd_z:
                continue

            if regime == 0:
                adj_scores[symbol] = 0.7 * zscores[symbol] - 0.3 * rsi_z[symbol]
            else:
                adj_scores[symbol] = -0.3 * zscores[symbol] + 0.7 * macd_z[symbol]

        adj_scores = pd.Series(adj_scores)
        selected = adj_scores[abs(adj_scores) > self._z_score_threshold]

        if not selected.empty:
            weights = selected * (1 / selected.abs().sum())
            return weights.sort_values()

        return pd.Series(dtype=float)

    def _apply_kalman_filter(self, series):
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
