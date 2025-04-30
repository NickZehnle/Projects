## Regime-Adaptive PCA Mean Reversion Strategy

#### Backtest Results
[Strategy Performance Report](PCA_Mean_Reversion2022-25.pdf)

#### Code
~~~python
import pandas as pd
from pykalman import KalmanFilter
from AlgorithmImports import *
from sklearn.decomposition import PCA
import statsmodels.api as sm
from arch import arch_model 

class PCAStatArbitrageAlgorithm(QCAlgorithm):
    
    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_end_date(2025, 1, 1)
        self.set_cash(1_000_000)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        self._num_components = self.get_parameter("num_components", 3)
        self._lookback = self.get_parameter("lookback_days", 756)
        self._z_score_threshold = self.get_parameter("z_score_threshold", 1)
        self._universe_size = self.get_parameter("universe_size", 50)

        schedule_symbol = Symbol.create("SPY", SecurityType.EQUITY, Market.USA)
        date_rule = self.date_rules.week_start(schedule_symbol)
        self.universe_settings.schedule.on(date_rule)
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        self._universe = self.add_universe(self._select_assets)

        self.schedule.on(
            date_rule, 
            self.time_rules.after_market_open(schedule_symbol, 1), 
            self._trade
        )

        chart = Chart('Explained Variance Ratio')
        self.add_chart(chart)
        for i in range(self._num_components):
            chart.add_series(
                Series(f"Component {i}", SeriesType.LINE, "")
            )

    def _select_assets(self, fundamental):
        return [
            f.symbol 
            for f in sorted(
                [f for f in fundamental if f.price > 5], 
                key=lambda f: f.dollar_volume
            )[-self._universe_size:]
        ]

    def _trade(self):
        tradeable_assets = [
            symbol 
            for symbol in self._universe.selected 
            if (self.securities[symbol].price and 
                symbol in self.current_slice.quote_bars)
        ]
        history = self.history(
            tradeable_assets, self._lookback, Resolution.DAILY, 
            data_normalization_mode=DataNormalizationMode.SCALED_RAW
        ).close.unstack(level=0)

        avg_vol = self._egarch_regime(history)
        self._z_score_threshold = self._adjust_z_score_threshold(avg_vol)
        weights = self._get_weights(history)
        
        self.set_holdings(
            [
                PortfolioTarget(symbol, -weight) 
                for symbol, weight in weights.items()
            ], 
            True
        )

    def _egarch_regime(self, history):
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
        alpha = 200 
        beta = 0.03
        sigmoid = 1 / (1 + np.exp(-alpha * (avg_vol - beta)))
        z_score_threshold = 1 + (sigmoid * (2.5 - 1))  # Map between 1 and 2.5
        self.Debug(f'z_zcore_threshold: {z_score_threshold}')
        return z_score_threshold

    def _get_weights(self, history):
        sample = np.log(history[-30:].dropna(axis=1))
        sample -= sample.mean()
        model = PCA().fit(sample)

        for i in range(self._num_components):
            self.plot(
                'Explained Variance Ratio', f"Component {i}", 
                model.explained_variance_ratio_[i]
            )

        factors = np.dot(sample, model.components_.T)[:,:self._num_components]
        factors = sm.add_constant(factors)

        model_by_ticker = {
            ticker: sm.OLS(sample[ticker], factors).fit() 
            for ticker in sample.columns
        }

        resids = pd.DataFrame(
            {ticker: self._apply_kalman_filter(model.resid) for ticker, model in model_by_ticker.items()}
        )

        zscores = ((resids - resids.mean()) / resids.std()).iloc[-1]
        
        selected = zscores[zscores < -self._z_score_threshold]
        
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
            transition_covariance=0.1
        )
        
        state_means, _ = kf.filter(series.values)
        return pd.Series(state_means.flatten(), index=series.index)

~~~
