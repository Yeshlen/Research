"""
Risk scoring models: rule-based heuristic scoring, unsupervised anomaly scoring (Isolation Forest, LOF,
One-Class SVM, autoencoder), and a hybrid of rules and one anomaly model.

Every scorer returns a score where higher means riskier. Rule scores are 0-100 with human-readable
alert reasons. Anomaly scores are converted to a 0-1 percentile against the training data, so both
can be combined on the same scale.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

BANDS = [(0, "Low"), (40, "Medium"), (70, "High"), (90, "Critical")]   # proposal section 4.3

# (name, points, stage, test, reason shown to an analyst)
# Weights are set from scam modus operandi described by banks and SABRIC, not fitted to the labels.
RULES = [
    ("new_recipient", 20, "A", lambda d: d.recipient_known == 0, "first voucher to this recipient"),
    ("recipient_added_now", 10, "A", lambda d: d.recipient_added_in_session == 1, "recipient added in this session"),
    ("amount_unusual", 20, "A", lambda d: d.amount_ratio_to_avg_90d >= 3, "amount 3x+ the customer's 90-day average"),
    ("near_limit", 15, "A", lambda d: d.amount_to_limit_ratio >= 0.8, "amount at 80%+ of the daily limit"),
    ("limit_raised", 20, "A", lambda d: d.limit_changed_24h == 1, "daily limit raised in the last 24h"),
    ("new_device", 15, "A", lambda d: d.new_device_7d == 1, "new device registered in the last 7 days"),
    ("first_cardless", 15, "A", lambda d: d.history_count == 0, "customer's first cardless voucher"),
    ("multi_in_session", 15, "A", lambda d: d.vouchers_in_session >= 1, "several vouchers in one session"),
    ("velocity_24h", 10, "A", lambda d: d.vouchers_24h >= 2, "3+ vouchers within 24h"),
    ("long_session", 10, "A", lambda d: (d.secs_since_login >= 300) | (d.screens_before_create >= 6),
     "long or heavily navigated session (possible coaching)"),
    ("night", 5, "A", lambda d: d.is_night == 1, "created at night"),
    ("fast_cashout", 20, "B", lambda d: d.redeem_delay_min < 15, "cashed out within 15 minutes"),
    ("far_cashout", 15, "B", lambda d: d.redeem_dist_usual_km >= 50, "cashed out 50km+ from usual area"),
]

STAGE_A_FEATURES = [
    "amount", "amount_ratio_to_avg_90d", "amount_ratio_to_max", "amount_to_limit_ratio", "history_count",
    "days_since_last_voucher", "vouchers_24h", "amount_24h", "vouchers_7d", "vouchers_in_session",
    "recipient_known", "is_self_send", "recipient_added_in_session", "secs_since_login",
    "screens_before_create", "hour_deviation", "is_night", "limit_changed_24h", "limit_increase_24h",
    "days_since_device_change", "new_device_7d", "password_reset_7d", "account_age_days",
]
STAGE_B_FEATURES = STAGE_A_FEATURES + ["redeem_delay_min", "redeem_dist_home_km", "redeem_dist_usual_km",
                                       "redeem_is_night", "redeem_at_atm"]
LOG_FEATURES = {"amount", "amount_ratio_to_avg_90d", "amount_ratio_to_max", "history_count",
                "days_since_last_voucher", "amount_24h", "secs_since_login", "days_since_device_change",
                "account_age_days", "limit_increase_24h", "redeem_delay_min", "redeem_dist_home_km",
                "redeem_dist_usual_km"}
# Distance-based models double-count near-duplicate features (EDA: |r| ~ 0.9), so one of each pair is dropped
REDUNDANT = {"amount_to_limit_ratio", "recipient_added_in_session"}


def distance_features(features):
    return [c for c in features if c not in REDUNDANT]


def band(score):
    label = BANDS[0][1]
    for lo, name in BANDS:
        if score >= lo:
            label = name
    return label


# ---------------------------------------------------------------- rules
def rule_score(df, stage="A"):
    """Return (score 0-100, list of fired reasons per row)."""
    stages = {"A"} if stage == "A" else {"A", "B"}
    score = np.zeros(len(df))
    fired = []
    for name, pts, st, test, reason in RULES:
        if st in stages:
            hit = test(df).fillna(False).to_numpy(dtype=bool)
            score += pts * hit
            fired.append((hit, reason))
    reasons = [[r for hit, r in fired if hit[i]] for i in range(len(df))]
    return np.minimum(score, 100), reasons


# ---------------------------------------------------------------- anomaly models
def model_matrix(df, features):
    X = df[features].astype(float).copy()
    for c in features:
        if c in LOG_FEATURES:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X.fillna(0).to_numpy()


class PercentileScorer:
    """Wraps an unsupervised model so scores become 'share of training rows less anomalous than this one'.

    scale: standardise features first (needed by distance and neural models).
    max_fit: fit on a random subsample of this size (LOF and OCSVM scale badly with n);
    the percentile reference still uses every training row.
    """

    def __init__(self, model, features, scale=False, max_fit=None, seed=0):
        self.model, self.features = model, features
        self.scale, self.max_fit, self.seed = scale, max_fit, seed

    def _matrix(self, df):
        X = model_matrix(df, self.features)
        return self.scaler.transform(X) if self.scale else X

    def fit(self, df):
        X = model_matrix(df, self.features)
        if self.scale:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        Xfit = X
        if self.max_fit and len(X) > self.max_fit:
            Xfit = X[np.random.default_rng(self.seed).choice(len(X), self.max_fit, replace=False)]
        self.model.fit(Xfit)
        self.ref = np.sort(-self.model.score_samples(X))          # higher = more anomalous
        return self

    def raw(self, df):
        return -self.model.score_samples(self._matrix(df))

    def score(self, df):
        return np.searchsorted(self.ref, self.raw(df)) / len(self.ref)


class Autoencoder:
    """Under-complete MLP autoencoder; score_samples is minus the mean squared reconstruction error."""

    def __init__(self, hidden=(32, 8, 32), epochs=40, seed=0):
        self.net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=epochs, batch_size=256, random_state=seed)

    def fit(self, X):
        self.net.fit(X, X)
        return self

    def score_samples(self, X):
        return -((self.net.predict(X) - X) ** 2).mean(axis=1)


def isolation_forest(features, seed=0):
    return PercentileScorer(IsolationForest(n_estimators=300, max_samples=256, random_state=seed, n_jobs=-1),
                            features)


def lof(features, n_neighbors=50, seed=0):
    return PercentileScorer(LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1),
                            distance_features(features), scale=True, max_fit=20000, seed=seed)


def one_class_svm(features, nu=0.05, seed=0):
    return PercentileScorer(OneClassSVM(kernel="rbf", nu=nu, gamma="scale"),
                            distance_features(features), scale=True, max_fit=10000, seed=seed)


def autoencoder(features, seed=0):
    return PercentileScorer(Autoencoder(seed=seed), distance_features(features), scale=True, seed=seed)


# ---------------------------------------------------------------- hybrid
def hybrid_score(rule, anomaly, w):
    """Weighted blend of the rule score (scaled to 0-1) and the anomaly percentile."""
    return w * (np.asarray(rule) / 100) + (1 - w) * np.asarray(anomaly)
