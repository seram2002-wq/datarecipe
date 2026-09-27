# # 반도체 공정 불량 예측 기반 화학물질 투입 최적화 및 폐기물 저감 효과 정량화
# **UCI SECOM 데이터 · 불량 예측 → SHAP 핵심변수 → 다목적 최적화(베이지안) → 수율·화학물질·CO₂e 환산**
#
# | 단계 | 내용 | 산출물 |
# |---|---|---|
# | 0 | 환경 설정 · 가정치(Config) | `CONFIG` |
# | 1 | 데이터 로드 · EDA | 불량률, 결측 분포, 시계열 불량 추이 |
# | 2 | 전처리 (학습셋 기준 fit → 누수 방지) | 결측>50%·상수·고상관(>0.95) 변수 제거, 중앙값 대체 |
# | 3 | 불량 예측 모델링 (불균형 대응) | LR / RF / XGBoost / LightGBM × SMOTE·가중치, 시간순 분할 검증 |
# | 4 | 변수중요도 (SHAP + 순열중요도 교차검증) | 상위 20개 공정변수 |
# | 5 | 반응표면법(RSM) | 상위 2개 변수의 불량확률 등고선 |
# | 6 | 다목적 베이지안 최적화 | 불량위험 ↓ · 화학물질 투입지수 ↓ 파레토 프론트, 시나리오 A/B/C |
# | 7 | 효과 시뮬레이션 (몬테카를로) | 수율 %p → 폐기 웨이퍼 → 화학물질 톤 → CO₂e · 비용, P10/P50/P90 |
# | 8 | 결과 요약 · 내보내기 | `outputs/` (그림, CSV, XLSX, JSON) |
#
# ### 데이터 준비
# 1. https://archive.ics.uci.edu/dataset/179/secom 에서 다운로드
# 2. `secom.data`, `secom_labels.data` 를 이 스크립트 옆 `secom/` 폴더에 넣기 (secom.zip 만 두면 자동 압축 해제)
# 3. 실행: python secom_analysis.py  → 결과는 outputs/ (그림 8장, results.xlsx, summary.json)
#
# > 파일이 없을 때 `USE_DEMO_DATA=True`로 두면 SECOM과 **구조만 같은 합성 데이터**(1567×590, 불량 6.6%, 결측·상수열 포함)로 파이프라인을 시험 실행할 수 있습니다. **데모 결과 수치는 제출에 사용하면 안 됩니다.**
#
# ### ⚠️ 정량화의 성격 (심사 대응 문구)
# SECOM의 센서변수는 익명화(F000~F589)되어 있고 화학물질 사용량 컬럼이 없습니다. 따라서 본 분석의 화학물질·폐기물·CO₂e 수치는
# **"모델 추정 불량 감소율 × 공개자료 기반 원단위 가정치"** 로 산출한 **가정 기반 추정치**이며, 모든 가정은 `CONFIG`에 출처와 함께 명시하고 몬테카를로로 불확실성 범위를 함께 제시합니다.


# ## 0. 환경 설정 · 가정치(Config)


import os, sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix, fbeta_score, recall_score, precision_score, balanced_accuracy_score)
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import optuna

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 출력 시 한글·기호 깨짐 방지
except Exception:
    pass
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
optuna.logging.set_verbosity(optuna.logging.WARNING)
set_config(transform_output="pandas")

# 한글 폰트 (Windows: 맑은 고딕 / macOS: AppleGothic / Linux: 나눔고딕)
_fonts = {f.name for f in font_manager.fontManager.ttflist}
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    if _f in _fonts:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 기준 폴더: 스크립트가 있는 폴더 (어디서 실행해도 같은 위치를 봄)
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"; FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 실데이터가 없으면 True (환경변수 SECOM_DEMO=1 로도 설정 가능)
USE_DEMO_DATA = os.environ.get("SECOM_DEMO", "0") == "1"

CONFIG = {
    # ---- 전처리 ----
    "missing_threshold": 0.50,     # 결측비율 50% 초과 변수 제거
    "corr_threshold": 0.95,        # |상관계수| 0.95 초과 쌍 중 하나 제거
    "test_ratio": 0.30,            # 시간순 마지막 30%를 테스트셋 (실제 양산 적용과 동일한 '미래 예측' 조건)
    # ---- 최적화 ----
    "n_controllable": 6,           # SHAP 상위 변수 중 '레시피 제어변수'로 간주할 개수
    "ctrl_max_missing": 0.10,      # 제어변수 후보의 최대 결측비율 (결측이 많으면 '측정 여부' 효과일 수 있어 제외)
    "bound_quantiles": (0.05, 0.95),  # 탐색 범위 = 양품 웨이퍼 관측값의 5~95% 분위 (외삽 방지)
    "max_shift_sd": 0.5,           # 레시피 조정폭 상한: 각 제어변수 σ의 ±0.5배 (현행 운전 범위 안의 현실적 조정)
    "chem_elasticity": 0.10,       # [가정·보수적] 제어변수를 범위 하단→상단으로 옮기면 화학물질 투입이 최대 10% 변한다고 가정
    "chem_elasticity_range": (0.05, 0.30),  # 민감도 분석용 범위
    "n_trials": 400,               # 베이지안 최적화(다목적 TPE) 시행 횟수
}

# ---- 효과 환산 가정치 (★반드시 출처 기입 후 값 교체★) ----
# 값은 '형식 예시'입니다. 출처 예: 환경부 화학물질배출·이동량(PRTR) 정보공개(icis.me.go.kr/prtr),
# 삼성전자·SK하이닉스 지속가능경영보고서의 화학물질 사용량/웨이퍼 생산량, 환경부 폐기물 처리 단가 등
# 형식: (최솟값, 기준값, 최댓값, 단위, 출처)
ASSUMPTIONS = {
    "wafer_starts_per_month": (80_000, 100_000, 120_000, "장/월", "가정: 300mm 중형 팹 투입량 — 기업 공시 CAPA로 교체"),
    "scrap_fraction":         (0.30, 0.50, 0.70, "비율", "가정: 불량 판정 웨이퍼 중 재작업 불가로 폐기되는 비율"),
    "chem_kg_per_wafer":      (5.0, 8.0, 12.0, "kg/장", "TODO: 보고서 화학물질 사용량(톤) ÷ 웨이퍼 생산량(장)"),
    "chem_waste_ratio":       (0.70, 0.85, 0.95, "비율", "가정: 투입 화학물질 중 폐액·폐기물로 배출되는 비율"),
    "waste_cost_krw_per_ton": (300_000, 500_000, 800_000, "원/톤", "TODO: 지정폐기물(폐산·폐알칼리) 위탁처리 단가"),
    "chem_cost_krw_per_kg":   (2_000, 4_000, 8_000, "원/kg", "TODO: 고순도 화학물질 평균 구매단가"),
    "ef_kgco2e_per_kg_chem":  (1.0, 2.0, 3.5, "kgCO2e/kg", "TODO: 화학물질 전과정(LCA) 배출계수 — 환경성적표지/ecoinvent"),
    "remaining_step_fraction":(0.30, 0.50, 0.70, "비율", "가정: 조기검출 시점 이후 남은 공정의 화학물질 투입 비중"),
}

def chem_intensity_from_report(total_chem_ton: float, wafer_output: float) -> float:
    """ESG 보고서 값으로 원단위(kg/장) 계산: 연간 화학물질 사용량(톤) ÷ 연간 웨이퍼 생산량(장)"""
    return total_chem_ton * 1000 / wafer_output

print(pd.DataFrame(ASSUMPTIONS, index=["min", "base", "max", "unit", "source"]).T)
print()


# ## 1. 데이터 로드 · EDA


def load_secom(data_dir: Path):
    X = pd.read_csv(data_dir / "secom.data", sep=r"\s+", header=None)
    X.columns = [f"F{i:03d}" for i in range(X.shape[1])]
    lab = pd.read_csv(data_dir / "secom_labels.data", sep=" ", header=None,
                      names=["label", "ts"], quotechar='"')
    y = (lab["label"] == 1).astype(int).rename("fail")          # 1 = 불량(Fail), 0 = 양품(Pass)
    ts = pd.to_datetime(lab["ts"], dayfirst=True).rename("ts")
    return X, y, ts


def make_demo_secom(n=1567, p=590, fail_rate=0.0664, seed=RANDOM_STATE):
    """SECOM과 구조가 같은 합성 데이터 (시험 실행 전용 — 결과를 제출에 쓰지 말 것)"""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 25))
    X = Z @ rng.normal(size=(25, p)) * 0.6 + rng.normal(size=(n, p))
    inf = rng.choice(p, 8, replace=False)
    s = (X[:, inf] - X[:, inf].mean(0)) / X[:, inf].std(0)
    logit = 1.3*s[:, 0] - 1.0*s[:, 1] + 0.8*s[:, 2]**2 + 0.7*s[:, 3]*s[:, 4] + 0.6*np.maximum(s[:, 5], 0) - 0.5*s[:, 6] + 0.4*s[:, 7]
    logit += rng.logistic(size=n) * 1.2
    y = (logit > np.quantile(logit, 1 - fail_rate)).astype(int)
    X = X * rng.uniform(0.1, 50, p) + rng.uniform(0, 500, p)
    const = rng.choice(np.setdiff1d(np.arange(p), inf), 116, replace=False)
    X[:, const] = np.round(rng.uniform(0, 10, len(const)), 3)
    hi_miss = rng.choice(np.setdiff1d(np.arange(p), inf), 28, replace=False)
    for j in hi_miss:
        X[rng.random(n) < rng.uniform(0.5, 0.92), j] = np.nan
    X[rng.random((n, p)) < 0.02] = np.nan
    X = pd.DataFrame(X, columns=[f"F{i:03d}" for i in range(p)])
    ts = pd.Series(pd.date_range("2008-07-19", "2008-10-17", periods=n), name="ts")
    return X, pd.Series(y, name="fail"), ts


def find_secom_dir(base: Path):
    """secom.data 위치 자동 탐색: secom/ → data/ → 기준 폴더 → 하위 폴더 전체. 압축파일(secom.zip)만 있으면 secom/에 자동 해제"""
    for d in [base / "secom", base / "data", base]:
        if (d / "secom.data").exists() and (d / "secom_labels.data").exists():
            return d
    for f in base.rglob("secom.data"):
        if (f.parent / "secom_labels.data").exists():
            return f.parent
    for z in list(base.glob("*.zip")) + list((base / "data").glob("*.zip")):
        import zipfile
        with zipfile.ZipFile(z) as zf:
            if any(n.endswith("secom.data") for n in zf.namelist()):
                zf.extractall(base / "secom"); print(f"압축 해제: {z.name} → secom/")
                return find_secom_dir(base)
    return None

SECOM_DIR = find_secom_dir(BASE_DIR)
if SECOM_DIR is not None and not USE_DEMO_DATA:
    print("데이터 폴더:", SECOM_DIR)
    X, y, ts = load_secom(SECOM_DIR); DATA_MODE = "SECOM (UCI)"
elif USE_DEMO_DATA:
    X, y, ts = make_demo_secom(); DATA_MODE = "DEMO (합성 데이터 — 제출 사용 금지)"
else:
    raise FileNotFoundError(f"secom.data / secom_labels.data 를 찾지 못했습니다.\n"
                            f"  {BASE_DIR} 폴더(또는 그 안의 secom 폴더)에 두 파일이나 secom.zip 을 넣어주세요.")

print(f"데이터: {DATA_MODE}")
print(f"웨이퍼 배치 수: {X.shape[0]:,} / 센서변수: {X.shape[1]} / 불량: {y.sum()} ({y.mean():.2%})")
print(f"기간: {ts.min():%Y-%m-%d} ~ {ts.max():%Y-%m-%d}")


miss = X.isna().mean()
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].bar(["양품(Pass)", "불량(Fail)"], y.value_counts().sort_index().values, color=["#4C78A8", "#E45756"])
ax[0].set_title(f"클래스 분포 (불량 {y.mean():.1%} → 심한 불균형)")
ax[1].hist(miss, bins=40, color="#72B7B2"); ax[1].axvline(CONFIG["missing_threshold"], color="k", ls="--")
ax[1].set_title("변수별 결측비율 분포"); ax[1].set_xlabel("결측비율")
weekly = pd.DataFrame({"ts": ts, "fail": y}).set_index("ts").resample("W")["fail"].mean()
ax[2].plot(weekly.index, weekly.values * 100, marker="o", color="#E45756")
ax[2].set_title("주간 불량률 추이 (%)"); ax[2].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(FIG_DIR / "01_eda.png", bbox_inches="tight"); plt.close()

print(f"결측>50% 변수: {(miss > 0.5).sum()}개 / 상수 변수: {(X.nunique() <= 1).sum()}개 / 결측 없는 변수: {(miss == 0).sum()}개")


# ## 2. 전처리
# - **시간순 분할**: 앞 70% 학습 / 뒤 30% 테스트. 무작위 분할보다 보수적이지만 '과거 데이터로 미래 웨이퍼를 예측'하는 실제 적용 조건과 같습니다.
# - 결측·상수·고상관 필터, 중앙값 대체는 모두 **학습셋에서만 fit** 하고 파이프라인 안에 넣어 교차검증 시 데이터 누수를 막습니다.
# - SMOTE도 파이프라인 안에서 학습 fold에만 적용됩니다.


class SecomCleaner(BaseEstimator, TransformerMixin):
    """결측비율·상수·고상관 변수 제거 (학습 데이터 기준)"""
    def __init__(self, missing_threshold=0.5, corr_threshold=0.95):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        keep = X.columns[X.isna().mean() <= self.missing_threshold]
        keep = [c for c in keep if X[c].nunique(dropna=True) > 1 and X[c].std(skipna=True) > 1e-8]
        corr = X[keep].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        drop = {c for c in upper.columns if (upper[c] > self.corr_threshold).any()}
        self.features_ = [c for c in keep if c not in drop]
        self.n_removed_ = {"결측": int((X.isna().mean() > self.missing_threshold).sum()),
                           "상수·고상관": int(X.shape[1] - (X.isna().mean() > self.missing_threshold).sum() - len(self.features_))}
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.features_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.features_)


order = np.argsort(ts.values, kind="stable")
X, y, ts = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), ts.iloc[order].reset_index(drop=True)
cut = int(len(X) * (1 - CONFIG["test_ratio"]))
X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
print(f"학습: {len(X_train)}건 (불량 {y_train.sum()}, {y_train.mean():.2%}) ~ {ts.iloc[cut-1]:%Y-%m-%d}")
print(f"테스트: {len(X_test)}건 (불량 {y_test.sum()}, {y_test.mean():.2%}) {ts.iloc[cut]:%Y-%m-%d} ~")

_c = SecomCleaner(CONFIG["missing_threshold"], CONFIG["corr_threshold"]).fit(X_train)
print(f"변수 {X.shape[1]}개 → {len(_c.features_)}개 (제거: {_c.n_removed_})")


# ## 3. 불량 예측 모델링
# 평가지표는 불균형 데이터에 적합한 **PR-AUC(Average Precision)** 를 주지표로, ROC-AUC를 보조지표로 씁니다. (불량 비율이 약 6.6%라 PR-AUC의 무작위 기준선은 ≈0.066)


pos_w = (y_train == 0).sum() / (y_train == 1).sum()
clean = lambda: SecomCleaner(CONFIG["missing_threshold"], CONFIG["corr_threshold"])
imp = lambda: SimpleImputer(strategy="median")
smote = lambda: SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

MODELS = {
    "LR + SMOTE": Pipeline([("clean", clean()), ("impute", imp()), ("scale", StandardScaler()), ("smote", smote()),
                            ("model", LogisticRegression(C=0.05, max_iter=3000))]),
    "RF + SMOTE": Pipeline([("clean", clean()), ("impute", imp()), ("smote", smote()),
                            ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                                             n_jobs=-1, random_state=RANDOM_STATE))]),
    "XGB + SMOTE": Pipeline([("clean", clean()), ("impute", imp()), ("smote", smote()),
                             ("model", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                                                     colsample_bytree=0.5, eval_metric="logloss", n_jobs=-1,
                                                     random_state=RANDOM_STATE))]),
    "XGB + 가중치": Pipeline([("clean", clean()), ("impute", imp()),
                             ("model", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                                                     colsample_bytree=0.5, scale_pos_weight=pos_w, eval_metric="logloss",
                                                     n_jobs=-1, random_state=RANDOM_STATE))]),
    "LGBM + SMOTE": Pipeline([("clean", clean()), ("impute", imp()), ("smote", smote()),
                              ("model", LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.03, subsample=0.8,
                                                       subsample_freq=1, colsample_bytree=0.5, min_child_samples=10,
                                                       random_state=RANDOM_STATE, verbose=-1))]),
    "LGBM + 가중치": Pipeline([("clean", clean()), ("impute", imp()),
                              ("model", LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.03, subsample=0.8,
                                                       subsample_freq=1, colsample_bytree=0.5, min_child_samples=10,
                                                       scale_pos_weight=pos_w, random_state=RANDOM_STATE, verbose=-1))]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows, oof, fitted, test_proba = [], {}, {}, {}
for name, pipe in MODELS.items():
    r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["average_precision", "roc_auc"], n_jobs=1)
    oof[name] = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    fitted[name] = pipe.fit(X_train, y_train)
    test_proba[name] = fitted[name].predict_proba(X_test)[:, 1]
    rows.append({"모델": name,
                 "CV PR-AUC": r["test_average_precision"].mean(), "CV PR-AUC sd": r["test_average_precision"].std(),
                 "CV ROC-AUC": r["test_roc_auc"].mean(),
                 "Test PR-AUC": average_precision_score(y_test, test_proba[name]),
                 "Test ROC-AUC": roc_auc_score(y_test, test_proba[name])})
    print(f"  ✓ {name}")

cv_table = pd.DataFrame(rows).set_index("모델").sort_values("CV PR-AUC", ascending=False)
print(cv_table.round(3))
print()


# **모델 선택 규칙**: SHAP·최적화에 쓸 모델은 트리 계열 중 **학습셋 CV PR-AUC 최고** 모델로 고정합니다. (테스트셋 성능으로 고르면 테스트 결과가 낙관적으로 부풀려지므로 선택에는 쓰지 않음)


tree_names = [n for n in cv_table.index if not n.startswith("LR")]
BEST = tree_names[0]
print("선택 모델:", BEST)

# 임계값: 학습셋 OOF 예측에서 F2 최대 (불량 놓침(FN)의 비용이 오탐(FP)보다 크므로 재현율 가중)
ths = np.linspace(0.01, 0.9, 180)
f2 = [fbeta_score(y_train, oof[BEST] >= t, beta=2, zero_division=0) for t in ths]
THRESH = float(ths[int(np.argmax(f2))])

p_best = test_proba[BEST]
pred = (p_best >= THRESH).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
TEST_METRICS = {"model": BEST, "threshold": THRESH,
                "PR_AUC": average_precision_score(y_test, p_best), "ROC_AUC": roc_auc_score(y_test, p_best),
                "recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
                "specificity": tn / (tn + fp), "balanced_acc": balanced_accuracy_score(y_test, pred),
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
print(pd.Series(TEST_METRICS).to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, p in test_proba.items():
    fpr, tpr, _ = roc_curve(y_test, p); pr, rc, _ = precision_recall_curve(y_test, p)
    lw = 2.5 if name == BEST else 1
    ax[0].plot(fpr, tpr, lw=lw, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
    ax[1].plot(rc, pr, lw=lw, label=f"{name} ({average_precision_score(y_test, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC (테스트셋)", xlabel="FPR", ylabel="TPR")
ax[1].axhline(y_test.mean(), color="k", ls=":"); ax[1].set(title="Precision-Recall (테스트셋)", xlabel="Recall", ylabel="Precision")
for a in ax: a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG_DIR / "02_model_curves.png", bbox_inches="tight"); plt.close()


# ## 4. 변수중요도 — SHAP + 순열중요도 교차검증


best_pipe = fitted[BEST]
pre = SkPipeline([(n, s) for n, s in best_pipe.steps if n in ("clean", "impute", "scale")])
model = best_pipe.named_steps["model"]
X_train_t, X_test_t = pre.transform(X_train), pre.transform(X_test)
X_all_t = pd.concat([X_train_t, X_test_t])

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_all_t)
if isinstance(sv, list): sv = sv[1]
if sv.ndim == 3: sv = sv[:, :, 1]

shap_imp = pd.Series(np.abs(sv).mean(0), index=X_all_t.columns).sort_values(ascending=False)
direction = pd.Series([np.corrcoef(X_all_t[c], sv[:, i])[0, 1] for i, c in enumerate(X_all_t.columns)],
                      index=X_all_t.columns)
TOP20 = shap_imp.head(20).index.tolist()
top_table = pd.DataFrame({"mean|SHAP|": shap_imp[TOP20],
                          "영향 방향": np.where(direction[TOP20] > 0, "값↑ → 불량위험↑", "값↑ → 불량위험↓"),
                          "결측비율": X[TOP20].isna().mean()})
shap.summary_plot(sv, X_all_t, max_display=20, show=False, plot_size=(9, 7))
plt.title(f"SHAP 요약 — {BEST}"); plt.savefig(FIG_DIR / "03_shap_summary.png", bbox_inches="tight"); plt.close()
print(top_table.round(4))
print()


# 순열중요도 (SHAP 상위 30개 대상, 테스트셋 PR-AUC 하락폭) — SHAP 결과의 강건성 확인
rng = np.random.default_rng(RANDOM_STATE)
base_ap = average_precision_score(y_test, model.predict_proba(X_test_t)[:, 1])
perm = {}
for c in shap_imp.head(30).index:
    drops = []
    for _ in range(10):
        Xp = X_test_t.copy(); Xp[c] = rng.permutation(Xp[c].values)
        drops.append(base_ap - average_precision_score(y_test, model.predict_proba(Xp)[:, 1]))
    perm[c] = np.mean(drops)
perm = pd.Series(perm).sort_values(ascending=False)
overlap = len(set(perm.head(10).index) & set(shap_imp.head(10).index))
print(f"SHAP Top10 ∩ 순열중요도 Top10 = {overlap}/10")

top4 = TOP20[:4]
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
for a, c in zip(ax, top4):
    i = X_all_t.columns.get_loc(c)
    a.scatter(X_all_t[c], sv[:, i], s=6, alpha=0.5, c=np.r_[y_train, y_test], cmap="coolwarm")
    a.axhline(0, color="k", lw=0.6); a.set(title=f"{c}", xlabel="센서값", ylabel="SHAP (불량 기여)")
plt.suptitle("상위 4개 변수 SHAP 의존도 (빨강=실제 불량)", y=1.04)
plt.tight_layout(); plt.savefig(FIG_DIR / "04_shap_dependence.png", bbox_inches="tight"); plt.close()


# > **해석 시 유의**: SHAP은 모델이 학습한 '상관적 기여도'이며 인과관계가 아닙니다. 실제 적용 시 상위 변수를 공정 엔지니어와 함께 물리적 의미(가스유량·온도·압력·시간 등)에 매핑하고, DOE(실험계획법)로 인과 효과를 확인해야 합니다.


# ## 5. 제어변수 정의 · 반응표면법(RSM)
# SHAP 상위 변수 중 `n_controllable`개를 **레시피로 조정 가능한 제어변수**로 가정합니다. 실제로는 엔지니어가 변수의 물리적 의미를 보고 선택(예: 측정값이 아닌 설정값)해야 하므로, 아래 `CONTROLLABLE`, `CHEM_DIRECTION`을 수정할 수 있게 두었습니다.
#
# - **제외 규칙**: ① 결측비율 > 10%(값 자체보다 '측정됐는지 여부'가 불량과 연관됐을 가능성), ② 0값 비중 > 10%(센서 꺼짐·모드 표시로 의심), ③ 고유값 < 20개(범주형·상태값)인 변수는 레시피로 연속 조정할 수 없다고 보고 제어변수에서 제외
# - 탐색 범위: 학습셋 **양품** 웨이퍼 관측값의 5~95% 분위 → 관측되지 않은 영역으로의 외삽 방지
# - `CHEM_DIRECTION`: +1 = 값이 클수록 화학물질 투입↑ (예: 유량·노출시간), −1 = 반대, 0 = 무관


def is_adjustable(c):
    s = X[c].dropna()
    return (X[c].isna().mean() <= CONFIG["ctrl_max_missing"]   # 결측 과다 → 센서값보다 '측정 여부'가 신호일 가능성
            and (s == 0).mean() <= 0.10 and s.nunique() >= 20)

excluded = [c for c in shap_imp.index[:30] if not is_adjustable(c)]
CONTROLLABLE = [c for c in shap_imp.index[:30] if is_adjustable(c)][:CONFIG["n_controllable"]]
CHEM_DIRECTION = {c: +1 for c in CONTROLLABLE}          # ← 변수 의미에 맞게 수정
def _why(c):
    r = []
    if X[c].isna().mean() > CONFIG["ctrl_max_missing"]: r.append(f"결측 {X[c].isna().mean():.0%}")
    if (X[c].dropna() == 0).mean() > 0.10: r.append(f"0값 {(X[c].dropna() == 0).mean():.0%}")
    if X[c].nunique() < 20: r.append(f"고유값 {X[c].nunique()}개")
    return ", ".join(r)
print("제어변수:", CONTROLLABLE)
print("제외:", {c: _why(c) for c in excluded})

pass_train = X_train_t[y_train.values == 0]
qlo, qhi = CONFIG["bound_quantiles"]
BOUNDS = {c: (float(pass_train[c].quantile(qlo)), float(pass_train[c].quantile(qhi))) for c in CONTROLLABLE}
# 조정 단위: 양품 웨이퍼의 로버스트 표준편차 (IQR/1.349) — 이상치에 둔감
SD = {c: float((pass_train[c].quantile(.75) - pass_train[c].quantile(.25)) / 1.349) or float(pass_train[c].std())
      for c in CONTROLLABLE}

print(pd.DataFrame({"하한(양품 5%)": {c: b[0] for c, b in BOUNDS.items()}, "중앙값": {c: X_train_t[c].median() for c in CONTROLLABLE},
              "상한(양품 95%)": {c: b[1] for c, b in BOUNDS.items()}, "조정단위 σ": SD,
              "결측비율": {c: X[c].isna().mean() for c in CONTROLLABLE}, "0값 비중": {c: (X[c] == 0).mean() for c in CONTROLLABLE}, "화학물질 방향": CHEM_DIRECTION}).round(4))
print()


# 고전적 RSM: 상위 2개 제어변수의 2차 다항 로지스틱 반응표면 (관측 데이터 직접 적합)
v1, v2 = CONTROLLABLE[:2]
D = X_all_t[[v1, v2]].copy(); yy = np.r_[y_train, y_test]
rsm = SkPipeline([("poly", PolynomialFeatures(2, include_bias=False)), ("sc", StandardScaler()),
                  ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))]).fit(D, yy)
coef = pd.Series(rsm.named_steps["lr"].coef_[0], index=rsm.named_steps["poly"].get_feature_names_out([v1, v2]))
print("RSM 2차 모형 계수(표준화):\n", coef.round(3).to_string())

g1 = np.linspace(*D[v1].quantile([0.01, 0.99]), 120); g2 = np.linspace(*D[v2].quantile([0.01, 0.99]), 120)
G1, G2 = np.meshgrid(g1, g2)
P = rsm.predict_proba(pd.DataFrame({v1: G1.ravel(), v2: G2.ravel()}))[:, 1].reshape(G1.shape)
fig, ax = plt.subplots(figsize=(7, 5.5))
cs = ax.contourf(G1, G2, P, levels=15, cmap="RdYlGn_r", alpha=0.85); plt.colorbar(cs, label="상대 불량위험 (RSM)")
ax.scatter(D[v1][yy == 0], D[v2][yy == 0], s=4, c="k", alpha=0.25, label="양품")
ax.scatter(D[v1][yy == 1], D[v2][yy == 1], s=14, c="red", marker="x", label="불량")
ax.add_patch(plt.Rectangle((BOUNDS[v1][0], BOUNDS[v2][0]), BOUNDS[v1][1] - BOUNDS[v1][0], BOUNDS[v2][1] - BOUNDS[v2][0],
                           fill=False, ls="--", lw=1.5, ec="navy", label="탐색 범위(양품 5~95%)"))
ax.set(xlabel=v1, ylabel=v2, title="반응표면 — 상위 2개 제어변수"); ax.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.savefig(FIG_DIR / "05_rsm_contour.png", bbox_inches="tight"); plt.close()


# ## 6. 다목적 베이지안 최적화 (불량위험 ↓ · 화학물질 투입지수 ↓)
# **레시피 = 현행 대비 조정폭(오프셋)**: 각 제어변수를 웨이퍼별 현재값에서 $\delta_j \cdot \sigma_j$ 만큼 이동($|\delta_j| \le$ `max_shift_sd`)하고, 양품 관측범위 밖으로는 더 밀려나지 않게 제한합니다.
# 모든 웨이퍼를 하나의 고정값으로 덮어쓰는 방식은 트리 모델이 드물게 본 영역(예: 0값 구간)으로 전체를 몰아넣어 불량 감소를 90% 이상으로 과대추정하므로 쓰지 않습니다.
#
# - **목적함수 1 — 상대 불량위험(RR)** = 조정 후 기대 불량 수 ÷ 현행 기대 불량 수 (테스트 기간 웨이퍼 기준). SMOTE로 왜곡된 확률은 학습셋 OOF 예측으로 **Platt 보정**한 뒤 사용합니다.
# - **목적함수 2 — 화학물질 투입지수(CI)** = $1 + \alpha \cdot \frac{1}{K}\sum_j d_j \frac{\delta_j}{2\,\delta_{max}}$ (현행 = 1, $\alpha$ = `chem_elasticity` 가정)
# - **알고리즘**: Optuna 다목적 TPE (베이지안 최적화) → 파레토 프론트


_logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
calib = LogisticRegression(C=1e4).fit(_logit(oof[BEST]).reshape(-1, 1), y_train)
def cal_proba(Xt):
    return calib.predict_proba(_logit(model.predict_proba(Xt)[:, 1]).reshape(-1, 1))[:, 1]
print(f"보정 전 테스트 평균 예측확률 {model.predict_proba(X_test_t)[:, 1].mean():.3f} → 보정 후 {cal_proba(X_test_t).mean():.3f} (실제 불량률 {y_test.mean():.3f})")

DMAX = CONFIG["max_shift_sd"]

def apply_recipe(delta, Xref=X_test_t):
    Xs = Xref.copy()
    for c in CONTROLLABLE:
        x = Xref[c].values; lo, hi = BOUNDS[c]
        Xs[c] = np.clip(x + delta[c] * SD[c], np.minimum(x, lo), np.maximum(x, hi))
    return Xs

def chem_index(delta):
    return 1 + CONFIG["chem_elasticity"] * sum(CHEM_DIRECTION[c] * delta[c] / (2 * DMAX) for c in CONTROLLABLE) / len(CONTROLLABLE)

ZERO = {c: 0.0 for c in CONTROLLABLE}
P_BASE = cal_proba(X_test_t)
BASE_RISK = P_BASE.mean()

def objective(trial):
    d = {c: trial.suggest_float(c, -DMAX, DMAX) for c in CONTROLLABLE}
    return cal_proba(apply_recipe(d)).mean() / BASE_RISK, chem_index(d)

study = optuna.create_study(directions=["minimize", "minimize"],
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True, n_startup_trials=40))
study.enqueue_trial(ZERO)                # 현행 레시피(조정 없음)를 기준점으로 포함
study.optimize(objective, n_trials=CONFIG["n_trials"], show_progress_bar=False)

trials = pd.DataFrame([{**t.params, "RR": t.values[0], "CI": t.values[1]} for t in study.trials if t.values])
pareto = pd.DataFrame([{**t.params, "RR": t.values[0], "CI": t.values[1]} for t in study.best_trials]).sort_values("CI")
print(f"시행 {len(trials)}회, 파레토 해 {len(pareto)}개")


# 시나리오 선정
#  A 수율우선   : 화학물질 증가 없이(CI ≤ 1.0) 불량위험 최소
#  C 절감우선   : 불량위험을 현행 이하로 유지(RR ≤ 1.0)하면서 CI 최소
#  B 균형(knee) : 파레토 프론트에서 (RR, CI) 정규화 원점 거리 최소
def pick(df, cond, key):
    d = df[cond(df)]
    return d.loc[d[key].idxmin()] if len(d) else None

scA = pick(pareto, lambda d: d.CI <= 1.0, "RR")
scC = pick(pareto, lambda d: d.RR <= 1.0, "CI")
nr = (pareto.RR - pareto.RR.min()) / (pareto.RR.max() - pareto.RR.min() + 1e-12)
nc = (pareto.CI - pareto.CI.min()) / (pareto.CI.max() - pareto.CI.min() + 1e-12)
scB = pareto.loc[np.hypot(nr, nc).idxmin()]
SCENARIOS = {k: v for k, v in {"A 수율우선": scA, "B 균형": scB, "C 절감우선": scC}.items() if v is not None}

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.scatter(trials.CI, trials.RR, s=10, c="lightgray", label="탐색한 레시피")
ax.plot(pareto.CI, pareto.RR, "o-", c="#4C78A8", ms=4, label="파레토 프론트")
ax.scatter([1], [1], marker="*", s=250, c="k", label="현행 레시피", zorder=5)
for (k, s), col in zip(SCENARIOS.items(), ["#E45756", "#F58518", "#54A24B"]):
    ax.scatter(s.CI, s.RR, s=140, c=col, edgecolor="k", zorder=6, label=f"{k} (RR {s.RR:.2f}, CI {s.CI:.3f})")
ax.axhline(1, c="k", lw=0.6, ls=":"); ax.axvline(1, c="k", lw=0.6, ls=":")
ax.set(xlabel="화학물질 투입지수 CI (현행=1)", ylabel="상대 불량위험 RR (현행=1)", title="다목적 최적화 결과 — 파레토 프론트")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG_DIR / "06_pareto.png", bbox_inches="tight"); plt.close()

scen_table = pd.DataFrame({k: {**{f"{c} 조정(σ)": v[c] for c in CONTROLLABLE},
                               **{f"{c} 조정(원단위)": v[c] * SD[c] for c in CONTROLLABLE}, "RR": v.RR, "CI": v.CI}
                           for k, v in SCENARIOS.items()})
scen_table.insert(0, "현행", {**{f"{c} 조정(σ)": 0 for c in CONTROLLABLE}, **{f"{c} 조정(원단위)": 0 for c in CONTROLLABLE}, "RR": 1.0, "CI": 1.0})
print(scen_table.round(4))
print()


# 강건성: 부트스트랩으로 RR의 신뢰구간 (테스트 웨이퍼 재표집) — 효과 시뮬레이션의 불확실성 입력으로 사용
def boot_rr(delta, B=500):
    po = cal_proba(apply_recipe(delta))
    idx = np.random.default_rng(RANDOM_STATE).integers(0, len(P_BASE), (B, len(P_BASE)))
    return po[idx].mean(1) / P_BASE[idx].mean(1)

RR_BOOT = {k: boot_rr({c: s[c] for c in CONTROLLABLE}) for k, s in SCENARIOS.items()}
print(pd.DataFrame({k: {"RR 점추정": SCENARIOS[k].RR, "RR 2.5%": np.quantile(v, .025), "RR 97.5%": np.quantile(v, .975)}
              for k, v in RR_BOOT.items()}).T.round(3))
print()


# ## 7. 효과 시뮬레이션 — 수율 → 폐기 웨이퍼 → 화학물질 → CO₂e · 비용
# 세 가지 경로로 절감 효과를 분리해 산정합니다 (중복 계산 방지).
#
# | 경로 | 산식 |
# |---|---|
# | ① 불량 감소 (레시피 최적화) | 연간 투입 × 불량률 × (1−RR) × 폐기비율 × 원단위 |
# | ② 투입량 감소 (레시피 최적화) | 연간 투입 × 원단위 × (1−CI) |
# | ③ 조기 검출 (예측모델 가상계측) | 연간 투입 × 최적화 후 불량률 × 폐기비율 × 재현율 × 잔여공정 비중 × 원단위 |
#
# 화학물질 절감량 → 폐기물 감소 = 절감량 × 폐기물 전환율, CO₂e = 절감량 × 배출계수, 비용 = 구매비 + 처리비.


BASE_FAIL = float(y.mean())
RECALL = TEST_METRICS["recall"]

def effect(p, rr, ci, recall=RECALL, fail=BASE_FAIL):
    annual = p["wafer_starts_per_month"] * 12
    fail_new = fail * rr
    scrap_avoided = annual * (fail - fail_new) * p["scrap_fraction"]
    chem1 = scrap_avoided * p["chem_kg_per_wafer"]
    chem2 = annual * p["chem_kg_per_wafer"] * max(0.0, 1 - ci)
    chem3 = annual * fail_new * p["scrap_fraction"] * recall * p["remaining_step_fraction"] * p["chem_kg_per_wafer"]
    chem = (chem1 + chem2 + chem3) / 1000                                  # 톤
    return {"수율 개선(%p)": (fail - fail_new) * 100,
            "폐기 웨이퍼 감소(장/년)": scrap_avoided,
            "화학물질 절감(톤/년)": chem,
            "  ①불량감소": chem1 / 1000, "  ②투입감소": chem2 / 1000, "  ③조기검출": chem3 / 1000,
            "유해폐기물 감소(톤/년)": chem * p["chem_waste_ratio"],
            "CO2e 감축(tCO2e/년)": chem * p["ef_kgco2e_per_kg_chem"],
            "비용 절감(억원/년)": (chem * 1000 * p["chem_cost_krw_per_kg"] + chem * p["chem_waste_ratio"] * p["waste_cost_krw_per_ton"]) / 1e8}

BASE_P = {k: v[1] for k, v in ASSUMPTIONS.items()}
effect_table = pd.DataFrame({k: effect(BASE_P, s.RR, s.CI) for k, s in SCENARIOS.items()})
print(f"기준 불량률 {BASE_FAIL:.2%}, 조기검출 재현율 {RECALL:.2f} (테스트셋), 가정치=기준값")
print(effect_table.round(2))
print()


# 몬테카를로 (가정치: 삼각분포(min, base, max), RR: 부트스트랩 분포) — 권장 시나리오 기준
REC = "B 균형" if "B 균형" in SCENARIOS else list(SCENARIOS)[0]
N = 10_000; rng = np.random.default_rng(RANDOM_STATE)
draws = {k: rng.triangular(v[0], v[1], v[2], N) for k, v in ASSUMPTIONS.items()}
rr_d = rng.choice(RR_BOOT[REC], N)
mc = pd.DataFrame([effect({k: draws[k][i] for k in draws}, rr_d[i], SCENARIOS[REC].CI) for i in range(N)])

KPIS = ["수율 개선(%p)", "폐기 웨이퍼 감소(장/년)", "화학물질 절감(톤/년)", "유해폐기물 감소(톤/년)", "CO2e 감축(tCO2e/년)", "비용 절감(억원/년)"]
mc_table = mc[KPIS].quantile([0.1, 0.5, 0.9]).T; mc_table.columns = ["P10", "P50", "P90"]

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, k, col in zip(ax, ["화학물질 절감(톤/년)", "CO2e 감축(tCO2e/년)", "비용 절감(억원/년)"], ["#4C78A8", "#54A24B", "#F58518"]):
    a.hist(mc[k], bins=60, color=col, alpha=0.8)
    for q, ls in zip([0.1, 0.5, 0.9], [":", "-", ":"]): a.axvline(mc[k].quantile(q), c="k", ls=ls)
    a.set_title(f"{k}\nP50 {mc[k].median():,.1f} (P10~P90 {mc[k].quantile(.1):,.1f}~{mc[k].quantile(.9):,.1f})", fontsize=10)
plt.suptitle(f"몬테카를로 {N:,}회 — 시나리오 {REC}", y=1.05)
plt.tight_layout(); plt.savefig(FIG_DIR / "07_monte_carlo.png", bbox_inches="tight"); plt.close()
print(mc_table.round(2))
print()


# 토네이도 민감도: 가정치 하나씩 min/max로 바꿨을 때 화학물질 절감량 변화 → 어떤 원단위를 가장 정확히 조사해야 하는지 우선순위
KEY = "화학물질 절감(톤/년)"
base_val = effect(BASE_P, SCENARIOS[REC].RR, SCENARIOS[REC].CI)[KEY]
sens = []
for k, (lo, _, hi, *_ ) in ASSUMPTIONS.items():
    vals = [effect({**BASE_P, k: v}, SCENARIOS[REC].RR, SCENARIOS[REC].CI)[KEY] for v in (lo, hi)]
    sens.append((k, min(vals) - base_val, max(vals) - base_val))
rr_lo, rr_hi = np.quantile(RR_BOOT[REC], [.025, .975])
vals = [effect(BASE_P, r, SCENARIOS[REC].CI)[KEY] for r in (rr_hi, rr_lo)]
sens.append(("RR (모델 불확실성)", min(vals) - base_val, max(vals) - base_val))
_ci = lambda a: 1 + (SCENARIOS[REC].CI - 1) * a / CONFIG["chem_elasticity"]   # 탄성계수 변경 시 CI 재계산
vals = [effect(BASE_P, SCENARIOS[REC].RR, _ci(a))[KEY] for a in CONFIG["chem_elasticity_range"]]
sens.append(("chem_elasticity (CI 가정)", min(vals) - base_val, max(vals) - base_val))
sens = sorted(sens, key=lambda s: s[2] - s[1])

fig, ax = plt.subplots(figsize=(8, 4.5))
for i, (k, lo, hi) in enumerate(sens):
    ax.barh(i, hi, color="#4C78A8"); ax.barh(i, lo, color="#E45756")
ax.set_yticks(range(len(sens))); ax.set_yticklabels([s[0] for s in sens]); ax.axvline(0, c="k", lw=0.8)
ax.set(xlabel=f"{KEY} 변화 (기준 {base_val:,.1f}톤)", title="민감도 분석 (토네이도)")
plt.tight_layout(); plt.savefig(FIG_DIR / "08_tornado.png", bbox_inches="tight"); plt.close()


# ## 8. 결과 요약 · 내보내기


s = SCENARIOS[REC]; e = effect_table[REC]
summary = {
    "data_mode": DATA_MODE,
    "n_wafers": int(len(X)), "n_features_raw": int(X.shape[1]), "n_features_used": len(pre.named_steps["clean"].features_),
    "base_fail_rate": BASE_FAIL,
    "model": TEST_METRICS,
    "top10_features": TOP20[:10], "shap_perm_overlap_top10": overlap,
    "controllable": CONTROLLABLE,
    "recommended_scenario": REC, "RR": float(s.RR), "CI": float(s.CI),
    "effect_base": {k: float(v) for k, v in e.items()},
    "effect_mc_P10_P50_P90": {k: [float(x) for x in mc_table.loc[k]] for k in KPIS},
}
print(f"""
[결과 요약 — {DATA_MODE}]
■ 데이터: 웨이퍼 {len(X):,}건, 센서 {X.shape[1]}개 → 전처리 후 {summary['n_features_used']}개, 기준 불량률 {BASE_FAIL:.2%}
■ 불량 예측: {BEST} | 테스트(시간순 미래구간) PR-AUC {TEST_METRICS['PR_AUC']:.3f} (무작위 {y_test.mean():.3f}), ROC-AUC {TEST_METRICS['ROC_AUC']:.3f}, 재현율 {TEST_METRICS['recall']:.2f}
■ 핵심 변수(SHAP Top5): {', '.join(TOP20[:5])}  (순열중요도 Top10 일치 {overlap}/10)
■ 권장 레시피({REC}): 상대 불량위험 {s.RR:.2f}배, 화학물질 투입지수 {s.CI:.2f}
■ 기대효과(기준 가정): 수율 +{e['수율 개선(%p)']:.2f}%p → 폐기 웨이퍼 {e['폐기 웨이퍼 감소(장/년)']:,.0f}장/년 감소
   → 화학물질 {e['화학물질 절감(톤/년)']:,.1f}톤/년 절감 (P10~P90: {mc_table.loc['화학물질 절감(톤/년)','P10']:,.1f}~{mc_table.loc['화학물질 절감(톤/년)','P90']:,.1f})
      = ①불량감소 {e['  ①불량감소']:,.1f} + ③조기검출 {e['  ③조기검출']:,.1f} (모델 근거)  + ②투입감소 {e['  ②투입감소']:,.1f} (CI 탄성계수 가정 의존)
   → 유해폐기물 {e['유해폐기물 감소(톤/년)']:,.1f}톤, CO2e {e['CO2e 감축(tCO2e/년)']:,.1f}t, 비용 {e['비용 절감(억원/년)']:,.1f}억원/년
※ 화학물질·CO2e·비용은 공개자료 원단위 가정 기반 추정치입니다 (ASSUMPTIONS 참조).
""")

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
with pd.ExcelWriter(OUT_DIR / "results.xlsx") as w:
    cv_table.round(4).to_excel(w, sheet_name="1_모델비교")
    top_table.round(5).to_excel(w, sheet_name="2_SHAP_Top20")
    scen_table.round(5).to_excel(w, sheet_name="3_레시피시나리오")
    pareto.round(5).to_excel(w, sheet_name="4_파레토", index=False)
    effect_table.round(3).to_excel(w, sheet_name="5_기대효과")
    mc_table.round(3).to_excel(w, sheet_name="6_몬테카를로")
    pd.DataFrame(ASSUMPTIONS, index=["min", "base", "max", "unit", "source"]).T.to_excel(w, sheet_name="7_가정치")
print("저장 완료:", sorted(p.name for p in OUT_DIR.rglob("*") if p.is_file()))


# ### 한계 및 후속 과제 (제출물에 함께 기재 권장)
# 1. **익명 변수** — SECOM 변수의 물리적 의미가 없어 '제어변수'와 화학물질 방향(`CHEM_DIRECTION`)은 가정입니다. 실공정에서는 변수-설비 매핑 후 재선정해야 합니다.
# 2. **상관 ≠ 인과** — 최적 레시피는 모델 기반 '후보'이며, DOE·파일럿 런으로 검증 후 적용해야 합니다.
# 3. **원단위 가정** — 효과 크기는 `chem_kg_per_wafer`·`scrap_fraction`·`chem_elasticity`에 민감(토네이도 참조)하므로 이 값들의 출처 확보가 신뢰도의 핵심입니다. 특히 ②투입감소 경로는 탄성계수 가정에 전적으로 의존하므로, 보수적으로 제시하려면 ①+③만 '핵심 효과'로, ②는 '추가 잠재효과'로 분리해 보고하는 것을 권장합니다.
# 4. **데이터 규모** — 1,567건·불량 104건의 소규모 데이터로 모델 성능 변동이 큽니다. 부트스트랩·CV 표준편차를 함께 보고합니다.
# 5. **확장** — 실시간 가상계측(VM) 연계, 화학물질 재이용(회수·정제) 공정과의 통합 최적화.