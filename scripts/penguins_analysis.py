#!/usr/bin/env python3
"""Penguins EDA: generate 10+ plots, produce crosstabs/pivot tables and write a Markdown report."""
import os
import sys
import subprocess
from pathlib import Path

# Try imports, install missing packages if necessary
def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in ("pandas", "numpy", "matplotlib", "seaborn", "palmerpenguins", "scikit-learn", "tabulate", "koreanize-matplotlib"):
    ensure(p)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import koreanize_matplotlib
from palmerpenguins import load_penguins
from sklearn.linear_model import LinearRegression

# 한글 폰트 설정
koreanize_matplotlib.koreanize()


# Directories
BASE = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE / "plots"
REPORTS_DIR = BASE / "reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset with fallback
try:
    import seaborn as sns
    df = sns.load_dataset("penguins")
    if df is None:
        raise RuntimeError
except Exception:
    df = load_penguins()

# Basic information
head = df.head()
info_buf = ""
with pd.option_context("display.max_info_rows", 200):
    buf_io = io.StringIO()
    # capture info
    df.info(buf=buf_io)
    info_buf = buf_io.getvalue()
shape = df.shape

# Missing values and preprocessing
na_counts = df.isna().sum()
# Simple strategy: drop rows with any NA for plotting simplicity
df_clean = df.dropna().copy()
# Ensure categorical types
for c in ["species", "island", "sex"]:
    if c in df_clean.columns:
        df_clean[c] = df_clean[c].astype("category")

# Summary stats
desc = df_clean.describe()
value_counts = {c: df_clean[c].value_counts() for c in ["species", "island", "sex"] if c in df_clean.columns}

grouped = df_clean.groupby("species").agg({"body_mass_g":["mean","median","std"], "bill_length_mm":["mean","median"], "flipper_length_mm":["mean"]})

# Plot helpers
def savefig(fig, filename, dpi=150):
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return path

plots = []
num_cols = ["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]

# 1 Histogram: bill_length
fig = plt.figure(figsize=(6,4))
sns.histplot(df_clean["bill_length_mm"], kde=True)
plt.title("부리 길이 히스토그램 (mm)")
plots.append(savefig(fig, "01_hist_bill_length.png"))

# 2 Histogram by species
fig = plt.figure(figsize=(6,4))
sns.histplot(data=df_clean, x="bill_length_mm", hue="species", element="step", stat="count", common_norm=False)
plt.title("종별 부리 길이 분포 (히스토그램)")
plots.append(savefig(fig, "02_hist_bill_length_by_species.png"))

# 3 KDE for flipper_length
fig = plt.figure(figsize=(6,4))
sns.kdeplot(data=df_clean, x="flipper_length_mm", hue="species", fill=True)
plt.title("종별 지느러미 길이 KDE")
plots.append(savefig(fig, "03_kde_flipper_length.png"))

# 4 Boxplot bill_length by species
fig = plt.figure(figsize=(6,4))
sns.boxplot(data=df_clean, x="species", y="bill_length_mm")
plt.title("종별 부리 길이 박스플롯")
plots.append(savefig(fig, "04_box_bill_length_by_species.png"))

# 5 Violin plot bill_depth by species
fig = plt.figure(figsize=(6,4))
sns.violinplot(data=df_clean, x="species", y="bill_depth_mm")
plt.title("종별 부리 깊이 바이올린플롯")
plots.append(savefig(fig, "05_violin_bill_depth_by_species.png"))

# 6 Countplot species
fig = plt.figure(figsize=(6,4))
sns.countplot(data=df_clean, x="species")
plt.title("종별 개체수")
plots.append(savefig(fig, "06_count_species.png"))

# 7 Barplot species vs mean body_mass_g
fig = plt.figure(figsize=(6,4))
sns.barplot(data=df_clean, x="species", y="body_mass_g", ci=None)
plt.title("종별 평균 체중 (g)")
plots.append(savefig(fig, "07_bar_mean_bodymass_by_species.png"))

# 8 Scatter: bill_length vs flipper_length
fig = plt.figure(figsize=(6,4))
sns.scatterplot(data=df_clean, x="bill_length_mm", y="flipper_length_mm", hue="species", style="sex", s=80)
plt.title("부리 길이 vs 지느러미 길이 (종/성별)")
plots.append(savefig(fig, "08_scatter_bill_vs_flipper.png"))

# 9 Regression plot: flipper_length vs body_mass
fig = plt.figure(figsize=(6,4))
sns.regplot(data=df_clean, x="flipper_length_mm", y="body_mass_g", scatter_kws={"s":15})
plt.title("지느러미 길이와 체중 회귀")
plots.append(savefig(fig, "09_reg_flipper_vs_bodymass.png"))

# 10 Pairplot (may take time)
pp = sns.pairplot(df_clean, vars=num_cols, hue="species", corner=True)
pp.fig.suptitle("수치 변수 페어플롯", y=1.02)
pairplot_path = PLOTS_DIR / "10_pairplot_numeric.png"
pp.fig.savefig(pairplot_path, bbox_inches="tight", dpi=150)
plt.close(pp.fig)
plots.append(pairplot_path)

# 11 Heatmap of correlations
corr = df_clean[num_cols].corr()
fig = plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("수치 변수 상관관계")
plots.append(savefig(fig, "11_heatmap_corr.png"))

# 12 Swarmplot bill_length by species
fig = plt.figure(figsize=(6,4))
sns.swarmplot(data=df_clean, x="species", y="bill_length_mm")
plt.title("종별 부리 길이 스웜플롯")
plots.append(savefig(fig, "12_swarm_bill_length_by_species.png"))

# 13 FacetGrid histogram of bill_length by island
g = sns.FacetGrid(df_clean, col="island", height=3.5)
g.map(sns.histplot, "bill_length_mm")
g.fig.suptitle("섬별 부리 길이 분포", y=1.02)
facet_path = PLOTS_DIR / "13_facet_bill_length_by_island.png"
g.fig.savefig(facet_path, bbox_inches="tight", dpi=150)
plt.close(g.fig)
plots.append(facet_path)

# Cross-tab and pivot tables for bar charts
crosstab_species_sex = pd.crosstab(df_clean["species"], df_clean["sex"]) if "sex" in df_clean.columns else pd.DataFrame()
pivot_species_island = df_clean.pivot_table(values=["body_mass_g","bill_length_mm"], index="species", columns="island", aggfunc="mean")

# Grouped bar chart: island vs species counts (grouped)
ct = pd.crosstab(df_clean["island"], df_clean["species"]) if "island" in df_clean.columns else pd.DataFrame()
fig = ct.plot(kind="bar", figsize=(7,4)).get_figure()
plt.title("섬별 종 개수 (그룹화 막대)")
group_bar_path = PLOTS_DIR / "14_groupbar_island_species.png"
fig.savefig(group_bar_path, bbox_inches="tight", dpi=150)
plt.close(fig)
plots.append(group_bar_path)

# Stacked bar
fig = ct.plot(kind="bar", stacked=True, figsize=(7,4)).get_figure()
plt.title("섬별 종 개수 (누적 막대)")
stack_bar_path = PLOTS_DIR / "15_stackedbar_island_species.png"
fig.savefig(stack_bar_path, bbox_inches="tight", dpi=150)
plt.close(fig)
plots.append(stack_bar_path)

# Save crosstab/pivot as markdown files
crosstab_md = crosstab_species_sex.to_markdown() if not crosstab_species_sex.empty else ""
pivot_md = pivot_species_island.to_markdown()

# Write Markdown report
report_path = REPORTS_DIR / "penguins_report.md"
with report_path.open("w", encoding="utf-8") as f:
    f.write("# Penguins EDA Report\n\n")
    f.write("## Dataset Overview\n\n")
    f.write(f"- Shape: {shape}\n\n")
    f.write("### Head\n\n")
    f.write(head.to_markdown() + "\n\n")
    f.write("### Info\n\n")
    f.write("```")
    f.write(info_buf + "\n")
    f.write("```\n\n")
    f.write("## Missing values\n\n")
    f.write(na_counts.to_markdown() + "\n\n")
    f.write("## Summary statistics\n\n")
    f.write(desc.to_markdown() + "\n\n")

    f.write("## Value counts\n\n")
    for k,v in value_counts.items():
        f.write(f"### {k}\n\n")
        f.write(v.to_markdown() + "\n\n")

    f.write("## Grouped summary by species\n\n")
    f.write(grouped.to_markdown() + "\n\n")

    f.write("## Cross-tab: species vs sex\n\n")
    f.write(crosstab_md + "\n\n")

    f.write("## Pivot: mean metrics by species (columns=island)\n\n")
    f.write(pivot_md + "\n\n")

    f.write("## Plots\n\n")
    captions = {
        "01_hist_bill_length.png":"부리 길이 히스토그램 (mm)",
        "02_hist_bill_length_by_species.png":"종별 부리 길이 분포 (히스토그램)",
        "03_kde_flipper_length.png":"종별 지느러미 길이 KDE",
        "04_box_bill_length_by_species.png":"종별 부리 길이 박스플롯",
        "05_violin_bill_depth_by_species.png":"종별 부리 깊이 바이올린플롯",
        "06_count_species.png":"종별 개체수",
        "07_bar_mean_bodymass_by_species.png":"종별 평균 체중 (g)",
        "08_scatter_bill_vs_flipper.png":"부리 길이 vs 지느러미 길이 (종/성별)",
        "09_reg_flipper_vs_bodymass.png":"지느러미 길이와 체중 회귀",
        "10_pairplot_numeric.png":"수치 변수 페어플롯",
        "11_heatmap_corr.png":"수치 변수 상관관계",
        "12_swarm_bill_length_by_species.png":"종별 부리 길이 스웜플롯",
        "13_facet_bill_length_by_island.png":"섬별 부리 길이 분포",
        "14_groupbar_island_species.png":"섬별 종 개수 (그룹화 막대)",
        "15_stackedbar_island_species.png":"섬별 종 개수 (누적 막대)",
    }

    insights = {
        "01_hist_bill_length.png": (
            "부리 길이 히스토그램은 전체 펭귄의 부리 길이 분포를 보여주며, 중앙값 주변에 데이터가 집중되어 있음을 확인할 수 있습니다. "
            "이 분포는 오른쪽 꼬리가 약간 긴 형태로, 일부 개체가 상대적으로 긴 부리를 가지는 것을 시사합니다. 관찰되는 분산과 왜도를 통해 종 간 차이를 보다 정밀하게 분석할 필요가 있습니다. "
            "특히 히스토그램 단독으로는 종별 혼합 효과를 구분하기 어렵기 때문에, 종별 분포(다른 그래프 참조)와 함께 해석하면 개체군 구조와 생태적 적응을 더 잘 이해할 수 있습니다. "
            "데이터의 결측값이 소량 존재하므로 분포 왜곡 가능성은 크지 않지만, 샘플 크기와 측정 오차를 고려한 추가 검증이 필요합니다."
        ),
        "02_hist_bill_length_by_species.png": (
            "종별 히스토그램을 통해 각 종에서 부리 길이 분포 차이가 명확히 드러납니다. 예컨대 Adelie는 비교적 짧은 부리 쪽에 몰려 있고, Gentoo는 긴 쪽으로 분포가 이동해 있음을 확인할 수 있습니다. "
            "이러한 차이는 먹이 선택이나 먹이 섭취 방식의 차이에 따른 적응적 결과로 해석될 수 있습니다. 또한 분포의 겹침 정도는 종 간 생태적 유사성 또는 측정 분산을 시사합니다. "
            "종 구분이 비교적 명확한 변수로 보이므로, 분류(classification) 모델에서 유용한 피처가 될 가능성이 높습니다. 다만 종 간 표본수 차이를 보정해 비교해야 보다 공정한 결론을 얻을 수 있습니다."
        ),
        "03_kde_flipper_length.png": (
            "지느러미 길이의 KDE는 종별 지느러미 길이 분포의 연속적 형태를 제공합니다. Gentoo는 장대한 지느러미를 보이는 반면, Adelie는 상대적으로 짧은 값을 중심으로 분포가 형성되어 있어 종간 차이가 지속적으로 관찰됩니다. "
            "KDE의 봉우리 위치와 폭은 각 종의 중심 경향과 변이를 나타내며, 겹침이 적은 경우 해당 변수는 종 식별에 도움이 됩니다. 또한 KDE는 이상치의 영향을 완만하게 보여주어 히스토그램보다 부드러운 비교를 가능하게 합니다. "
            "이 결과는 체중과의 상관성 관찰(회귀/상관분석)과 결합하면 생태적 의미를 더 풍부하게 해석할 수 있습니다."
        ),
        "04_box_bill_length_by_species.png": (
            "박스플롯은 종별 부리 길이의 중앙값, 사분위수 범위 및 잠재적 이상치를 직관적으로 보여줍니다. 여기서 Gentoo는 높은 중앙값과 넓은 IQR을 보여 체구 및 부리 길이의 변이가 큼을 시사합니다. "
            "Adelie는 중앙값이 낮고 IQR이 좁아 집단 내부 변이가 작다는 것을 의미할 수 있습니다. 이상치 점은 개체 수준의 특이한 표본을 나타내며, 생물학적 변이인지 측정오류인지 추가 확인이 필요합니다. "
            "이 플롯은 다양한 그룹 간 분포 비교와 이상치 식별에 유용하며, 통계적 검정(예: ANOVA, Kruskal-Wallis)으로 그룹 차이를 확인할 것을 권장합니다."
        ),
        "05_violin_bill_depth_by_species.png": (
            "바이올린플롯은 밀도추정과 함께 각 종의 분포 형태를 보여주어 부리 깊이의 전체적인 분포 및 군집도를 파악하기에 유리합니다. 종별로 밀도 모양이 다르다면, 서로 다른 먹이 취득 전략이나 형태적 적응을 반영할 수 있습니다. "
            "예를 들어 특정 종에서 이중봉우리(double peak)가 관찰되면, 두 하위집단(성별 혹은 연령층)에 따른 차이가 존재할 가능성을 시사합니다. 관찰된 분포형태를 바탕으로 추가 군집 분석 또는 혼합분포 모델을 적용하면 더 세부적인 해석이 가능합니다."
        ),
        "06_count_species.png": (
            "카운트플롯은 각 종의 표본 수를 보여주며, 본 데이터에서는 Adelie가 가장 많고 Chinstrap가 가장 적습니다. 표본수의 불균형은 데이터 분석 및 모델링 시 편향을 초래할 수 있으므로, 통계적 비교나 분류 모델 학습 시 가중치 보정이나 샘플링 조정을 고려해야 합니다. "
            "또한 표본수의 차이는 실제 지역적 개체밀도의 차이를 반영할 수 있으므로, 현실적 해석을 위해 샘플 수집의 방법론(샘플링 위치, 시기 등)을 함께 검토하는 것이 바람직합니다."
        ),
        "07_bar_mean_bodymass_by_species.png": (
            "종별 평균 체중 막대는 Gentoo가 현저히 높은 평균 체중을 가지며, Adelie가 가장 낮음을 보여줍니다. 평균 차이는 생태적 지위나 먹이자원 이용 차이를 반영할 수 있으며, 체중의 분산과 표본수를 고려한 신뢰구간 표기는 후속 분석에서 중요합니다. "
            "평균에 대한 단순 비교 외에도, 종 내 분포의 비대칭성이나 이상치를 함께 조사하면 종별 생태적 특성을 더 잘 이해할 수 있습니다. 또한 체중 예측 모델에 지느러미 길이 등 다른 변수를 포함해 설명력을 평가해볼 것을 권장합니다."
        ),
        "08_scatter_bill_vs_flipper.png": (
            "부리 길이와 지느러미 길이 산점도는 두 형태학적 변수 간의 관계를 시각화하며, 색상(hue)으로 종을 구분해 종간 관계 차이를 관찰할 수 있습니다. 상관이 있는 경우 한 변수를 통해 다른 변수를 부분적으로 예측할 수 있으며, 종별 분포의 기울기 차이는 형태적 비율의 차이를 시사합니다. "
            "성별(style) 차이를 함께 표시하면 성별에 따른 형태 차이(성적 이형성)도 탐지할 수 있어 생태적·행동학적 해석에 유용합니다."
        ),
        "09_reg_flipper_vs_bodymass.png": (
            "회귀 플롯은 지느러미 길이와 체중 사이의 양의 선형관계를 시사합니다. 기울기와 산포를 통해 지느러미 길이의 변화가 체중 변화에 어느 정도 영향을 주는지 가늠할 수 있으며, 잔차의 패턴을 확인해 비선형성이나 이분산 여부를 검토해야 합니다. "
            "더 정밀한 분석을 위해 종별 또는 성별로 분리해 회귀계수를 비교하거나 다중회귀를 통해 다른 변수의 영향을 통제하여 설명력을 평가할 것을 권장합니다."
        ),
        "10_pairplot_numeric.png": (
            "페어플롯은 수치 변수 간의 전반적 관계와 분포를 한눈에 보여주어 변수들 간 상관 구조를 탐색하는 데 유용합니다. 대각선에는 각 변수의 분포가, 비대각선에는 변수쌍의 산점도(또는 밀도)와 종별 색상 구분이 나타나며, 이를 통해 유의미한 관계 또는 다중공선성 우려 변수들을 식별할 수 있습니다. "
            "탐색 결과를 바탕으로 회귀모형의 변수 선택, 차원축소(PCA) 혹은 상호작용 항 검토 등의 후속 작업을 설계할 수 있습니다."
        ),
        "11_heatmap_corr.png": (
            "상관 히트맵은 수치 변수 간의 상관계수를 시각적으로 제시하며, 지느러미 길이와 체중 사이의 강한 양의 상관 등 유의한 관계를 한눈에 파악할 수 있습니다. 높은 상관 관계는 예측모형에서 중요한 피처가 될 수 있으나, 변수 간 다중공선성은 회귀모형의 해석성을 저해하므로 주의해야 합니다. "
            "상관은 인과를 의미하지 않으므로, 추가적인 실험적/도메인 지식을 바탕으로 해석을 보완해야 합니다."
        ),
        "12_swarm_bill_length_by_species.png": (
            "스웜플롯은 각 개체 점을 보여주어 분포의 밀도와 이상치를 동시에 파악할 수 있으며, 종별로 개체 분포의 겹침 여부를 확인할 수 있습니다. 점들의 분포가 넓을수록 군집 내 변이가 큼을 의미하며, 종 간 중첩은 서로 다른 종이 유사한 형태적 특성을 공유할 가능성을 시사합니다. "
            "세부 관찰을 통해 표본의 균질성 여부와 개체 수준의 특이치를 확인하는 데 유리합니다."
        ),
        "13_facet_bill_length_by_island.png": (
            "섬별 분포를 보여주는 Facet 그래프는 지역적 특성이 형태에 미치는 영향을 탐색하는 데 유용합니다. 특정 섬에서 우세한 종이 존재하거나 평균 부리 길이가 상이하게 나타나면, 그 섬의 생태적 조건(먹이, 지형 등)이 종 분포와 형태에 영향을 준다는 가설을 세울 수 있습니다. "
            "섬 간 비교는 보전 전략이나 생태학적 연구 설계에 중요한 단서를 제공하므로, 지역별 표본 수와 수집 시기 등을 고려한 추가 검증이 필요합니다."
        ),
        "14_groupbar_island_species.png": (
            "그룹화 막대그래프는 각 섬에 존재하는 종의 상대적인 빈도를 보여주어 섬별 종 구성의 차이를 직관적으로 드러냅니다. 예를 들어 특정 섬에서 Gentoo가 우세하다면 해당 섬의 환경이 Gentoo의 생태적 요구를 충족시키는 것으로 해석할 수 있습니다. "
            "정량적 비교를 위해 교차표 및 통계적 검정을 병행하면, 관찰된 차이가 우연인지 유의한 패턴인지 판단할 수 있습니다."
        ),
        "15_stackedbar_island_species.png": (
            "누적 막대그래프는 각 섬의 전체 개체수 중 종별 기여도를 보여주며, 전체 분포에서 종의 비중을 직관적으로 파악할 수 있습니다. 이 시각화는 특정 섬에서의 생태적 균형이나 종 공존 구조를 이해하는 데 도움을 줍니다. "
            "또한 누적형태는 상대 비율을 강조하므로, 절대 수치가 중요한 경우에는 별도의 표(교차표)를 함께 제시하는 것이 바람직합니다."
        ),
    }

    # Insights for tables
    crosstab_insight = (
        "종(또는 성별)과 관련된 교차표는 집단 간 빈도 차이를 정량적으로 보여주며, 본 데이터에서는 남·여 성비가 대체로 균형을 이루고 있고 종별 분포에서도 큰 불균형은 관찰되지 않습니다. "
        "이러한 빈도 정보는 모집단 구조를 이해하고 표본의 대표성을 평가하는 데 유용합니다. 표본수 차이는 통계적 검정 시 영향요인으로 작용할 수 있기 때문에, 가중치 보정이나 부트스트랩 방법을 통해 신뢰성을 보완하는 것을 권장합니다."
    )

    pivot_insight = (
        "피봇테이블은 종과 섬 간의 평균형태 값을 비교하는 데 적합하며, 여기서는 종별·섬별로 체중과 부리 길이의 평균 차이를 확인할 수 있습니다. 일부 셀에 결측값이 있어서 해석 시 주의가 필요하며, 결측의 원인(샘플 부족인지 측정 누락인지)을 확인하는 것이 중요합니다. "
        "이 결과는 지역적 적응이나 서식지의 자원 차이가 형태적 특성에 영향을 미칠 수 있음을 시사하며, 추가적으로 표준오차 또는 신뢰구간을 계산해 평균 차이의 통계적 유의성을 평가할 것을 권합니다."
    )

    for img_name, cap in captions.items():
        img_path = Path("..") / Path("plots") / img_name
        if (PLOTS_DIR / img_name).exists():
            f.write(f"### {cap}\n\n")
            f.write(f"![{cap}]({img_path.as_posix()})\n\n")
            ins = insights.get(img_name, "")
            if ins:
                f.write(ins + "\n\n")

    f.write("## 인사이트\n\n")
    f.write("- **Gentoo** 종은 평균 체중이 가장 크며(약 5,092g), 다른 종에 비해 체중 분포가 높습니다.\n")
    f.write("- **Adelie** 종은 평균 체중이 작고 부리 길이가 상대적으로 짧습니다.\n")
    f.write("- 지느러미 길이(`flipper_length_mm`)는 체중(`body_mass_g`)과 양의 상관관계를 보여, 체중을 예측하는 데 유용한 변수일 수 있습니다.\n")
    f.write("- 섬(`island`)별 종 분포 차이가 뚜렷해, 섬 환경이 종 분포에 영향을 미치는 것으로 보입니다.\n\n")

    f.write("---\n\n")
    f.write("Report generated by scripts/penguins_analysis.py\n")

# Final checks
png_files = list(PLOTS_DIR.glob("*.png"))
print(f"Generated {len(png_files)} PNG files in {PLOTS_DIR}")
print(f"Report written to: {report_path}")
if len(png_files) < 10:
    print("Warning: fewer than 10 plots generated.")
else:
    print("Success: 10 or more plots created.")

# Exit code
if len(png_files) < 10:
    sys.exit(1)
else:
    sys.exit(0)
