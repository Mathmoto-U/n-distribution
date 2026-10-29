"""
標本平均が正規分布に収束するシミュレーション (Streamlit版)
====================================================================
実行方法:
    pip install streamlit numpy plotly
    streamlit run app.py
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. ページ設定 & カスタムCSS（Web版のUI・10×10グリッドを忠実に再現）
# ============================================================================
st.set_page_config(
    page_title="標本平均の正規分布収束シミュレーター",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    /* サイドバーを非表示にし、全体の余白を調整 */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .block-container {
       <style>
    /* 1. Streamlit標準の上部ヘッダーとサイドバーを非表示にする（上部の見切れ防止） */
    header[data-testid="stHeader"],
    [data-testid="stDecoration"],
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* 2. ページ全体の縦・横スクロールを確実に有効化する */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main {
        overflow-y: auto !important;
        overflow-x: auto !important;
        height: auto !important;
        min-height: 100vh !important;
    }

    /* 3. 上下の余白調整（上が見切れないよう padding-top を確保） */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
    }

    /* 4. PCの横画面（左：標本、右：ヒストグラム）を強制維持 */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: flex-start !important;
    }
    [data-testid="column"] {
        min-width: 0 !important;
    }
    }

    /* PCのような横画面（左：標本、右：ヒストグラム）を強制維持 */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: flex-start !important;
    }
    [data-testid="column"] {
        min-width: 0 !important;
    }

    /* ヘッダーバナー */
    .sim-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 14px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .sim-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.3;
    }
    .sim-subtitle {
        font-size: 0.80rem;
        color: #64748b;
        margin: 4px 0 0 0;
    }

    /* 左カラム：最新結果バナー（「あたりの本数」を中央揃え） */
    .result-banner {
        background: linear-gradient(90deg, #ecfdf5 0%, #f0fdfa 55%, #f8fafc 100%);
        border: 1px solid #a7f3d0;
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 14px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .result-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 4px;
    }
    .result-val-emerald {
        font-size: 2.1rem;
        font-weight: 800;
        color: #059669;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 1.1;
    }

    /* 10x10 サンプリング・グリッド */
    .sampling-grid-wrapper {
        background: #f1f5f9;
        padding: 10px;
        border-radius: 14px;
        border: 1px solid #cbd5e1;
        max-width: 390px;
        margin: 0 auto 6px auto;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.03);
    }
    .sampling-grid {
        display: grid;
        grid-template-columns: repeat(10, 1fr);
        gap: 5px;
        aspect-ratio: 1 / 1;
    }
    .cell-circle {
        background: #10b981;
        color: #ffffff;
        font-weight: 800;
        font-size: 0.92rem;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(5, 150, 105, 0.35);
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        user-select: none;
        aspect-ratio: 1 / 1;
    }
    .cell-cross {
        background: rgba(226, 232, 240, 0.85);
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.80rem;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #e2e8f0;
        user-select: none;
        aspect-ratio: 1 / 1;
    }

    /* 右カラム：累計試行回数カード */
    .stat-cards-grid {
        margin-bottom: 12px;
    }
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 12px 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .stat-card-top {
        font-size: 0.85rem;
        font-weight: 700;
        color: #475569;
    }
    .stat-card-main {
        font-size: 1.65rem;
        font-weight: 800;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 1.1;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# 2. 定数・セッション状態の初期化
# ============================================================================
N_CELLS = 100  # 10x10 マス = 1標本あたり100回のベルヌーイ試行
SLEEP_INTERVAL = 0.15  # 連続実行時の更新間隔（秒）
STEP_BATCH_SIZE = 1    # 連続実行時の1ステップあたり試行回数


def init_session_state() -> None:
    """Streamlitのセッション状態を初期化する"""
    if "prob" not in st.session_state:
        st.session_state.prob = 0.60  # 〇が出る確率 (初期値 60%)
    if "bin_width" not in st.session_state:
        st.session_state.bin_width = 3  # 推奨階級幅: 3 (±2σを7等分)
    if "exact_counts" not in st.session_state:
        st.session_state.exact_counts = np.zeros(N_CELLS + 1, dtype=int)
    if "total_trials" not in st.session_state:
        st.session_state.total_trials = 0
    if "sum_x" not in st.session_state:
        st.session_state.sum_x = 0
    if "sum_x2" not in st.session_state:
        st.session_state.sum_x2 = 0
    if "latest_count" not in st.session_state:
        st.session_state.latest_count = None
    if "current_grid" not in st.session_state:
        st.session_state.current_grid = np.random.random(N_CELLS) < 0.60
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "sort_grid" not in st.session_state:
        st.session_state.sort_grid = False
    if "zoom_around_mean" not in st.session_state:
        st.session_state.zoom_around_mean = True
    if "show_normal_curve" not in st.session_state:
        st.session_state.show_normal_curve = True
    if "show_sigma_bands" not in st.session_state:
        st.session_state.show_sigma_bands = True


init_session_state()


# ============================================================================
# 3. サンプリング & 階級（ビン）構築ロジック
# ============================================================================
def reset_simulation() -> None:
    """累積データをリセットして初期状態に戻す"""
    st.session_state.is_running = False
    st.session_state.exact_counts = np.zeros(N_CELLS + 1, dtype=int)
    st.session_state.total_trials = 0
    st.session_state.sum_x = 0
    st.session_state.sum_x2 = 0
    st.session_state.latest_count = None
    st.session_state.current_grid = np.random.random(N_CELLS) < st.session_state.prob


def run_trials(num_samples: int = 1) -> None:
    """
    指定回数のサンプリングを実行し、10×10グリッドおよび度数分布を更新する。
    num_samples が大きい場合は NumPy の二項分布乱数で高速に一括処理する。
    """
    p = st.session_state.prob

    if num_samples <= 0:
        return

    # 最後の1回分は10x10マスの個別セルを生成して画面左に表示する
    last_grid = np.random.random(N_CELLS) < p
    last_count = int(np.sum(last_grid))

    if num_samples == 1:
        st.session_state.exact_counts[last_count] += 1
        st.session_state.total_trials += 1
        st.session_state.sum_x += last_count
        st.session_state.sum_x2 += last_count * last_count
    else:
        # (num_samples - 1) 回分はベクトル化で一括生成
        batch_counts = np.random.binomial(N_CELLS, p, size=num_samples - 1)
        all_counts = np.append(batch_counts, last_count)

        bincount = np.bincount(all_counts, minlength=N_CELLS + 1)
        st.session_state.exact_counts += bincount
        st.session_state.total_trials += num_samples
        st.session_state.sum_x += int(np.sum(all_counts))
        st.session_state.sum_x2 += int(np.sum(all_counts.astype(np.int64) ** 2))

    st.session_state.current_grid = last_grid
    st.session_state.latest_count = last_count


def build_bins(bin_width: int, prob: float, exact_counts: np.ndarray, latest_count: Optional[int]) -> List[Dict]:
    """
    平均 μ = 100p と ±2σ が美しく対称に分割されるように階級（ビン）を構築する。
    - 幅 3: 平均 60 が中央階級 (59〜61) の真ん中になり、±2σ (50〜70) を7等分
    - 幅 4: 50〜69 (±2σ) を 50〜53, 54〜57, 58〜61, 62〜65, 66〜69 で5等分
    - 幅 2: 50〜69 (±2σ) を10等分
    - 幅 1: 1個刻み (0〜100)
    - 幅 10: 0〜9, 10〜19, ..., 90〜100
    """
    bins = []
    mean_int = int(round(N_CELLS * prob))

    if bin_width == 1:
        for k in range(N_CELLS + 1):
            bins.append({
                "min_val": k,
                "max_val": k,
                "actual_width": 1,
                "center": float(k),
                "cont_start": k - 0.5,
                "cont_end": k + 0.5,
                "count": int(exact_counts[k]),
                "label": f"{k}",
                "range_label": f"{k}個",
                "contains_latest": (latest_count == k),
            })
        return bins

    if bin_width == 10:
        for b in range(10):
            min_v = b * 10
            max_v = N_CELLS if b == 9 else (min_v + 9)
            actual_w = max_v - min_v + 1
            c = int(np.sum(exact_counts[min_v : max_v + 1]))
            bins.append({
                "min_val": min_v,
                "max_val": max_v,
                "actual_width": actual_w,
                "center": (min_v + max_v) / 2.0,
                "cont_start": min_v - 0.5,
                "cont_end": max_v + 0.5,
                "count": c,
                "label": f"{min_v}〜{max_v}",
                "range_label": f"{min_v}〜{max_v}個",
                "contains_latest": (latest_count is not None and min_v <= latest_count <= max_v),
            })
        return bins

    # 幅 3, 4, 2 の対称オフセット計算
    if bin_width == 3:
        center_bin_start = mean_int - 1  # μ=60 -> 59〜61
    elif bin_width == 4:
        center_bin_start = mean_int - 2  # μ=60 -> 58〜61 (50〜69を5等分)
    else:
        center_bin_start = (mean_int // 2) * 2  # μ=60 -> 60〜61

    offset = center_bin_start % bin_width
    intervals: List[Tuple[int, int]] = []
    if offset > 0:
        intervals.append((0, offset - 1))
    for start in range(offset, N_CELLS + 1, bin_width):
        end = min(N_CELLS, start + bin_width - 1)
        intervals.append((start, end))

    for min_v, max_v in intervals:
        actual_w = max_v - min_v + 1
        c = int(np.sum(exact_counts[min_v : max_v + 1]))
        lbl = f"{min_v}" if min_v == max_v else f"{min_v}〜{max_v}"
        r_lbl = f"{min_v}個" if min_v == max_v else f"{min_v}〜{max_v}個"
        bins.append({
            "min_val": min_v,
            "max_val": max_v,
            "actual_width": actual_w,
            "center": (min_v + max_v) / 2.0,
            "cont_start": min_v - 0.5,
            "cont_end": max_v + 0.5,
            "count": c,
            "label": lbl,
            "range_label": r_lbl,
            "contains_latest": (latest_count is not None and min_v <= latest_count <= max_v),
        })

    return bins


# ============================================================================
# 4. HTML描画ヘルパー（10×10グリッド ＆ 統計カード）
# ============================================================================
def render_10x10_grid_html(grid_bools: np.ndarray, sort_grid: bool) -> str:
    """10×10の〇✕グリッドをHTMLとして生成する"""
    display_arr = np.sort(grid_bools)[::-1] if sort_grid else grid_bools
    cells_html = []
    for is_circle in display_arr:
        if is_circle:
            cells_html.append('<div class="cell-circle">〇</div>')
        else:
            cells_html.append('<div class="cell-cross">✕</div>')
    return f'<div class="sampling-grid-wrapper"><div class="sampling-grid">{"".join(cells_html)}</div></div>'


def render_left_result_banner_html(
    latest_count: Optional[int],
) -> str:
    """左カラム上部の「あたりの本数」中央揃えバナーHTML"""
    c_str = "--" if latest_count is None else f"{latest_count}"

    return f"""
    <div class="result-banner">
        <div class="result-label">あたりの本数</div>
        <div>
            <span class="result-val-emerald">{c_str}</span>
        </div>
    </div>
    """


def render_stat_cards_html() -> str:
    """右カラム上部の累計試行回数カードのHTMLを生成する"""
    N = st.session_state.total_trials

    return f"""
    <div class="stat-cards-grid">
        <div class="stat-card">
            <div class="stat-card-top">累計試行回数</div>
            <div class="stat-card-main" style="color:#0f172a;">
                {N:,} <span style="font-size:0.85rem; font-weight:600; color:#64748b;">回</span>
            </div>
        </div>
    </div>
    """


# ============================================================================
# 5. Plotlyによるヒストグラム＋正規分布曲線の描画
# ============================================================================
def create_histogram_figure(bins: List[Dict]) -> go.Figure:
    """ヒストグラムと正規分布曲線をPlotlyで描画する（上部凡例なし）"""
    N = st.session_state.total_trials
    p = st.session_state.prob
    w = st.session_state.bin_width

    mean = N_CELLS * p
    var = N_CELLS * p * (1.0 - p)
    std = math.sqrt(var)

    # X軸の表示範囲（0〜100全体）
    x_min, x_max = -0.5, 100.5
    visible_bins = [b for b in bins if b["cont_end"] > x_min and b["cont_start"] < x_max]

    max_bar_count = max([b["count"] for b in visible_bins], default=0)
    theo_peak = (N * w / (std * math.sqrt(2 * math.pi))) if N > 0 else 0.0
    raw_max_y = max(max_bar_count, theo_peak if st.session_state.show_normal_curve else 0.0, 5.0)
    y_max = math.ceil(raw_max_y * 1.18)

    fig = go.Figure()

    # ヒストグラムの棒グラフ
    centers = [b["center"] for b in visible_bins]
    counts = [b["count"] for b in visible_bins]
    widths = [b["actual_width"] * 0.88 for b in visible_bins]
    range_labels = [b["range_label"] for b in visible_bins]

    # 最新の試行が入った階級をハイライト
    bar_colors = [
        "#0d9488" if b["contains_latest"] else "rgba(16, 185, 129, 0.82)"
        for b in visible_bins
    ]
    border_colors = [
        "#065f46" if b["contains_latest"] else "#059669"
        for b in visible_bins
    ]
    border_widths = [2.2 if b["contains_latest"] else 1.0 for b in visible_bins]

    # 各階級の理論期待度数（ツールチップ用）
    expected_counts = []
    pct_list = []
    for b in visible_bins:
        c_val = b["center"]
        z = (c_val - mean) / std
        pdf = (1.0 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * z * z)
        exp_c = N * b["actual_width"] * pdf if N > 0 else 0.0
        expected_counts.append(exp_c)
        pct_list.append((b["count"] / N * 100.0) if N > 0 else 0.0)

    custom_data = np.column_stack([range_labels, pct_list, expected_counts])

    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            name="実測度数",
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=border_widths),
            ),
            customdata=custom_data,
            hovertemplate=(
                "<b>階級: %{customdata[0]}</b><br>"
                "実測度数: <b>%{y:,}回</b> (%{customdata[1]:.1f}%)<br>"
                "正規近似期待値: 約 %{customdata[2]:.1f}回"
                "<extra></extra>"
            ),
        )
    )

    # 正規分布の理論曲線オーバーレイ
    if st.session_state.show_normal_curve:
        x_curve = np.linspace(0, 100, 400)
        z_curve = (x_curve - mean) / std
        pdf_curve = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z_curve**2)

        if N > 0:
            y_curve = N * w * pdf_curve
            line_dash = "solid"
        else:
            # 未実行時はガイドとして高さを合わせて破線表示
            peak_pdf = 1.0 / (std * math.sqrt(2.0 * math.pi))
            y_curve = (y_max * 0.75) * (pdf_curve / peak_pdf)
            line_dash = "dash"

        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=y_curve,
                mode="lines",
                line=dict(color="#f59e0b", width=3.2, dash=line_dash),
                hoverinfo="skip",
            )
        )

    xaxis_config = dict(
        range=[0, 100],
        dtick=10,
        gridcolor="#f1f5f9",
        zeroline=False,
        linecolor="#cbd5e1",
        tickfont=dict(size=11, color="#475569"),
    )

    fig.update_layout(
        height=420,
        margin=dict(l=44, r=20, t=16, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=xaxis_config,
        yaxis=dict(
            range=[0, y_max],
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            tickfont=dict(size=11, color="#64748b"),
        ),
        showlegend=False,  # 上部の凡例（謎の図）を非表示
        bargap=0.08,
        hovermode="closest",
    )

    return fig


# ============================================================================
# 6. ヘッダー & トップコントロールバー
# ============================================================================
st.markdown(
    """
    <div class="sim-header">
        <div class="sim-title">
            標本平均が正規分布に収束するシミュレーション
        </div>
        <p class="sim-subtitle">
            くじを100本引き、あたりの本数を数えます。これを何度も繰り返し、ヒストグラムに記録していきます。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 操作ボタンバー
ctrl_cols = st.columns([1.2, 1.6, 1.0, 1.0, 1.0, 1.0])

with ctrl_cols[0]:
    if st.button("1回実行", use_container_width=True):
        st.session_state.is_running = False
        run_trials(1)

with ctrl_cols[1]:
    run_btn_label = "一時停止" if st.session_state.is_running else "連続実行"
    run_btn_type = "secondary" if st.session_state.is_running else "primary"
    if st.button(run_btn_label, type=run_btn_type, use_container_width=True):
        st.session_state.is_running = not st.session_state.is_running
        if st.session_state.is_running:
            run_trials(STEP_BATCH_SIZE)

with ctrl_cols[2]:
    if st.button("+10回", use_container_width=True):
        st.session_state.is_running = False
        run_trials(10)

with ctrl_cols[3]:
    if st.button("+100回", use_container_width=True):
        st.session_state.is_running = False
        run_trials(100)

with ctrl_cols[4]:
    if st.button("+1000回", use_container_width=True):
        st.session_state.is_running = False
        run_trials(1000)

with ctrl_cols[5]:
    if st.button("リセット", use_container_width=True):
        reset_simulation()
        st.rerun()


# ============================================================================
# 7. メイン2カラムレイアウト（左: 10×10グリッド / 右: 統計 & ヒストグラム）
# ============================================================================
# 階級幅が 3, 2, 1 以外になっていた場合は 3 に戻す
if st.session_state.bin_width not in (3, 2, 1):
    st.session_state.bin_width = 3

all_bins = build_bins(
    st.session_state.bin_width,
    st.session_state.prob,
    st.session_state.exact_counts,
    st.session_state.latest_count,
)

col_left, col_right = st.columns([4.2, 6.8], gap="large")

# ---------------- 左カラム: 10×10 サンプリング・グリッド ----------------
with col_left:
    with st.container(border=True):
        # 「あたりの本数」中央揃えバナー
        st.markdown(
            render_left_result_banner_html(st.session_state.latest_count),
            unsafe_allow_html=True,
        )

        # 10x10 マス本体
        st.markdown(
            render_10x10_grid_html(st.session_state.current_grid, st.session_state.sort_grid),
            unsafe_allow_html=True,
        )


# ---------------- 右カラム: 統計メトリクス & ヒストグラム ----------------
with col_right:
    # 累計試行回数カード
    st.markdown(render_stat_cards_html(), unsafe_allow_html=True)

    with st.container(border=True):
        # 階級幅（Bin Width）セレクター（3, 2, 1 のみ） & 正規分布曲線トグル
        BIN_WIDTH_LABELS = {
            3: "幅 3",
            2: "幅 2",
            1: "幅 1",
        }

        toolbar_cols = st.columns([6, 4])
        with toolbar_cols[0]:
            selected_bw = st.radio(
                "階級幅の選択",
                options=[3, 2, 1],
                format_func=lambda x: BIN_WIDTH_LABELS[x],
                index=[3, 2, 1].index(st.session_state.bin_width),
                horizontal=True,
            )
            if selected_bw != st.session_state.bin_width:
                st.session_state.bin_width = selected_bw
                st.rerun()

        with toolbar_cols[1]:
            st.write("")  # ラジオボタンと高さを揃えるスペーサー
            st.session_state.show_normal_curve = st.toggle(
                "正規分布の理論曲線を表示",
                value=st.session_state.show_normal_curve,
            )

        # Plotly ヒストグラム描画
        fig = create_histogram_figure(all_bins)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================================
# 8. 連続実行モードの自動更新ループ
# ============================================================================
if st.session_state.is_running:
    time.sleep(SLEEP_INTERVAL)
    run_trials(STEP_BATCH_SIZE)
    st.rerun()
