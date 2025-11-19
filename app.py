# 标准库
import io
import datetime
import time
from PIL import Image

# 第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import streamlit as st
from scipy import sparse

# 配置与警告
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置（必须是第一个Streamlit命令）
st.set_page_config(
    page_title="CrOmLineSCNET",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 初始化scanpy设置
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# 初始化session state
if 'adata' not in st.session_state:
    st.session_state.adata = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = {
        'qc': False,
        'clustering': False,
        'trajectory': False,
        'grn': False
    }
# 存储分析历史（每次上传数据算一次分析）
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
# 侧边栏选中状态
if 'nav_selected' not in st.session_state:
    st.session_state.nav_selected = "项目介绍"
# 预计算的图像缓存
if 'hvg_result' not in st.session_state:
    st.session_state.hvg_result = None


def load_sample_data():
    """加载示例数据并记录历史"""
    try:
        adata = sc.datasets.pbmc3k()
        # 记录本次分析到历史
        record_analysis(adata, "示例数据(PBMC3K)")
        return adata
    except Exception as e:
        st.error(f"加载示例数据失败: {str(e)}。请检查网络连接或尝试使用本地数据。")
        return None


def load_uploaded_data(uploaded_file):
    """加载用户上传的数据并记录历史"""
    try:
        if uploaded_file.name.endswith('.h5ad'):
            adata = sc.read_h5ad(uploaded_file)
            # 记录本次分析到历史
            record_analysis(adata, uploaded_file.name)
            return adata
        else:
            st.error("请上传.h5ad格式的文件")
            return None
    except Exception as e:
        st.error(f"文件读取失败: {str(e)}。请检查：1. 文件是否为合法 .h5ad 格式；2. 文件未损坏；3. 权限是否足够。")
        return None


def record_analysis(adata, data_source):
    """记录单次分析的基本信息到历史，并清零当前步骤"""
    # 生成当前时间戳
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 整理数据基本信息
    analysis_info = {
        "timestamp": current_time,
        "data_source": data_source,  # 数据来源（文件名或示例数据标识）
        "n_cells": adata.n_obs,      # 细胞数量
        "n_genes": adata.n_vars,     # 基因数量
        "completed_steps": []        # 已完成的步骤（初始为空）
    }
    
    # 清零当前分析进度（重新上传数据后步骤重置）
    st.session_state.analysis_completed = {
        'qc': False,
        'clustering': False,
        'trajectory': False,
        'grn': False
    }
    st.session_state.current_step = 0
    
    # 将记录添加到历史列表
    st.session_state.analysis_history.append(analysis_info)
    # 限制历史记录数量（保留最近10条）
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history.pop(0)


def update_analysis_step(step_name):
    """更新当前分析的已完成步骤（用于历史记录）"""
    if st.session_state.adata is None:
        return
    # 找到最新的一条分析记录（当前正在进行的分析）
    if st.session_state.analysis_history:
        latest_record = st.session_state.analysis_history[-1]
        # 避免重复添加步骤
        if step_name not in latest_record["completed_steps"]:
            latest_record["completed_steps"].append(step_name)


def perform_qc_analysis(adata):
    """执行质量控制分析"""
    try:
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        # 标记线粒体基因、核糖体基因和血红蛋白基因
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
        adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
        
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
        )
        
        # 创建QC可视化图
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # 第一行：小提琴图
        sc.pl.violin(adata, 'n_genes_by_counts', ax=axes[0,0], jitter=0.4, show=False)
        axes[0,0].set_title('Genes per Cell', fontsize=14, fontweight='bold')
        
        sc.pl.violin(adata, 'total_counts', ax=axes[0,1], jitter=0.4, show=False)
        axes[0,1].set_title('Total Counts per Cell', fontsize=14, fontweight='bold')
        
        sc.pl.violin(adata, 'pct_counts_mt', ax=axes[0,2], jitter=0.4, show=False)
        axes[0,2].set_title('Mitochondrial Gene Percentage', fontsize=14, fontweight='bold')
        
        # 第二行：散点图和最高表达基因
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=axes[1,0], show=False)
        axes[1,0].set_title('Total Counts vs. Mito (%)', fontsize=14, fontweight='bold')
        
        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=axes[1,1], show=False)
        axes[1,1].set_title('Total Counts vs. Genes', fontsize=14, fontweight='bold')
        
        sc.pl.highest_expr_genes(adata, n_top=20, ax=axes[1,2], show=False)
        axes[1,2].set_title('Top Expressed Genes', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 显示QC统计信息
        st.write("**QC统计信息：**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均每个细胞的基因数", f"{adata.obs['n_genes_by_counts'].mean():.0f}")
        with col2:
            st.metric("平均每个细胞的总计数", f"{adata.obs['total_counts'].mean():.0f}")
        with col3:
            st.metric("线粒体基因平均占比", f"{adata.obs['pct_counts_mt'].mean():.2f}%")
        
        # 标记QC步骤完成
        update_analysis_step("质量控制")
        return fig
    
    except Exception as e:
        st.error(f"QC分析失败: {str(e)}")
        return None


def perform_clustering(adata):
    """执行聚类分析"""
    try:
        # 数据预处理
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata = adata[adata.obs.n_genes_by_counts < 2500, :]
        adata = adata[adata.obs.pct_counts_mt < 5, :]
        
        # 标准化
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # 高变基因筛选
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        
        # 去除批次效应
        sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
        sc.pp.scale(adata, max_value=10)
        
        # PCA降维
        sc.tl.pca(adata, svd_solver='arpack')
        
        # 邻居图构建
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        
        # UMAP降维
        sc.tl.umap(adata)
        
        # Louvain聚类
        sc.tl.louvain(adata, resolution=1.0, flavor="igraph")
        
        # 可视化聚类结果
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        sc.pl.umap(adata, color='louvain', ax=ax, show=False, legend_loc='right margin')
        ax.set_title('Louvain聚类结果', fontsize=16, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=14)
        ax.set_ylabel('UMAP 2', fontsize=14)
        
        plt.tight_layout()
        
        # 显示聚类统计信息
        n_clusters = len(adata.obs['louvain'].unique())
        st.write(f"**聚类概览：** 共识别 {n_clusters} 个细胞簇")
        
        # 显示每个簇的细胞数量
        cluster_counts = adata.obs['louvain'].value_counts().sort_index()
        st.write("**各簇的细胞数量：**")
        cluster_df = pd.DataFrame({
            '细胞簇': cluster_counts.index,
            '细胞数量': cluster_counts.values
        })
        st.dataframe(cluster_df, use_container_width=True)
        
        # 标记聚类步骤完成
        update_analysis_step("细胞聚类")
        return fig, adata
    
    except Exception as e:
        st.error(f"聚类分析失败: {str(e)}")
        return None, None


def perform_trajectory_analysis(adata):
    """执行轨迹推断分析"""
    try:
        # 扩散映射
        sc.tl.diffmap(adata)
        
        # 选择根细胞
        cluster_labels = adata.obs['louvain'].unique()
        cluster_labels = sorted(cluster_labels, key=lambda x: int(x))
        root_cluster = cluster_labels[0]
        root_cell = adata.obs_names[adata.obs['louvain'] == root_cluster][0]
        
        # 设置根细胞
        root_index = np.where(adata.obs_names == root_cell)[0][0]
        adata.uns['iroot'] = root_index
        
        # 计算DPT
        sc.tl.dpt(adata)
        
        # 可视化结果
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        sc.pl.umap(adata, color='dpt_pseudotime', ax=axes[0], show=False, cmap='plasma')
        axes[0].set_title('Pseudotime on UMAP', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('UMAP 1', fontsize=14)
        axes[0].set_ylabel('UMAP 2', fontsize=14)
        
        sc.pl.diffmap(adata, color='dpt_pseudotime', ax=axes[1], show=False, cmap='plasma')
        axes[1].set_title('Pseudotime on Diffusion Map', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Diffusion Component 1', fontsize=14)
        axes[1].set_ylabel('Diffusion Component 2', fontsize=14)
        
        plt.tight_layout()
        
        # 显示各簇的假时间分布
        st.write("**各簇的假时间统计：**")
        cluster_pseudotime = adata.obs.groupby('louvain')['dpt_pseudotime'].agg(['mean', 'std', 'min', 'max']).round(3)
        cluster_pseudotime.columns = ['平均值', '标准差', '最小值', '最大值']
        cluster_pseudotime = cluster_pseudotime.reset_index()
        cluster_pseudotime.columns = ['细胞簇'] + list(cluster_pseudotime.columns[1:])
        st.dataframe(cluster_pseudotime, use_container_width=True)
        
        # 标记轨迹步骤完成
        update_analysis_step("轨迹推断")
        return fig
        
    except Exception as e:
        st.error(f"轨迹分析失败: {str(e)}")
        return None


def perform_hvg_visualization(adata):
    """高变基因可视化"""
    try:
        temp = adata.copy()
        if sparse.issparse(temp.X):
            temp.X.data = np.nan_to_num(temp.X.data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            temp.X = np.nan_to_num(temp.X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 确保QC指标存在
        if 'n_genes_by_counts' not in temp.obs.columns or 'pct_counts_mt' not in temp.obs.columns:
            temp.var["mt"] = temp.var_names.str.upper().str.startswith("MT-")
            sc.pp.calculate_qc_metrics(temp, percent_top=None, log1p=False, inplace=True)
            if 'pct_counts_mt' not in temp.obs.columns:
                sc.pp.calculate_qc_metrics(temp, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        sc.pp.filter_cells(temp, min_genes=200)
        sc.pp.filter_genes(temp, min_cells=3)
        temp = temp[temp.obs.n_genes_by_counts < 2500, :]
        temp = temp[temp.obs.pct_counts_mt < 5, :]
        sc.pp.normalize_total(temp, target_sum=1e4)
        sc.pp.log1p(temp)
        try:
            sc.pp.highly_variable_genes(temp, min_mean=0.0125, max_mean=3, min_disp=0.5)
        except ValueError as err:
            if "Bin edges must be unique" in str(err):
                st.warning("默认的高变基因方法出现分箱问题，正在自动使用 Seurat v3 流程重试。")
                sc.pp.highly_variable_genes(
                    temp,
                    flavor='seurat_v3',
                    n_top_genes=2000,
                    span=0.3
                )
            else:
                raise
        
        # remove problematic NaN/inf entries for plotting
        for col in ['means', 'dispersions', 'dispersions_norm']:
            if col in temp.var.columns:
                temp.var[col] = temp.var[col].replace([np.inf, -np.inf], np.nan)
        valid_mask = temp.var[['means', 'dispersions', 'dispersions_norm']].notna().all(axis=1)
        if valid_mask.sum() == 0:
            st.warning("由于去除异常值后全部变为缺失，暂无法绘制高变基因散点图。")
            return None, None
        if not valid_mask.all():
            temp = temp[:, temp.var_names[valid_mask]].copy()
        
        hv_mask = temp.var['highly_variable'].values.astype(bool)
        if hv_mask.sum() == 0:
            st.warning("在当前阈值下没有检测到高变基因，请尝试放宽筛选条件。")
            return None, None
        
        means = temp.var['means'].to_numpy()
        disp_norm = temp.var['dispersions_norm'].to_numpy()
        safe_means = np.clip(means, 1e-5, None)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(
            safe_means[~hv_mask],
            disp_norm[~hv_mask],
            s=8,
            color="#d3d3d3",
            alpha=0.5,
            label="Non-HVG"
        )
        ax.scatter(
            safe_means[hv_mask],
            disp_norm[hv_mask],
            s=12,
            color="#d62728",
            alpha=0.9,
            label="HVG"
        )
        ax.set_xscale('log')
        ax.set_xlabel("Mean expression", fontsize=12)
        ax.set_ylabel("Normalized dispersion", fontsize=12)
        ax.set_title("Highly Variable Gene Selection", fontsize=16, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        
        stats = {
            "total_genes": temp.n_vars,
            "n_hvg": int(hv_mask.sum())
        }
        return fig, stats
    except Exception as e:
        st.error(f"高变基因分析失败：{str(e)}")
        return None, None


def perform_cell_annotation(adata, marker_file):
    """基因marker自动注释细胞类型"""
    try:
        if 'louvain' not in adata.obs.columns:
            st.warning("请先完成细胞聚类，再进行细胞类型注释。")
            return None, None
        
        try:
            markers_df = pd.read_csv(marker_file, sep=r'\s*,\s*', engine='python', header=0, usecols=[0, 1])
        except Exception:
            markers_df = pd.read_csv(marker_file, header=0)
        if markers_df.shape[1] < 2:
            st.error("Marker 文件至少需要包含 cell_type 和 markers 两列。")
            return None, None
        markers_df = markers_df.iloc[:, :2]
        markers_df.columns = ['cell_type', 'markers']
        
        cell_type_markers = {}
        for _, row in markers_df.iterrows():
            if pd.isna(row['cell_type']) or pd.isna(row['markers']):
                continue
            genes = [g.strip() for g in str(row['markers']).split(',') if g.strip()]
            if genes:
                cell_type_markers[str(row['cell_type']).strip()] = genes
        
        valid_markers = {
            ct: [gene for gene in genes if gene in adata.var_names]
            for ct, genes in cell_type_markers.items()
        }
        valid_markers = {ct: genes for ct, genes in valid_markers.items() if genes}
        if not valid_markers:
            st.error("在当前数据中未找到任何有效的 marker 基因。")
            return None, None
        
        cluster_means = {}
        gene_to_idx = {gene: idx for idx, gene in enumerate(adata.var_names)}
        for cluster in sorted(adata.obs['louvain'].unique(), key=lambda x: int(x)):
            mask = adata.obs['louvain'] == cluster
            subset = adata[mask, :]
            mean_expr = subset.X.mean(axis=0)
            if hasattr(mean_expr, "A1"):
                mean_expr = mean_expr.A1
            else:
                mean_expr = np.asarray(mean_expr).ravel()
            cluster_means[cluster] = mean_expr
        
        cluster_scores = {}
        for cluster, mean_expr in cluster_means.items():
            scores = {}
            for cell_type, genes in valid_markers.items():
                idxs = [gene_to_idx[g] for g in genes]
                scores[cell_type] = float(np.mean(mean_expr[idxs]))
            cluster_scores[cluster] = scores
        
        cell_type_annot = {cluster: max(scores, key=scores.get) for cluster, scores in cluster_scores.items()}
        adata.obs['cell_type'] = adata.obs['louvain'].map(cell_type_annot)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sc.pl.umap(adata, color='cell_type', ax=ax, legend_loc='right margin', show=False)
        ax.set_title('Cell Type Annotation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_rows = []
        for cluster, scores in cluster_scores.items():
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            summary_rows.append({
                "细胞簇": cluster,
                "预测类型": cell_type_annot[cluster],
                "最高得分": round(sorted_scores[0][1], 4),
                "次高类型": sorted_scores[1][0] if len(sorted_scores) > 1 else "-",
                "次高得分": round(sorted_scores[1][1], 4) if len(sorted_scores) > 1 else "-"
            })
        summary_df = pd.DataFrame(summary_rows)
        
        return fig, summary_df
    except Exception as e:
        st.error(f"细胞类型注释失败：{str(e)}")
        return None, None


def perform_differential_expression(adata, n_genes=20):
    """差异基因分析可视化"""
    try:
        if 'louvain' not in adata.obs.columns:
            st.warning("请先完成细胞聚类，再进行差异基因分析。")
            return None, None, None
        
        sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', n_genes=n_genes)
        
        sc.pl.rank_genes_groups_heatmap(
            adata,
            n_genes=n_genes,
            show_gene_labels=True,
            use_raw=False,
            show=False
        )
        fig_heatmap = plt.gcf()
        fig_heatmap.suptitle(f"Top {n_genes} Marker Genes per Cluster", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        sc.pl.rank_genes_groups(
            adata,
            n_genes=5,
            sharey=False,
            show=False
        )
        fig_summary = plt.gcf()
        fig_summary.suptitle("Top 5 Marker Genes per Cluster", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        result = adata.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        marker_genes = pd.DataFrame({group: result['names'][group][:5] for group in groups})
        
        return fig_heatmap, fig_summary, marker_genes
    except Exception as e:
        st.error(f"差异基因分析失败：{str(e)}")
        return None, None, None


def create_download_button(fig, filename, button_text="下载图像"):
    """创建图片下载按钮"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    st.download_button(
        label=button_text,
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png"
    )


def cache_hvg_plot(adata):
    """预计算并缓存高变基因图，供按钮快速展示"""
    if adata is None:
        st.session_state.hvg_result = None
        return
    fig, stats = perform_hvg_visualization(adata)
    if fig is None:
        st.session_state.hvg_result = None
        return
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    st.session_state.hvg_result = {
        "fig": fig,
        "image": buffer.getvalue(),
        "stats": stats,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def download_cached_image(cache_entry, filename, label):
    if cache_entry and cache_entry.get("image"):
        st.download_button(
            label=label,
            data=cache_entry["image"],
            file_name=filename,
            mime="image/png"
        )


def render_data_preview(adata, window=5):
    """展示靠近首个非零值的局部数据"""
    
    if adata is None or not hasattr(adata, "X") or adata.X is None:
        st.write("数据矩阵为空。")
        return
    X = adata.X
    n_obs, n_vars = adata.n_obs, adata.n_vars
    found_nonzero = False
    if sparse.issparse(X):
        rows, cols = X.nonzero()
        if len(rows) > 0:
            r0, c0 = int(rows[0]), int(cols[0])
            found_nonzero = True
    else:
        arr = np.asarray(X)
        nz = np.argwhere(arr != 0)
        if nz.size > 0:
            r0, c0 = map(int, nz[0])
            found_nonzero = True
    if not found_nonzero:
        r0 = c0 = 0
    half = max(window // 2, 1)
    def compute_range(center, length):
        start = max(center - half, 0)
        end = min(start + window, length)
        if end - start < window:
            start = max(end - window, 0)
            end = min(start + window, length)
        return start, end
    row_start, row_end = compute_range(r0, n_obs)
    col_start, col_end = compute_range(c0, n_vars)
    if sparse.issparse(X):
        submatrix = X[row_start:row_end, col_start:col_end].toarray()
    else:
        submatrix = np.asarray(X[row_start:row_end, col_start:col_end])
    df_preview = pd.DataFrame(
        np.round(submatrix, 3),
        index=adata.obs_names[row_start:row_end],
        columns=adata.var_names[col_start:col_end]
    )
    
    
    st.dataframe(df_preview, use_container_width=True)


# 自定义CSS样式
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* 顶端固定深色栏 */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 0.4rem 2rem;
        margin: 0;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        height: 60px;
        display: flex;
        align-items: center;
    }
    .main-header h1 {
        font-size: 1.5rem;
        margin: 0;
        padding: 0;
    }

    /* 主内容区防遮挡 */
    .appview-container {
        padding-top: 70px !important;
    }

    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding-top: 10px !important;
    }

    /* 主内容卡片 */
    .main-content {
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        line-height: 1.8;
    }

    /* 步骤按钮样式 */
    .step-button {
        width: 100%;
        margin: 0.5rem 0;
        padding: 0.75rem;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .step-button:disabled {
        background-color: #6c757d;
        cursor: not-allowed;
    }
    .step-button.completed {
        background-color: #28a745;
    }

    /* 历史记录卡片样式 */
    .history-card {
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        background-color: #f8f9fa;
    }
    
    /* 侧边栏按钮样式 - 简洁现代风格 */
    .sid-btn {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin: 4px;
        border: none;
        border-radius: 8px;
        background: white;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 13px;
        color: #333;
        text-decoration: none;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        flex-grow: 1;
        min-width: fit-content;
        justify-content: center;
    }
    .sid-btn:hover {
        background-color: #f0f2f6;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .sid-btn.active {
        background-color: #1e3c72;
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(30, 60, 114, 0.3);
        border-color: #1e3c72;
    }
    .sid-btn i {
        margin-right: 8px;
        width: 18px;
        text-align: center;
        font-size: 15px;
    }
    
    /* 隐藏radio组件 */
    [data-testid="stRadio"] {
        display: none;
    }

    /* 隐藏radio组件 */
    [data-testid="stRadio"] {
        display: none !important;
    }
    
    /* 自定义按钮样式 */
    .sidebar button {
        outline: none !important;
    }
    
    .sidebar button:focus {
        outline: 2px solid #1e3c72 !important;
        outline-offset: 2px !important;
    }
    
    /* 图标列样式 */
    .sidebar .stColumn:first-child {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin-right: -20px !important; /* 进一步减少右侧间距 */
    }
    
    .sidebar .stColumn:first-child .stMarkdown {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 按钮列样式 */
    .sidebar .stColumn:last-child {
        padding: 0 !important;
        margin-left: -20px !important; /* 进一步减少左侧间距 */
    }
    
    .sidebar .stColumn:last-child .stButton {
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# 主界面
def main():
    # 顶端固定标题栏（只保留CrOmLineSCNET）
    st.markdown("""
    <div class="main-header">
        <div>
            <h1>🧬 CrOmLineSCNET</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏导航
    with st.sidebar:
        st.markdown("## 导航菜单")
        
        # 定义导航项：图标类名、显示名称、对应值
        nav_items = [
            {"icon": "fa-info-circle", "label": "项目介绍", "value": "项目介绍"},
            {"icon": "fa-upload", "label": "上传数据", "value": "上传数据"},
            {"icon": "fa-flask", "label": "分析流程", "value": "分析流程"},
            {"icon": "fa-tools", "label": "分析工具", "value": "分析工具"},
            {"icon": "fa-history", "label": "历史记录", "value": "历史记录"}
        ]
        
        # 使用图标文本 + 按钮的组合布局
        for item in nav_items:
            is_active = st.session_state.nav_selected == item["value"]
            
            # 创建两列布局：左侧图标 + 右侧按钮（进一步缩小间距）
            col1, col2 = st.columns([1, 6])
            
            with col1:
                # 左侧FontAwesome风格图标（颜色固定不变）
                icon_html = {
                    "项目介绍": """
                    <div style="
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        height: 40px;
                        font-size: 20px;
                        font-weight: bold;
                        color: #333;
                    ">
                        ⓘ
                    </div>
                    """,
                    "上传数据": """
                    <div style="
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        height: 40px;
                        font-size: 20px;
                        font-weight: bold;
                        color: #333;
                    ">
                        ↑
                    </div>
                    """,
                    "分析流程": """
                    <div style="
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        height: 40px;
                        font-size: 20px;
                        font-weight: bold;
                        color: #333;
                    ">
                        ⚗
                    </div>
                    """,
                    "分析工具": """
                    <div style="
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        height: 40px;
                        font-size: 20px;
                        font-weight: bold;
                        color: #333;
                    ">
                        ⚒
                    </div>
                    """,
                    "历史记录": """
                    <div style="
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        height: 40px;
                        font-size: 20px;
                        font-weight: bold;
                        color: #333;
                    ">
                        🕐
                    </div>
                    """
                }
                
                st.markdown(icon_html[item['label']], unsafe_allow_html=True)
            
            with col2:
                # 右侧按钮
                if st.button(
                    item['label'],
                    key=f"nav_btn_{item['value']}",
                    help=f"切换到{item['label']}页面",
                    use_container_width=True
                ):
                    st.session_state.nav_selected = item["value"]
                    st.rerun()
                
                # 按钮样式
                st.markdown(f"""
                <style>
                .stButton > button[key="nav_btn_{item['value']}"] {{
                    background-color: {'#1e3c72' if is_active else 'white'} !important;
                    color: {'white' if is_active else '#333'} !important;
                    font-weight: {'600' if is_active else '400'} !important;
                    border: 1px solid {'#1e3c72' if is_active else '#e0e0e0'} !important;
                    box-shadow: {'0 2px 8px rgba(30, 60, 114, 0.3)' if is_active else '0 1px 3px rgba(0,0,0,0.1)'} !important;
                    height: 40px !important;
                    font-size: 14px !important;
                    text-align: center !important;
                }}
                .stButton > button[key="nav_btn_{item['value']}"]:hover {{
                    background-color: {'#2a5298' if is_active else '#f0f2f6'} !important;
                }}
                </style>
                """, unsafe_allow_html=True)
    
    # 主内容区域
    if st.session_state.nav_selected == "项目介绍":
        st.markdown("""
        <div class="main-content">
            <h2>CrOmLineSCNET - 干细胞定向分化驱动因子识别</h2>
            <p>
                CrOmLineSCNET是一个完整的单细胞分析框架，用于预测干细胞定向分化驱动因子，从数据预处理到轨迹推断都可以一站式完成。
                可以点击左侧的"分析流程"按钮来开始您的分析，此外，我们还提供了许多常用的单细胞分析工具，您可以通过"分析工具"获得。
            </p>
            <p>
                您可以通过以下链接来取得我们的最新成果：<a href="https://github.com/fhcjashcjshjahxhjchshcahc/CrOmLineSCNET" target="_blank" style="color: #2a5298;">
                https://github.com/fhcjashcjshjahxhjchshcahc/CrOmLineSCNET
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.nav_selected == "上传数据":
        st.markdown("""
        <div class="main-content">
            <h2>📁 数据上传</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择 .h5ad 文件",
            type=['h5ad'],
            help="支持 AnnData 格式（包含表达矩阵、细胞/基因注释）"
        )
        
        if uploaded_file is not None:
            with st.spinner("正在加载数据..."):
                adata = load_uploaded_data(uploaded_file)
                if adata is not None:
                    st.session_state.adata = adata
                    st.success("✅ 数据加载成功！")
                    cache_hvg_plot(adata)
                    
                    # 显示数据基本信息
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("细胞数量", adata.n_obs)
                    with col2:
                        st.metric("基因数量", adata.n_vars)
                    
                    st.write("**数据形状:**", adata.shape)
                    render_data_preview(adata)
        else:
            # 示例数据加载
            st.write("### 或使用示例数据")
            if st.button("加载示例数据 (PBMC3K)"):
                with st.spinner("正在加载示例数据..."):
                    adata = load_sample_data()
                    if adata is not None:
                        st.session_state.adata = adata
                        st.success("✅ 示例数据加载成功！")
                        cache_hvg_plot(adata)
                        st.rerun()
            
            # 如果已经加载了数据，显示数据信息
            if st.session_state.adata is not None:
                st.success("✅ 数据已加载！")
                cache_hvg_plot(st.session_state.adata)
                
                # 显示数据基本信息（与上传数据保持一致）
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("细胞数量", st.session_state.adata.n_obs)
                with col2:
                    st.metric("基因数量", st.session_state.adata.n_vars)
                
                st.write("**数据形状:**", st.session_state.adata.shape)
                render_data_preview(st.session_state.adata)
    
    elif st.session_state.nav_selected == "分析流程":
        if st.session_state.adata is None:
            st.warning("⚠️ 请先在『上传数据』页面加载数据（本地文件或示例数据）")
        else:
            st.markdown("""
            <div class="main-content">
                <h2>🔬 分析流程</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # 步骤按钮
            col1, col2 = st.columns(2)
            
            with col1:
                # 质量控制
                if st.button("1️⃣ 质量控制", disabled=st.session_state.analysis_completed['qc']):
                    with st.spinner("正在分析：计算QC指标→标记线粒体基因→生成可视化..."):
                        fig = perform_qc_analysis(st.session_state.adata)
                        if fig is not None:
                            st.pyplot(fig)
                            st.session_state.analysis_completed['qc'] = True
                            st.success("✅ 质量控制分析完成！")
                            create_download_button(fig, "qc_analysis.png", "📥 下载QC结果图")
                
                # 细胞聚类
                if st.button("2️⃣ 细胞聚类", disabled=not st.session_state.analysis_completed['qc']):
                    with st.spinner("正在分析：过滤细胞→标准化→高变基因→UMAP聚类..."):
                        fig, processed_adata = perform_clustering(st.session_state.adata)
                        if fig is not None:
                            st.pyplot(fig)
                            st.session_state.adata = processed_adata
                            st.session_state.analysis_completed['clustering'] = True
                            st.success("✅ 细胞聚类分析完成！")
                            create_download_button(fig, "clustering_analysis.png", "📥 下载聚类结果图")
            
            with col2:
                # 轨迹推断
                if st.button("3️⃣ 轨迹推断", disabled=not st.session_state.analysis_completed['clustering']):
                    with st.spinner("正在分析：扩散映射→选择根细胞→计算DPT轨迹..."):
                        fig = perform_trajectory_analysis(st.session_state.adata)
                        if fig is not None:
                            st.pyplot(fig)
                            st.session_state.analysis_completed['trajectory'] = True
                            st.success("✅ 轨迹推断分析完成！")
                            create_download_button(fig, "trajectory_analysis.png", "📥 下载轨迹结果图")
                
                # 基因调控网络（开发中）
                if st.button("4️⃣ 基因调控网络", disabled=not st.session_state.analysis_completed['trajectory']):
                    st.info("ℹ️ 基因调控网络分析功能正在开发中，预计下一版本上线。")
    
    elif st.session_state.nav_selected == "分析工具":
        if st.session_state.adata is None:
            st.warning("⚠️ 请先在『上传数据』页面加载数据（本地文件或示例数据）")
        else:
            st.markdown("""
            <div class="main-content">
                <h2>🛠️ 辅助分析工具</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔍 高变基因筛选")
            if st.button("执行高变基因筛选"):
                if st.session_state.hvg_result is None:
                    with st.spinner("正在准备高变基因图..."):
                        cache_hvg_plot(st.session_state.adata)
                if st.session_state.hvg_result is not None:
                    st.pyplot(st.session_state.hvg_result["fig"])
                    stats = st.session_state.hvg_result.get("stats")
                    if stats:
                        col1, col2 = st.columns(2)
                        col1.metric("总基因数", stats["total_genes"])
                        col2.metric("高变基因数量", stats["n_hvg"])
                    download_cached_image(st.session_state.hvg_result, "hvg_plot.png", "下载高变基因图")
                    st.success("高变基因可视化完成。")
                else:
                    st.warning("当前数据暂无法生成高变基因图。")
            
            st.divider()
            st.markdown("### 🧬 细胞类型注释")
            marker_file = st.file_uploader(
                "上传 marker 文件（需包含 cell_type 与 markers 两列）",
                type=['csv', 'txt'],
                key="marker_file_uploader"
            )
            if st.button("执行细胞注释"):
                if marker_file is None:
                    st.warning("请先上传 marker 文件，再进行注释。")
                else:
                    with st.spinner("正在根据 marker 基因评分并注释细胞类型..."):
                        fig, summary_df = perform_cell_annotation(st.session_state.adata, marker_file)
                        if fig is not None:
                            st.pyplot(fig)
                            create_download_button(fig, "cell_annotation.png", "下载细胞注释图")
                            if summary_df is not None:
                                st.write("**细胞注释评分汇总：**")
                                st.dataframe(summary_df, use_container_width=True)
                            st.success("细胞类型注释已完成。")
            
            st.divider()
            st.markdown("### 📈 差异基因分析")
            n_genes = st.slider("每个细胞簇展示的差异基因数量", min_value=5, max_value=30, value=20, step=5)
            if st.button("执行差异基因分析"):
                with st.spinner("正在执行差异基因分析..."):
                    fig_heatmap, fig_summary, marker_genes = perform_differential_expression(st.session_state.adata, n_genes=n_genes)
                    if fig_heatmap is not None:
                        st.pyplot(fig_heatmap)
                        create_download_button(fig_heatmap, "deg_heatmap.png", "下载差异基因热图")
                    if fig_summary is not None:
                        st.pyplot(fig_summary)
                        create_download_button(fig_summary, "deg_top_genes.png", "下载差异基因概览图")
                    if marker_genes is not None:
                        st.write("**各簇Top5差异基因：**")
                        st.dataframe(marker_genes, use_container_width=True)
                    if fig_heatmap is not None:
                        st.success("差异基因分析完成。")
    
    elif st.session_state.nav_selected == "历史记录":
        st.markdown("""
        <div class="main-content">
            <h2>📚 分析历史记录</h2>
            <p>记录每次数据上传后的分析基本信息（最多保留10条）。</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.analysis_history:
            st.write("⚠️ 暂无分析记录，请先在『上传数据』页面加载数据并进行分析。")
        else:
            # 倒序显示（最新的在最上面）
            for i, record in enumerate(reversed(st.session_state.analysis_history)):
                st.markdown(f"""
                <div class="history-card">
                    <h4>分析记录 #{len(st.session_state.analysis_history) - i}</h4>
                    <p><strong>上传时间:</strong> {record['timestamp']}</p>
                    <p><strong>数据来源:</strong> {record['data_source']}</p>
                    <p><strong>数据规模:</strong> 细胞数 {record['n_cells']} | 基因数 {record['n_genes']}</p>
                    <p><strong>已完成步骤:</strong> {', '.join(record['completed_steps']) if record['completed_steps'] else '无'}</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
