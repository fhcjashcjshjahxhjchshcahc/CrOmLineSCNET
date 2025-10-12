# 标准库
import io
import datetime
from PIL import Image

# 第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import streamlit as st

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
    st.session_state.nav_selected = "intro"


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
        axes[0,0].set_title('每个细胞的基因数量', fontsize=14, fontweight='bold')
        
        sc.pl.violin(adata, 'total_counts', ax=axes[0,1], jitter=0.4, show=False)
        axes[0,1].set_title('每个细胞的总计数', fontsize=14, fontweight='bold')
        
        sc.pl.violin(adata, 'pct_counts_mt', ax=axes[0,2], jitter=0.4, show=False)
        axes[0,2].set_title('线粒体基因比例', fontsize=14, fontweight='bold')
        
        # 第二行：散点图和最高表达基因
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=axes[1,0], show=False)
        axes[1,0].set_title('总计数 vs 线粒体基因比例', fontsize=14, fontweight='bold')
        
        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=axes[1,1], show=False)
        axes[1,1].set_title('总计数 vs 基因数量', fontsize=14, fontweight='bold')
        
        sc.pl.highest_expr_genes(adata, n_top=20, ax=axes[1,2], show=False)
        axes[1,2].set_title('最高表达基因', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 显示QC统计信息
        st.write("**QC统计信息:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均基因数", f"{adata.obs['n_genes_by_counts'].mean():.0f}")
        with col2:
            st.metric("平均总计数", f"{adata.obs['total_counts'].mean():.0f}")
        with col3:
            st.metric("平均线粒体比例", f"{adata.obs['pct_counts_mt'].mean():.2f}%")
        
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
        st.write(f"**聚类结果:** 发现 {n_clusters} 个细胞簇")
        
        # 显示每个簇的细胞数量
        
