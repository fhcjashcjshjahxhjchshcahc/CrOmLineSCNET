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
        cluster_counts = adata.obs['louvain'].value_counts().sort_index()
        st.write("**各簇细胞数量:**")
        cluster_df = pd.DataFrame({
            '簇': cluster_counts.index,
            '细胞数量': cluster_counts.values
        })
        st.dataframe(cluster_df)
        
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
        axes[0].set_title('UMAP上的假时间分布', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('UMAP 1', fontsize=14)
        axes[0].set_ylabel('UMAP 2', fontsize=14)
        
        sc.pl.diffmap(adata, color='dpt_pseudotime', ax=axes[1], show=False, cmap='plasma')
        axes[1].set_title('扩散映射上的假时间分布', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Diffusion Component 1', fontsize=14)
        axes[1].set_ylabel('Diffusion Component 2', fontsize=14)
        
        plt.tight_layout()
        
        # 标记轨迹步骤完成
        update_analysis_step("轨迹推断")
        return fig
        
    except Exception as e:
        st.error(f"轨迹分析失败: {str(e)}")
        return None


def create_download_button(fig, filename, button_text="下载图片"):
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


# 主界面
def main():
    # 引入Font Awesome并设置自定义样式
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        /* 顶端固定深色栏 */
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem 2rem;
            margin: -1rem -1rem 2rem -1rem;
            color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            font-size: 1.8rem;
            margin: 0;
            padding: 0;
        }

        /* 侧边栏样式 */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }

        /* 导航按钮样式 */
        .nav-button {
            width: 100%;
            padding: 0.75rem 1rem;
            margin: 0.25rem 0;
            border: none;
            border-radius: 8px;
            background: none;
            text-align: left;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
            color: #333;
        }
        .nav-button:hover {
            background-color: #e9ecef;
            transform: translateX(5px);
        }
        .nav-button.active {
            background-color: #1e3c72;
            color: white;
            font-weight: 600;
        }
        .nav-button i {
            margin-right: 0.75rem;
            width: 1.25rem;
            text-align: center;
        }

        /* 主内容卡片 */
        .main-content {
            padding: 2rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
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

        /* 历史记录卡片样式 */
        .history-card {
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.8rem 0;
            background-color: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 顶端标题栏
    st.markdown("""
    <div class="main-header">
        <h1>🧬 CrOmLineSCNET - 干细胞定向分化驱动因子识别</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏导航 - 使用简单的按钮方式
    with st.sidebar:
        st.markdown("## 🧭 导航")
        
        # 定义导航项
        nav_items = [
            {"icon": "📊", "label": "项目介绍", "value": "intro"},
            {"icon": "📁", "label": "上传数据", "value": "upload"},
            {"icon": "🔬", "label": "分析流程", "value": "pipeline"},
            {"icon": "🛠️", "label": "分析工具", "value": "tools"},
            {"icon": "📚", "label": "历史记录", "value": "history"}
        ]
        
        # 创建导航按钮
        for item in nav_items:
            is_active = st.session_state.nav_selected == item["value"]
            button_type = "primary" if is_active else "secondary"
            
            if st.button(
                f"{item['icon']} {item['label']}",
                key=f"nav_{item['value']}",
                use_container_width=True,
                type=button_type
            ):
                st.session_state.nav_selected = item["value"]
                st.rerun()
        
        st.markdown("---")
        
        # 数据状态显示
        st.markdown("### 📊 数据状态")
        if st.session_state.adata is not None:
            st.success("✅ 数据已加载")
            st.write(f"**细胞数:** {st.session_state.adata.n_obs}")
            st.write(f"**基因数:** {st.session_state.adata.n_vars}")
            
            # 显示分析进度
            completed_steps = sum(st.session_state.analysis_completed.values())
            total_steps = len(st.session_state.analysis_completed)
            st.write(f"**分析进度:** {completed_steps}/{total_steps} 步骤完成")
        else:
            st.warning("⚠️ 未加载数据")
            st.info("请在『上传数据』页面加载数据开始分析")

    # 根据选中的导航项显示对应内容
    if st.session_state.nav_selected == "intro":
        show_intro_page()
    elif st.session_state.nav_selected == "upload":
        show_upload_page()
    elif st.session_state.nav_selected == "pipeline":
        show_pipeline_page()
    elif st.session_state.nav_selected == "tools":
        show_tools_page()
    elif st.session_state.nav_selected == "history":
        show_history_page()


def show_intro_page():
    """显示项目介绍页面"""
    st.markdown("""
    <div class="main-content">
        <h2>📊 项目介绍</h2>
        <p>
            <strong>CrOmLineSCNET</strong> 是一个完整的单细胞分析框架，专门用于预测干细胞定向分化驱动因子。
            从数据预处理到轨迹推断，所有分析步骤都可以在这个平台上一站式完成。
        </p>
        
        <h3>🎯 主要功能</h3>
        <ul>
            <li><strong>数据质量控制</strong> - 全面的数据质量评估和可视化</li>
            <li><strong>细胞聚类分析</strong> - 自动化的细胞分群和可视化</li>
            <li><strong>发育轨迹推断</strong> - 构建细胞分化路径和假时间分析</li>
            <li><strong>分析工具集</strong> - 多种单细胞分析辅助工具</li>
            <li><strong>历史记录管理</strong> - 分析过程的完整追踪</li>
        </ul>
        
        <h3>🚀 快速开始指南</h3>
        <ol>
            <li>在<strong>「上传数据」</strong>页面加载单细胞数据文件(.h5ad格式)或使用示例数据</li>
            <li>进入<strong>「分析流程」</strong>页面，按顺序执行各个分析步骤</li>
            <li>在<strong>「分析工具」</strong>页面使用额外的分析功能</li>
            <li>查看<strong>「历史记录」</strong>页面了解分析历程</li>
        </ol>
        
        <h3>🔗 相关资源</h3>
        <p>
            项目代码和详细文档请访问：
            <a href="https://github.com/fhcjashcjshjahxhjchshcahc/CrOmLineSCNET" target="_blank">
            https://github.com/fhcjashcjshjahxhjchshcahc/CrOmLineSCNET
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_upload_page():
    """显示数据上传页面"""
    st.markdown("""
    <div class="main-content">
        <h2>📁 数据上传</h2>
        <p>请上传您的单细胞数据文件或使用示例数据开始分析。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 文件上传区域
    st.subheader("上传数据文件")
    uploaded_file = st.file_uploader(
        "选择 .h5ad 格式的单细胞数据文件",
        type=['h5ad'],
        help="支持 AnnData 格式文件，包含基因表达矩阵和细胞注释信息"
    )
    
    if uploaded_file is not None:
        with st.spinner("正在加载数据..."):
            adata = load_uploaded_data(uploaded_file)
            if adata is not None:
                st.session_state.adata = adata
                st.success("✅ 数据加载成功！")
                
                # 显示数据基本信息
                st.subheader("数据概览")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("细胞数量", adata.n_obs)
                with col2:
                    st.metric("基因数量", adata.n_vars)
                with col3:
                    st.metric("数据维度", f"{adata.shape[0]} × {adata.shape[1]}")
                
                # 数据预览
                st.subheader("数据预览")
                st.write("前5个细胞和前5个基因的表达矩阵（仅显示非零值）:")
                if hasattr(adata, 'X') and adata.X is not None:
                    if hasattr(adata.X, 'toarray'):
                        preview_data = adata.X[:5, :5].toarray()
                    else:
                        preview_data = adata.X[:5, :5]
                    # 将极小值设为0以便于显示
                    preview_data[preview_data < 1e-6] = 0
                    st.dataframe(pd.DataFrame(
                        preview_data.round(3),
                        index=adata.obs_names[:5],
                        columns=adata.var_names[:5]
                    ))
                else:
                    st.error("❌ 数据矩阵为空或无法访问")
    
    # 示例数据区域
    st.markdown("---")
    st.subheader("使用示例数据")
    st.write("如果您没有准备好数据文件，可以使用我们的示例数据进行体验：")
    
    if st.button("加载 PBMC3K 示例数据", use_container_width=True):
        with st.spinner("正在加载PBMC3K示例数据..."):
            adata = load_sample_data()
            if adata is not None:
                st.session_state.adata = adata
                st.success("✅ 示例数据加载成功！")
                st.rerun()


def show_pipeline_page():
    """显示分析流程页面"""
    st.markdown("""
    <div class="main-content">
        <h2>🔬 分析流程</h2>
        <p>请按顺序执行以下分析步骤，每个步骤都依赖前一步的结果。</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.adata is None:
        st.warning("""
        ⚠️ 尚未加载数据！
        
        请先在 **「上传数据」** 页面：
        - 上传您的 .h5ad 格式数据文件，或
        - 使用 PBMC3K 示例数据进行体验
        """)
        return
    
    # 分析步骤展示
    steps = [
        {
            "name": "质量控制",
            "description": "数据质量评估和预处理",
            "completed": st.session_state.analysis_completed['qc'],
            "function": perform_qc_analysis
        },
        {
            "name": "细胞聚类", 
            "description": "细胞分群和可视化",
            "completed": st.session_state.analysis_completed['clustering'],
            "function": perform_clustering
        },
        {
            "name": "轨迹推断",
            "description": "发育轨迹构建和假时间分析", 
            "completed": st.session_state.analysis_completed['trajectory'],
            "function": perform_trajectory_analysis
        },
        {
            "name": "基因调控网络",
            "description": "基因调控网络分析",
            "completed": st.session_state.analysis_completed['grn'],
            "function": None
        }
    ]
    
    # 显示步骤状态
    st.subheader("分析进度")
    for i, step in enumerate(steps, 1):
        status = "✅" if step["completed"] else "⏳"
        st.write(f"{i}. {status} **{step['name']}** - {step['description']}")
    
    st.markdown("---")
    
    # 分析执行区域
    st.subheader("执行分析")
    
    for i, step in enumerate(steps):
        with st.expander(f"步骤 {i+1}: {step['name']}", expanded=not step['completed']):
            st.write(step['description'])
            
            # 检查前置条件
            if i > 0 and not steps[i-1]["completed"]:
                st.warning(f"请先完成前一步骤: **{steps[i-1]['name']}**")
                continue
                
            if step["completed"]:
                st.success("✅ 此步骤已完成")
            elif step["function"] is not None:
                if st.button(f"执行 {step['name']}", key=f"step_{i}", use_container_width=True):
                    with st.spinner(f"正在执行 {step['name']}..."):
                        if step['name'] == "质量控制":
                            fig = step["function"](st.session_state.adata)
                            if fig is not None:
                                st.pyplot(fig)
                                st.session_state.analysis_completed['qc'] = True
                                st.success(f"✅ {step['name']}完成！")
                                create_download_button(fig, "qc_analysis.png", "📥 下载QC结果图")
                                
                        elif step['name'] == "细胞聚类":
                            fig, processed_adata = step["function"](st.session_state.adata)
                            if fig is not None:
                                st.pyplot(fig)
                                st.session_state.adata = processed_adata
                                st.session_state.analysis_completed['clustering'] = True
                                st.success(f"✅ {step['name']}完成！")
                                create_download_button(fig, "clustering_analysis.png", "📥 下载聚类结果图")
                                
                        elif step['name'] == "轨迹推断":
                            fig = step["function"](st.session_state.adata)
                            if fig is not None:
                                st.pyplot(fig)
                                st.session_state.analysis_completed['trajectory'] = True
                                st.success(f"✅ {step['name']}完成！")
                                create_download_button(fig, "trajectory_analysis.png", "📥 下载轨迹结果图")
            else:
                st.info("🔧 此功能正在开发中，敬请期待...")


def show_tools_page():
    """显示分析工具页面"""
    st.markdown("""
    <div class="main-content">
        <h2>🛠️ 分析工具</h2>
        <p>使用以下工具进行更深入的单项分析：</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.adata is None:
        st.warning("⚠️ 请先在『上传数据』页面加载数据")
        return
    
    # 工具卡片布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 高变基因筛选")
        st.write("识别数据中高度可变的基因，用于下游分析")
        if st.button("运行高变基因分析", use_container_width=True):
            st.info("ℹ️ 高变基因筛选功能将在下一版本中开放")
            
        st.subheader("📊 PCA降维分析") 
        st.write("主成分分析，探索数据的主要变异方向")
        if st.button("运行PCA分析", use_container_width=True):
            st.info("ℹ️ PCA降维分析功能将在下一版本中开放")
    
    with col2:
        st.subheader("🧬 差异基因分析")
        st.write("识别不同细胞簇之间的差异表达基因")
        if st.button("运行差异分析", use_container_width=True):
            st.info("ℹ️ 差异基因分析功能将在下一版本中开放")
            
        st.subheader("📈 基因表达可视化")
        st.write("查看特定基因在不同细胞中的表达模式")
        if st.button("查看基因表达", use_container_width=True):
            st.info("ℹ️ 基因表达可视化功能将在下一版本中开放")


def show_history_page():
    """显示历史记录页面"""
    st.markdown("""
    <div class="main-content">
        <h2>📚 分析历史记录</h2>
        <p>记录每次数据上传后的分析基本信息（最多保留10条记录）。</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.info("""
        📝 暂无分析记录
        
        要开始记录分析历史，请：
        1. 在 **「上传数据」** 页面加载数据
        2. 在 **「分析流程」** 页面执行分析步骤
        3. 分析记录将自动保存在这里
        """)
    else:
        # 显示历史记录
        for i, record in enumerate(reversed(st.session_state.analysis_history)):
            with st.container():
                st.markdown(f"""
                <div class="history-card">
                    <h4>📋 分析记录 #{len(st.session_state.analysis_history) - i}</h4>
                    <p><strong>🕒 上传时间:</strong> {record['timestamp']}</p>
                    <p><strong>📂 数据来源:</strong> {record['data_source']}</p>
                    <p><strong>📊 数据规模:</strong> {record['n_cells']} 个细胞 × {record['n_genes']} 个基因</p>
                    <p><strong>✅ 已完成步骤:</strong> {', '.join(record['completed_steps']) if record['completed_steps'] else '暂无完成的步骤'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 清空历史记录按钮
        if st.button("清空所有历史记录", type="secondary"):
            st.session_state.analysis_history = []
            st.success("历史记录已清空")
            st.rerun()


if __name__ == "__main__":
    main()
        

