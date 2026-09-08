import glob
import hmac
import os
import time

import streamlit as st

# Active application services are separated from the Streamlit interface.
from config import KB_PATH as CONFIG_KB_PATH
from interaction_rules import (
    find_interaction as lookup_interaction,
    get_drug_list as list_demo_drugs,
)
from llm_service import (
    generate_grounded_answer,
    page_number,
    source_name,
)
from rag_core import extract_pdf_text, save_uploaded_pdf, search_knowledge_base


st.set_page_config(page_title="AI Pharmacist", layout="wide", initial_sidebar_state="expanded")

# ---- 初始化会话状态 ----
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'triage_messages' not in st.session_state:
    st.session_state.triage_messages = []
if 'session_queries' not in st.session_state:
    st.session_state.session_queries = 0
if 'session_safety_flags' not in st.session_state:
    st.session_state.session_safety_flags = 0
if 'last_api_latency_ms' not in st.session_state:
    st.session_state.last_api_latency_ms = None

# ---- 双语词典 ----
T = {
    'en': {
        'title': 'AI Pharmacist',
        'subtitle': 'Document-Grounded Medication Information Assistant',
        'status_online': '● Online',
        'status_admin': 'Admin',
        'sign_in': '🔐 Sign In',
        'password_placeholder': 'Enter access key',
        'auth_failed': 'Authentication failed. Please try again.',
        'nav_title': '📋 Navigation',
        'nav_pharmacist': '💊 AI Pharmacist',
        'nav_triage': '💬 Guided Questions',
        'nav_dashboard': '📊 Dashboard',
        'nav_settings': '⚙️ Settings',
        'logout': '🚪 Logout',
        'smart_query_title': '🔍 Medication AI Query',
        'smart_query_hint': 'Enter a drug name or medication-information question to search the package-insert knowledge base.',
        'smart_query_placeholder': 'Example: What warnings are listed for ibuprofen?',
        'execute_query': 'AI Search',
        'scanning': 'Searching drug insert knowledge base via RAG...',
        'kb_empty': 'Knowledge base is empty. Please add PDF files to Pharmacist_Pro/knowledge_base/.',
        'empty_warning': 'Please enter a drug name or medication-information question.',
        'clinical_report': '📋 Query Result',
        'awaiting': 'Awaiting query...',
        'awaiting_hint': 'Enter a drug name or medical question and click AI Search',
        'view_source': '📄 View Source Documents (Vector Retrieval)',
        'source_label': 'Source',
        'page_label': 'Page',
        'no_source': 'No source documents available.',
        'pdf_upload': '📁 PDF Manual Parser',
        'pdf_upload_hint': 'Upload a drug package insert (PDF) to extract text content.',
        'pdf_parsed': '📄 Parsed',
        'pdf_chars': 'chars',
        'pdf_parse_error': 'PDF parse error',
        'pdf_drop_hint': 'Drag & drop or click to upload PDF',
        'interaction_title': '⚠️ Interaction Demo Rules',
        'drug_a_label': 'Drug A',
        'drug_b_label': 'Drug B',
        'check_interaction': '🔍 Check Interaction',
        'interaction_select_hint': 'Please select both drugs to check.',
        'interaction_safe': 'ℹ️ No matching rule exists in this small demo dataset. This does not establish that the combination is safe.',
        'interaction_caution': '⚠️ Potential interaction in the demo rule set — verify with an authoritative interaction database or pharmacist.',
        'interaction_danger': '🚨 Potential high-risk interaction in the demo rule set — professional verification is required.',
        'triage_title': '💬 Guided Medication Questions',
        'triage_subtitle': 'Multi-turn, source-grounded information retrieval from local package inserts',
        'triage_welcome': 'Hello. I can search the local package-insert collection and summarize what the documents say. I cannot diagnose, prescribe, or decide whether a medicine is safe for you.\n\nAsk a medication question and include relevant context. If the documents do not contain the answer, I will say so rather than fill the gap from model memory.',
        'triage_placeholder': 'Describe your symptoms here...',
        'triage_thinking': 'Analyzing your symptoms...',
        'triage_disclaimer': '⚠️ Educational information only; not diagnosis or treatment advice.',
        'triage_clear': '🗑️ Clear Conversation',
        'dashboard_title': '📊 System Dashboard',
        'api_latency': 'API Latency',
        'vectors_scanned': 'Vectors Scanned',
        'risk_alerts': 'Risk Alerts',
        'uptime': 'System Uptime',
        'throughput': '📈 Query & Token Throughput',
        'recent_activity': '📋 Recent Activity',
        'settings_title': '⚙️ System Settings',
        'api_config': '🔧 API Configuration',
        'rag_status': '📚 RAG Knowledge Base',
        'disclaimer_title': '🛡️ Medical Disclaimer',
        'disclaimer_text': 'This portfolio application demonstrates document-grounded medication information retrieval. It is not a medical device and must not be used to diagnose, prescribe, select treatment, or make medication changes. Verify every output against the cited package insert and an authoritative clinical source. Seek professional or emergency care when appropriate.',
    },
    'zh': {
        'title': 'AI药剂师',
        'subtitle': '基于药品说明书的资料检索助手',
        'status_online': '● 在线',
        'status_admin': '行政',
        'sign_in': '🔐 登录系统',
        'password_placeholder': '请输入访问密码',
        'auth_failed': '认证失败，请重试。',
        'nav_title': '📋 功能导航',
        'nav_pharmacist': '💊 AI 药剂师',
        'nav_triage': '💬 连续资料问答',
        'nav_dashboard': '📊 系统看板',
        'nav_settings': '⚙️ 系统设置',
        'logout': '🚪 退出登录',
        'smart_query_title': '🔍 药品智能查询',
        'smart_query_hint': '输入药品名称或临床问题（如"肚子疼能不能吃布洛芬"），基于药品说明书知识库进行 RAG 检索',
        'smart_query_placeholder': '药品名称 / 临床问题，如：阿司匹林、肚子疼能不能吃布洛芬',
        'execute_query': 'AI 智能查询',
        'scanning': '正在 RAG 检索药品说明书知识库...',
        'kb_empty': '知识库为空，请将 PDF 说明书放入 Pharmacist_Pro/knowledge_base/ 目录。',
        'empty_warning': '请输入药品名称或临床问题。',
        'clinical_report': '📋 查询结果',
        'awaiting': '等待查询...',
        'awaiting_hint': '输入药品名称或医疗问题后点击智能查询',
        'view_source': '📄 查看源文档（矢量检索）',
        'source_label': '来源',
        'page_label': '页码',
        'no_source': '暂无可显示的源文档。',
        'pdf_upload': '📁 PDF 说明书上传解析',
        'pdf_upload_hint': '上传药品说明书 PDF 文件，自动提取文本内容。',
        'pdf_parsed': '📄 解析结果',
        'pdf_chars': '字符',
        'pdf_parse_error': 'PDF 解析失败',
        'pdf_drop_hint': '拖拽或点击上传 PDF 文件',
        'interaction_title': '⚠️ 相互作用演示规则',
        'drug_a_label': '药品 A',
        'drug_b_label': '药品 B',
        'check_interaction': '🔍 检查相互作用',
        'interaction_select_hint': '请选择两种药品进行检查。',
        'interaction_safe': 'ℹ️ 这个小型演示数据集中没有匹配规则；这不代表两种药物可以安全联用。',
        'interaction_caution': '⚠️ 演示规则中存在潜在相互作用，请使用权威相互作用数据库或咨询药师核实。',
        'interaction_danger': '🚨 演示规则中存在潜在高风险相互作用，必须由专业人员核实。',
        'triage_title': '💬 连续用药资料问答',
        'triage_subtitle': '基于本地药品说明书的多轮、有来源资料检索',
        'triage_welcome': '您好。我可以检索本地药品说明书，并概括文档中明确写出的内容；不能诊断、开药，也不能判断某种药是否适合您。\n\n请提出用药问题并提供必要背景。资料中没有答案时，我会明确说明“未找到”，不会用模型记忆补写。',
        'triage_placeholder': '请描述您的症状...',
        'triage_thinking': '正在分析您的症状...',
        'triage_disclaimer': '⚠️ 仅作资料学习，不构成诊断或治疗建议。',
        'triage_clear': '🗑️ 清除对话',
        'dashboard_title': '📊 系统看板',
        'api_latency': 'API 延迟',
        'vectors_scanned': '向量检索数',
        'risk_alerts': '风险警报',
        'uptime': '系统运行时间',
        'throughput': '📈 查询与 Token 吞吐量',
        'recent_activity': '📋 最近活动',
        'settings_title': '⚙️ 系统设置',
        'api_config': '🔧 API 配置',
        'rag_status': '📚 RAG 知识库状态',
        'disclaimer_title': '🛡️ 医疗免责声明',
        'disclaimer_text': '这是展示“基于文档检索”的作品集应用，不是医疗器械，不得用于诊断、开药、选择治疗或调整用药。所有回答都必须对照所引说明书和权威临床资料独立核实；必要时请及时就医。',
    }
}

def t(key):
    return T[st.session_state.lang].get(key, key)

# ============================================================
# 🎨 薄荷绿柔和医疗商务风 CSS
# ============================================================
st.html("""
<style>
    .stApp { background-color: #faf9f6 !important; color: #1f2937 !important; }

    /* Precise show/reopen control for the COLLAPSED SIDEBAR only.
       Streamlit exposes this under a few native test-ids depending on version.
       IMPORTANT: we must NOT match the "three-dots" main menu
       ([data-testid="stMainMenu"]) or generic header buttons, so we do not
       include header/button selectors here. The real sidebar-reopen control
       is one of these, and we leave stMainMenu untouched. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        position: fixed !important;
        left: 12px !important;
        top: 12px !important;
        z-index: 999999 !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        width: 34px;
        height: 34px;
        background: #ffffff !important;
        color: #0d9488 !important;
        fill: #0d9488 !important;
        border: 1px solid #d5dbe4 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18) !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="collapsedControl"]:hover,
    [data-testid="stSidebarCollapseButton"]:hover {
        background: #e0f2ef !important;
        border-color: #0d9488 !important;
    }
    /* Colour the icon inside the collapsed-sidebar control only, never the
       main-menu dots or generic header buttons. */
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #0d9488 !important;
        color: #0d9488 !important;
    }

    /* Never touch Streamlit's main ("three-dots") menu button. */
    [data-testid="stMainMenu"],
    [data-testid="stMainMenu"] svg {}

    /* Keep the app header element in normal flow so the top chrome and the
       sidebar controls are not clipped; hide only the extra page chrome. */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef1f5 0%, #edf0f5 100%) !important;
        border-right: 1px solid #dde1e8 !important;
    }
    [data-testid="stSidebar"] * { color: #374151 !important; }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        color: #4b5563 !important; font-weight: 500;
        padding: 10px 14px; border-radius: 10px; transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stRadio [data-checked="true"] label {
        background: #e0f2ef !important; color: #0d9488 !important; font-weight: 700;
    }
    [data-testid="stSidebar"] button {
        background: #ffffff !important; color: #6b7280 !important;
        border: 1px solid #dde1e8 !important; border-radius: 10px !important;
        font-weight: 500 !important; transition: all 0.25s ease !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: #e0f2ef !important; color: #0d9488 !important;
        border-color: #0d9488 !important;
    }

    .glass-card {
        background: #f8f9fb;
        border: 1px solid #e8ecf1; border-radius: 14px;
        padding: 22px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03);
        margin-bottom: 16px;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(13,148,136,0.06), 0 6px 20px rgba(0,0,0,0.05);
    }

    /* 临床报告专用白底容器 */
    .report-card {
        background: #ffffff;
        border: 1px solid #e8ecf1; border-radius: 14px;
        padding: 24px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }

    .top-nav {
        display: flex; justify-content: space-between; align-items: center;
        padding: 16px 28px; margin-top: 0; margin-bottom: 22px;
        background: #f8f9fb; border-bottom: 1px solid #e8ecf1;
        border-radius: 0 0 14px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .logo-text { font-size: 21px; font-weight: 700; color: #1f2937; letter-spacing: 0.3px; }
    .logo-accent {
        background: linear-gradient(135deg, #0d9488, #14b8a6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 13px; margin-left: 6px;
    }
    .lang-btn {
        display: inline-block; padding: 5px 16px; border-radius: 20px;
        font-size: 12px; font-weight: 600;
        border: 1px solid #dde1e8; background: #fff; color: #6b7280;
        transition: all 0.2s ease; margin: 0 3px;
    }
    .lang-btn.active { background: #0d9488; color: #fff; border-color: #0d9488; }

    .metric-value { font-size: 36px; font-weight: 700; font-family: 'Inter', 'Segoe UI', sans-serif; }
    .metric-teal  { color: #0d9488; }
    .metric-blue  { color: #5b9bd5; }
    .metric-amber { color: #d4a853; }
    .metric-label {
        font-size: 11px; color: #9ca3af; text-transform: uppercase;
        letter-spacing: 1.3px; margin-bottom: 6px; font-weight: 600;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox > div {
        background-color: #ffffff !important; color: #1f2937 !important;
        border: 1px solid #dde1e8 !important; border-radius: 10px !important;
        font-size: 14px !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 3px rgba(13,148,136,0.08) !important;
    }

    .stButton > button {
        background: #0d9488 !important; color: #ffffff !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; letter-spacing: 0.3px;
        padding: 10px 22px !important; transition: all 0.25s ease !important;
        box-shadow: 0 1px 2px rgba(13,148,136,0.18) !important;
    }
    .stButton > button:hover {
        background: #0f766e !important;
        box-shadow: 0 6px 18px rgba(13,148,136,0.26) !important;
        transform: translateY(-1px) !important;
    }

    .login-card {
        background: #f8f9fb; border: 1px solid #e8ecf1; border-radius: 20px;
        padding: 44px 36px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 10px 36px rgba(0,0,0,0.05);
    }

    .streamlit-expanderHeader {
        background: #f8f9fb !important; border-radius: 10px !important;
        border: 1px solid #e8ecf1 !important; color: #374151 !important;
        font-weight: 600 !important; padding: 10px 16px !important;
    }
    .streamlit-expanderHeader:hover {
        background: #e0f2ef !important; border-color: #0d9488 !important;
    }
    .streamlit-expanderContent {
        border: 1px solid #e8ecf1 !important; border-top: none !important;
        border-radius: 0 0 10px 10px !important; padding: 12px 16px !important;
    }

    hr { border-color: #e8ecf1 !important; margin: 1.2rem 0 !important; }

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
    }

    .block-container {
        /* Leave room below the (now visible) Streamlit header so the top-nav
           title is never clipped on desktop or mobile. */
        padding-top: 4rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-top: 4.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .top-nav { padding-left: 14px; padding-right: 14px; }
    }

    [data-testid="stFileUploader"] section {
        background: #ffffff !important; border: 2px dashed #dde1e8 !important;
        border-radius: 12px !important; padding: 16px !important;
        transition: border-color 0.25s !important;
    }
    [data-testid="stFileUploader"] section:hover { border-color: #0d9488 !important; }

    .tag {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 11px; font-weight: 600;
    }
    .tag-teal  { background: #e0f2ef; color: #0d9488; }
    .tag-blue  { background: #e8f0fa; color: #5b9bd5; }
    .tag-amber { background: #fef9ee; color: #b8860b; }
    .tag-red   { background: #fef0f0; color: #c53030; }

    .compact-card {
        background: #f8f9fb; border: 1px solid #e8ecf1; border-radius: 14px;
        padding: 20px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03);
    }

    .stSelectbox label {
        font-size: 12px !important; color: #6b7280 !important; font-weight: 500 !important;
    }
</style>
""")

# ============================================================
# --- 登录页面 ---
# ============================================================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 1.1, 1])
    with col2:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        st.markdown('<div class="login-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown(
            f'<h1 style="font-size:36px; color:#1f2937; font-weight:700; margin-bottom:4px;">'
            f'{t("title")}</h1>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:#9ca3af; font-size:14px; margin-bottom:28px;">'
            f'{t("subtitle")}</p>', unsafe_allow_html=True)
        pwd = st.text_input(
            "SYSTEM PASSWORD", type="password",
            placeholder=t('password_placeholder'), label_visibility="collapsed")
        if st.button(t('sign_in'), use_container_width=True):
            configured_password = os.getenv("ADMIN_PASSWORD")
            if not configured_password:
                st.error("ADMIN_PASSWORD is not configured in the environment.")
            elif hmac.compare_digest(pwd, configured_password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error(t('auth_failed'))
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# --- 主系统页面 ---
# ============================================================
else:
    # ---- 侧边栏 ----
    with st.sidebar:
        st.markdown(
            f'<div style="padding:8px 0 16px 0;">'
            f'<h3 style="color:#374151; font-weight:700; letter-spacing:0.2px; margin:0;">'
            f'{t("nav_title")}</h3></div>', unsafe_allow_html=True)
        menu = st.radio("", [
            t('nav_pharmacist'), t('nav_triage'), t('nav_dashboard'), t('nav_settings'),
        ], label_visibility="collapsed")
        st.markdown("<br>" * 10, unsafe_allow_html=True)

        if st.session_state.search_history:
            st.markdown(
                '<p style="color:#9ca3af; font-size:11px; font-weight:600; '
                'text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">🕐 History</p>',
                unsafe_allow_html=True)
            for h in st.session_state.search_history[-8:]:
                st.markdown(
                    f'<div style="font-size:11px; color:#6b7280; padding:4px 8px; '
                    f'background:#fff; border-radius:6px; margin-bottom:4px; '
                    f'border:1px solid #e8ecf1; overflow:hidden; text-overflow:ellipsis; '
                    f'white-space:nowrap;">{h[:45]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="border-top:1px solid #dde1e8; padding-top:14px;">',
            unsafe_allow_html=True)
        if st.button(t('logout'), use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- 顶部 Header ----
    st.markdown(f"""
        <div class="top-nav">
            <div class="logo-text">
                {t('title')}<span class="logo-accent">PRO</span>
            </div>
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="color:#9ca3af; font-size:12px; font-weight:500;">
                    <span style="color:#0d9488; font-weight:700;">{t('status_online')}</span>
                    &nbsp;|&nbsp;
                    <span style="color:#6b7280;">{t('status_admin')}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 语言切换按钮（右侧对齐）
    _, _, lang_col = st.columns([4, 3.6, 0.6])
    with lang_col:
        next_lang = '中' if st.session_state.lang == 'en' else 'EN'
        if st.button(next_lang, key="lang_toggle", help="Switch Language / 切换语言",
                     use_container_width=True):
            st.session_state.lang = 'zh' if st.session_state.lang == 'en' else 'en'
            st.rerun()

    # ============================================================
    # 视图: AI 药剂师
    # ============================================================
    if menu == t('nav_pharmacist'):
        # ---- 查询与知识库上传 ----
        tab1, tab2 = st.tabs(["🔍 文本检索", "📄 PDF 文献上传"])

        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                f'<h3 style="color:#0d9488; font-weight:700; margin-top:0; margin-bottom:6px;">'
                f'{t("smart_query_title")}</h3>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:#9ca3af; font-size:12px; margin-bottom:10px; line-height:1.5;">'
                f'{t("smart_query_hint")}</p>', unsafe_allow_html=True)
            query = st.text_area(
                "", placeholder=t('smart_query_placeholder'), height=90,
                label_visibility="collapsed", key="smart_query")
            if st.button(t('execute_query'), use_container_width=True, key="btn_query"):
                # ---- Start of a new AI search: drop any previous result ----
                for _k in ('res', 'docs', 'search_results'):
                    st.session_state.pop(_k, None)
                st.session_state.last_api_latency_ms = None
                if not query.strip():
                    st.warning(t('empty_warning'))
                else:
                    with st.spinner(t('scanning')):
                        results = search_knowledge_base(query)
                    if not results:
                        st.warning(
                            "No sufficiently relevant source was found. Try a drug name "
                            "or add the correct package insert."
                            if st.session_state.lang == "en"
                            else "没有找到相关度足够的资料。请换成药品名称，或上传对应说明书。"
                        )
                    else:
                        with st.spinner(t('scanning')):
                            docs = [item.document for item in results]
                            try:
                                started_at = time.perf_counter()
                                res = generate_grounded_answer(docs, query)
                                st.session_state.last_api_latency_ms = round(
                                    (time.perf_counter() - started_at) * 1000
                                )
                                st.session_state['res'] = res
                                st.session_state['docs'] = docs
                                st.session_state['search_results'] = results
                                st.session_state.session_queries += 1
                            except Exception as e:
                                st.error(f"大模型调用失败: {str(e)}")
                        st.session_state.search_history.insert(0, query.strip())
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                f'<h4 style="color:#1f2937; font-weight:700; margin-top:0; margin-bottom:4px;">'
                f'{t("pdf_upload")}</h4>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:#9ca3af; font-size:12px; margin-bottom:10px;">'
                f'{t("pdf_upload_hint")}</p>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "", type=["pdf"], label_visibility="collapsed", key="pdf_uploader")
            if uploaded_file is not None:
                try:
                    pdf_bytes = uploaded_file.getvalue()
                    pdf_text = extract_pdf_text(pdf_bytes)
                    if len(pdf_text) < 30:
                        raise ValueError(
                            "No usable text detected; this PDF may be scanned and require OCR."
                        )
                    st.markdown(
                        f'<p style="color:#0d9488; font-size:12px; font-weight:600; margin-top:8px;">'
                        f'{t("pdf_parsed")}: {len(pdf_text):,} {t("pdf_chars")}</p>',
                        unsafe_allow_html=True)
                    with st.expander(f"📄 {uploaded_file.name}", expanded=False):
                        st.text_area(
                            "", value=pdf_text[:5000], height=180,
                            label_visibility="collapsed", key="pdf_content")
                    add_label = (
                        "Add to knowledge base"
                        if st.session_state.lang == "en"
                        else "加入知识库"
                    )
                    if st.button(add_label, use_container_width=True, key="index_uploaded_pdf"):
                        saved_path, _ = save_uploaded_pdf(uploaded_file.name, pdf_bytes)
                        st.success(
                            f"Indexed: {saved_path.name}"
                            if st.session_state.lang == "en"
                            else f"已加入知识库：{saved_path.name}"
                        )
                except Exception as e:
                    st.error(f"{t('pdf_parse_error')}: {str(e)}")
            else:
                st.markdown(
                    f'<div style="text-align:center; padding:40px 16px; color:#c4cad4;">'
                    f'<div style="font-size:32px; margin-bottom:8px;">📁</div>'
                    f'<div style="font-size:12px;">{t("pdf_drop_hint")}</div></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- 查询结果 | 药物相互作用检查 ----
        r2_L, r2_R = st.columns([1.25, 1])

        with r2_L:
            # 临床报告 — 白色底色容器
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown(
                f'<h3 style="color:#1f2937; font-weight:700; margin-top:0;">'
                f'{t("clinical_report")}</h3>', unsafe_allow_html=True)

            if 'res' in st.session_state and st.session_state['res']:
                st.markdown(st.session_state['res'])

                # 查看源文档 (禁止嵌套 expander)
                with st.expander(t('view_source'), expanded=False):
                    if ('docs' in st.session_state and st.session_state['docs']
                            and len(st.session_state['docs']) > 0):
                        for i, doc in enumerate(st.session_state['docs'], 1):
                            src = source_name(doc)
                            page = page_number(doc)
                            score_text = ""
                            if i <= len(st.session_state.get('search_results', [])):
                                score = st.session_state['search_results'][i - 1].display_score
                                score_text = f" · match {score:.3f}"
                            st.markdown(
                                f"**{t('source_label')} {i}** — {src} "
                                f"({t('page_label')} {page}){score_text}")
                            st.code(doc.page_content, language="text")
                    else:
                        st.info(t('no_source'))
            else:
                st.markdown(
                    f'<div style="text-align:center; padding:50px 20px; color:#9ca3af;">'
                    f'<div style="font-size:40px; margin-bottom:10px;">💊</div>'
                    f'<div style="font-size:14px; font-weight:500;">{t("awaiting")}</div>'
                    f'<div style="font-size:11px; margin-top:4px; color:#c4cad4;">'
                    f'{t("awaiting_hint")}</div></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2_R:
            st.markdown('<div class="compact-card">', unsafe_allow_html=True)
            st.markdown(
                f'<h4 style="color:#1f2937; font-weight:700; margin-top:0; margin-bottom:10px;">'
                f'{t("interaction_title")}</h4>', unsafe_allow_html=True)
            drug_list = list_demo_drugs(st.session_state.lang)
            drug_a = st.selectbox(t('drug_a_label'), ['--'] + drug_list, key="drug_a")
            drug_b = st.selectbox(t('drug_b_label'), ['--'] + drug_list, key="drug_b")
            if st.button(t('check_interaction'), use_container_width=True, key="btn_interaction"):
                if drug_a == '--' or drug_b == '--':
                    st.warning(t('interaction_select_hint'))
                elif drug_a == drug_b:
                    st.info("Same drug selected. Choose two different drugs to check interactions.")
                else:
                    result = lookup_interaction(drug_a, drug_b)
                    if result is None:
                        st.info(t('interaction_safe'))
                    elif result[0] == 'danger':
                        st.session_state.session_safety_flags += 1
                        st.error(f"{t('interaction_danger')}\n\n{result[1]}")
                    else:
                        st.session_state.session_safety_flags += 1
                        st.warning(f"{t('interaction_caution')}\n\n{result[1]}")
                    st.caption(
                        "Demo rules only — not a complete interaction database."
                        if st.session_state.lang == "en"
                        else "仅为演示规则，不是完整的药物相互作用数据库。"
                    )
            st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # 视图: AI 智能问诊 (多轮对话)
    # ============================================================
    elif menu == t('nav_triage'):
        # ---- 页面标题 ----
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            f'<h3 style="color:#0d9488; font-weight:700; margin-top:0; margin-bottom:4px;">'
            f'{t("triage_title")}</h3>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:#9ca3af; font-size:12px; margin-bottom:0;">'
            f'{t("triage_subtitle")}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- 聊天容器 ----
        st.markdown('<div class="report-card">', unsafe_allow_html=True)

        # 初始化欢迎消息
        if not st.session_state.triage_messages:
            st.session_state.triage_messages.append({
                "role": "assistant",
                "content": t('triage_welcome')
            })

        # 清除对话按钮
        clear_col1, clear_col2 = st.columns([5, 0.8])
        with clear_col2:
            if st.button(t('triage_clear'), use_container_width=True, key="btn_clear_triage"):
                st.session_state.triage_messages = []
                st.rerun()

        # 渲染历史消息
        for msg in st.session_state.triage_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 用户输入
        if user_input := st.chat_input(t('triage_placeholder'), key="triage_input"):
            # 添加用户消息
            st.session_state.triage_messages.append({
                "role": "user",
                "content": user_input
            })
            with st.chat_message("user"):
                st.markdown(user_input)

            # 调用 LLM (异常直接展示真实报错)
            with st.chat_message("assistant"):
                with st.spinner(t('triage_thinking')):
                    try:
                        triage_results = search_knowledge_base(user_input)
                        triage_docs = [item.document for item in triage_results]
                        started_at = time.perf_counter()
                        response = generate_grounded_answer(
                            triage_docs,
                            user_input,
                            history=st.session_state.triage_messages[:-1],
                        )
                        st.session_state.last_api_latency_ms = round(
                            (time.perf_counter() - started_at) * 1000
                        )
                        st.session_state.session_queries += 1
                    except Exception as e:
                        response = f"大模型调用失败: {str(e)}"
                st.markdown(response)
            st.session_state.triage_messages.append({
                "role": "assistant",
                "content": response
            })

        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # 视图: 系统看板
    # ============================================================
    elif menu == t('nav_dashboard'):
        kb_pdfs = glob.glob(str(CONFIG_KB_PATH / "*.pdf"))
        latency_text = (
            f"{st.session_state.last_api_latency_ms} ms"
            if st.session_state.last_api_latency_ms is not None
            else "—"
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="glass-card">'
                f'<div class="metric-label">{t("api_latency")}</div>'
                f'<div class="metric-value metric-teal">{latency_text}</div>'
                f'<div style="font-size:11px; color:#9ca3af; margin-top:4px;">Last request</div></div>',
                unsafe_allow_html=True)
        with m2:
            st.markdown(
                f'<div class="glass-card">'
                f'<div class="metric-label">Documents Indexed</div>'
                f'<div class="metric-value metric-blue">{len(kb_pdfs)}</div>'
                f'<div style="font-size:11px; color:#9ca3af; margin-top:4px;">Local PDF files</div></div>',
                unsafe_allow_html=True)
        with m3:
            st.markdown(
                f'<div class="glass-card">'
                f'<div class="metric-label">Session Queries</div>'
                f'<div class="metric-value metric-amber">{st.session_state.session_queries}</div>'
                f'<div style="font-size:11px; color:#9ca3af; margin-top:4px;">Current browser session</div></div>',
                unsafe_allow_html=True)
        with m4:
            api_configured = bool(os.getenv("DEEPSEEK_API_KEY"))
            api_text = "Ready" if api_configured else "Missing"
            st.markdown(
                f'<div class="glass-card">'
                f'<div class="metric-label">API Configuration</div>'
                f'<div class="metric-value metric-teal">{api_text}</div>'
                f'<div style="font-size:11px; color:#9ca3af; margin-top:4px;">Environment variable status</div></div>',
                unsafe_allow_html=True)

        col_a, col_b = st.columns([1.25, 1])
        with col_a:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4 style="color:#1f2937; font-weight:700; margin-top:0;">Safety Status</h4>',
                unsafe_allow_html=True)
            st.write(f"Interaction flags this session: **{st.session_state.session_safety_flags}**")
            st.write("Grounding mode: **fail closed**")
            st.write("Source paths: **filename only**")
            st.caption("This panel shows live session state; no synthetic production metrics are displayed.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4 style="color:#1f2937; font-weight:700; margin-top:0;">{t("recent_activity")}</h4>',
                unsafe_allow_html=True)
            logs = [("[RAG]", f"Query: {q[:70]}", "this session")
                    for q in st.session_state.search_history[:5]]
            if not logs:
                logs = [("[SYS]", "No searches in this session", "now")]
            for tag, msg, ts in logs:
                tag_cls = 'tag-teal' if tag == "[RAG]" else ('tag-blue' if tag == "[SYS]" else 'tag-amber')
                st.markdown(
                    f'<div style="padding:7px 0; border-bottom:1px solid #edf0f4; '
                    f'display:flex; justify-content:space-between; align-items:center;">'
                    f'<span><span class="tag {tag_cls}">{tag}</span> '
                    f'<span style="color:#4b5563; font-size:12px;">{msg}</span></span>'
                    f'<span style="color:#c4cad4; font-size:10px;">{ts}</span></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # 视图: 系统设置
    # ============================================================
    elif menu == t('nav_settings'):
        s1, s2 = st.columns([1, 1])
        with s1:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4 style="color:#1f2937; font-weight:700; margin-top:0;">{t("api_config")}</h4>',
                unsafe_allow_html=True)
            api_ready = bool(os.getenv("DEEPSEEK_API_KEY"))
            st.write(f"API key: **{'Configured' if api_ready else 'Missing'}**")
            st.write(f"Model: **{DEEPSEEK_MODEL}**")
            st.write("Temperature: **0.0**")
            st.write("Maximum response tokens: **1200**")
            st.caption("Secrets are read from environment variables and are never shown or edited here.")
            st.markdown('</div>', unsafe_allow_html=True)

        with s2:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4 style="color:#1f2937; font-weight:700; margin-top:0;">{t("rag_status")}</h4>',
                unsafe_allow_html=True)
            # 实际读取知识库状态
            kb_pdfs = glob.glob(str(CONFIG_KB_PATH / "*.pdf"))
            kb_status_color = "#0d9488" if kb_pdfs else "#ef4444"
            kb_status_text = "● Active" if kb_pdfs else "○ Empty"
            kb_doc_count = len(kb_pdfs)
            kb_doc_names = " | ".join([os.path.basename(f) for f in kb_pdfs]) if kb_pdfs else "No documents"
            st.markdown(
                f'<div style="padding:14px; background:#ffffff; border-radius:10px; '
                f'border:1px solid #e8ecf1; margin-bottom:12px;">'
                f'<span style="color:#374151; font-weight:600;">Status: </span>'
                f'<span style="color:{kb_status_color}; font-weight:700;">{kb_status_text}</span>'
                f'<br><span style="color:#9ca3af; font-size:11px;">'
                f'Chunk Size: 600 | Overlap: 60 | Embedding: text2vec-base-chinese</span></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div style="padding:14px; background:#ffffff; border-radius:10px; '
                f'border:1px solid #e8ecf1;">'
                f'<span style="color:#374151; font-weight:600;">Documents Indexed: </span>'
                f'<span style="color:#0d9488; font-weight:700;">{kb_doc_count}</span>'
                f'<br><span style="color:#9ca3af; font-size:11px;">{kb_doc_names}</span></div>',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="glass-card">'
            f'<h4 style="color:#1f2937; font-weight:700; margin-top:0;">{t("disclaimer_title")}</h4>'
            f'<p style="color:#6b7280; font-size:12px; line-height:1.8;">{t("disclaimer_text")}</p>'
            f'</div>',
            unsafe_allow_html=True)

    # ============================================================
    # 🌐 全局医疗免责声明 (所有页面底部)
    # ============================================================
    st.markdown(
        "<hr style='border-color:#e8ecf1; margin:2rem 0 1rem 0;'>"
        "<div style='text-align:center; color:#86909C; font-size:12px; padding:20px 0;'>"
        "⚠️ 医疗免责声明：本系统（MedVision Enterprise）基于人工智能大模型与本地知识库构建。"
        "所有生成的用药建议、查询结果及问诊回复均仅供参考，不构成任何专业医疗诊断、治疗方案或处方建议。"
        "在做出任何医疗决定或服用任何药物前，请务必遵医嘱或线下咨询专业执业医师。"
        "</div>",
        unsafe_allow_html=True)
