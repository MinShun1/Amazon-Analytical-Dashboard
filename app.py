"""E-Commerce Analytics Dashboard — entry point.

Run from the project root:
    streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ROOT / "pages"

navigation = st.navigation(
    {
        "Overview": [
            st.Page(PAGES / "1_Business_Overview.py", title="Business Overview",
                    icon=":material/dashboard:", default=True),
        ],
        "Domains": [
            st.Page(PAGES / "2_Amazon_Analytics.py", title="Amazon Analytics",
                    icon=":material/shopping_cart:"),
            st.Page(PAGES / "3_International_Analytics.py", title="International Analytics",
                    icon=":material/public:"),
            st.Page(PAGES / "4_Product_Analytics.py", title="Product Analytics",
                    icon=":material/inventory_2:"),
            st.Page(PAGES / "5_Financial_Operational_Analytics.py",
                    title="Financial & Operational", icon=":material/account_balance:"),
        ],
    }
)

with st.sidebar:
    st.caption("E-Commerce Analytics Dashboard · data: Kaggle *Unlock Profits with "
               "E-Commerce Sales Data* · analysis reproduced from the project notebook.")

navigation.run()
