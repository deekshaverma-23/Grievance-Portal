import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import json
import urllib.parse

from multi_agent import app
from database import init_db, get_all_complaints

load_dotenv()

init_db()

st.set_page_config(
    page_title="Grievance Redressal System",
    layout="wide",
    initial_sidebar_state="expanded"
)

query_params = st.query_params
page = query_params.get("page", ["user"])
if isinstance(page, list):
    page = page[0]

def slugify(category):
    return urllib.parse.quote(
        category.lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace(" ", "_")
    )

st.sidebar.title("Navigation")
st.sidebar.markdown(f"[User Page](?page=user)")
st.sidebar.title("Admin Dashboard")

def get_categories():
    complaints = get_all_complaints()
    df = pd.DataFrame(
        complaints,
        columns=[
            'ID','Complaint Text','Sentiment','Severity',
            'Credibility','Category','Priority','Resolution','Timestamp'
        ]
    )
    df['Category'] = df['Category'].astype(str)
    df['Category'] = df['Category'].apply(lambda c: c.split('\n')[0].strip())

    DEFAULT = [
        "Healthcare", "Water", "Roads", "Electricity",
        "Sanitation", "Law & Order"
    ]

    dynamic = sorted(set(df['Category'].dropna()) - set(DEFAULT))

    return DEFAULT + dynamic



for cat in get_categories():
    st.sidebar.markdown(f"[{cat}](?page={slugify(cat)})")



def format_resolution(res):
    try:
        res = res.replace("```json", "").replace("```", "").strip()
        data = json.loads(res)
        return f"""
        Summary: {data['summary']}

        Immediate Actions:
        - {'\n- '.join(data['immediate_actions'])}

        Department: {data['responsible_department']}
        SLA: {data['sla_hours']} hours
        """
    except:
        return res


def user_page():
    st.title("Citizen Complaint Submission ")
    st.markdown("Submit your complaint below and get an instant AI-generated resolution plan.")

    with st.form(key='grievance_form'):
        complaint_text = st.text_area("Enter your complaint:", height=200, help="e.g., The streetlights in my locality are not working.")
        submit_button = st.form_submit_button(label='Submit Complaint')

    if submit_button and complaint_text:
        with st.spinner("Processing your complaint... Please wait."):
            try:
                final_state = app.invoke({"complaint_text": complaint_text})
                
                st.success("Your complaint has been successfully submitted and processed!")
                st.info("A resolution plan and priority have been generated and forwarded to the concerned department.")
                st.write(f"**Your Complaint:** {complaint_text}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

def admin_page():
    st.title("Admin Dashboard")
    st.markdown("View and manage all submitted complaints.")

    st.header("All Submitted Complaints")
    
    complaints = get_all_complaints()

    if not complaints:
        st.info("No complaints have been submitted yet.")
        return
    else:
        df = pd.DataFrame(
        complaints,
        columns=[
            'ID',
            'Complaint Text',
            'Sentiment',
            'Severity',
            'Credibility',
            'Priority',
            'Resolution',
            'Timestamp'
        ]
    )


    df["Resolution"] = df["Resolution"].apply(format_resolution)
    df = df[['ID', 'Complaint Text', 'Sentiment', 'Severity', 'Credibility', 'Priority', 'Resolution', 'Timestamp']]

    st.dataframe(df, use_container_width=True, hide_index=True)


def compute_urgency(row):
    severity_map = {"critical":4, "high":3, "medium":2, "low":1}
    priority_map = {"critical":4, "high":3, "medium":2, "low":1}
    sentiment_map = {"negative":3, "neutral":2, "positive":1}
    cred_map = {"factual":3, "mildly exaggerated":2, "highly exaggerated":1}

    s = severity_map.get(row['Severity'], 1)
    p = priority_map.get(row['Priority'], 1)
    se = sentiment_map.get(row['Sentiment'], 1)
    c = cred_map.get(row['Credibility'], 1)

    urgency = 0.45*s + 0.30*p + 0.15*se + 0.10*c
    return round(urgency, 2)

def render_category_page(category):
    st.title(f"{category} Complaints")

    complaints = get_all_complaints()
    df = pd.DataFrame(
    complaints,
    columns=[
        'ID','Complaint Text','Sentiment','Severity',
        'Credibility','Category','Priority','Resolution','Timestamp'
    ]
    )

    df = df[df['Category'].str.lower() == category.lower()]
    df = df.drop(columns=['Category'])

    if df.empty:
        st.info("No complaints submitted in this category yet.")
        return

    df['Urgency'] = df.apply(compute_urgency, axis=1)
    df = df.sort_values(by="Urgency", ascending=False)
    df['Resolution'] = df['Resolution'].apply(format_resolution)
    df = df[['Urgency', 'ID', 'Complaint Text', 'Sentiment', 'Severity', 'Credibility', 'Priority', 'Resolution', 'Timestamp']]
    st.dataframe(df, use_container_width=True, hide_index=True)

valid_slugs = [slugify(c) for c in get_categories()]

if page == "user":
    user_page()

elif page == "admin":
    admin_page()

elif page in valid_slugs:
    cat = next(c for c in get_categories() if slugify(c) == page)
    render_category_page(cat)

else:
    st.error("Page not found. Please use the navigation links.")
