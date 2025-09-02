import streamlit as st
import os
from dotenv import load_dotenv

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
page = query_params.get("page", "user")

st.sidebar.title("Navigation")
st.sidebar.markdown(f"[User Page](?page=user)")
st.sidebar.markdown(f"[Admin Page](?page=admin)")

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
    else:
        import pandas as pd
        df = pd.DataFrame(complaints, columns=['ID', 'Complaint Text', 'Sentiment', 'Priority', 'Resolution', 'Timestamp'])
        
        st.dataframe(df, use_container_width=True)

if page == "user":
    user_page()
elif page == "admin":
    admin_page()
else:
    st.error("Page not found. Please use the navigation links.")