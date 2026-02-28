import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Product Recommender",
    page_icon="🛒",
    layout="centered"
)

# Title and description
st.title("🛒 Grocery Product Recommender")
st.markdown("""
This app recommends products based on your shopping cart using **Association Rules Mining**.
Add items to your cart and get personalized recommendations!
""")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Technology Stack:**
    - Machine Learning: Association Rules (Apriori)
    - Backend: Flask REST API
    - Frontend: Streamlit
    - Deployment: Streamlit Community Cloud
    
    **Student:** Amin (ID: 147473)  
    **Course:** BBT 4206 - Business Intelligence II  
    **University:** Strathmore University
    """)

# Initialize session state for cart
if 'cart' not in st.session_state:
    st.session_state.cart = []

# Common grocery items for suggestions
COMMON_ITEMS = [
    "whole milk", "yogurt", "other vegetables", "rolls/buns",
    "tropical fruit", "bottled water", "soda", "root vegetables",
    "pork", "sausage", "citrus fruit", "beef", "frankfurter",
    "chicken", "butter", "fruit/vegetable juice", "packaged fruit/vegetables",
    "chocolate", "specialty bar", "butter milk", "pastry", "canned beer",
    "newspapers", "shopping bags", "whipped/sour cream", "brown bread",
    "domestic eggs", "margarine", "ham", "coffee", "curd", "white bread"
]

# Main content
st.header("Build Your Shopping Cart")

# Two columns for input
col1, col2 = st.columns([3, 1])

with col1:
    # Text input with autocomplete
    item_input = st.selectbox(
        "Select item to add:",
        [""] + COMMON_ITEMS,
        key="item_selector"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    add_button = st.button("➕ Add", type="primary", use_container_width=True)

# Add item to cart
if add_button and item_input:
    if item_input not in st.session_state.cart:
        st.session_state.cart.append(item_input)
        st.success(f"Added: {item_input}")
    else:
        st.warning(f"{item_input} is already in your cart!")

# Display current cart
if st.session_state.cart:
    st.subheader("🛒 Your Cart:")
    
    # Display cart items with remove buttons
    cart_cols = st.columns(3)
    for idx, item in enumerate(st.session_state.cart):
        col_idx = idx % 3
        with cart_cols[col_idx]:
            if st.button(f"❌ {item}", key=f"remove_{idx}", use_container_width=True):
                st.session_state.cart.remove(item)
                st.rerun()
    
    # Clear cart button
    if st.button("🗑️ Clear Cart", type="secondary"):
        st.session_state.cart = []
        st.rerun()
    
    st.divider()
    
    # Get recommendations button
    if st.button("✨ Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Finding best recommendations..."):
            try:
                # Call your API
                # NOTE: Change this URL to your actual API endpoint
                # For local testing: http://127.0.0.1:5000
                # For production: your deployed API URL
                API_URL = "http://127.0.0.1:5000/api/v1/models/recommender/predictions"
                
                response = requests.post(
                    API_URL,
                    json={"cart": st.session_state.cart},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    recommendations = result.get('recommendations', [])
                    
                    if recommendations:
                        st.success("###  Recommended Products:")
                        
                        # Display recommendations in a nice format
                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"**{i}.** {rec}")
                        
                        st.balloons()
                    else:
                        st.info("No recommendations available for these items.")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"""
                 Could not connect to API.
                
                **For demo purposes**, here are some sample recommendations based on common patterns:
                """)
                
                # Fallback dummy recommendations for demo
                dummy_recs = []
                if 'whole milk' in st.session_state.cart:
                    dummy_recs = ['other vegetables', 'rolls/buns', 'yogurt']
                elif 'yogurt' in st.session_state.cart:
                    dummy_recs = ['whole milk', 'tropical fruit']
                else:
                    dummy_recs = ['whole milk', 'other vegetables']
                
                # Remove items already in cart
                dummy_recs = [item for item in dummy_recs if item not in st.session_state.cart]
                
                st.success("### Sample Recommendations:")
                for i, rec in enumerate(dummy_recs[:5], 1):
                    st.markdown(f"**{i}.** {rec}")
                
                st.info(" This is a fallback demo. In production, this connects to the Flask API.")

else:
    st.info(" Add items to your cart to get started!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built using Streamlit | BBT 4206 - Business Intelligence II Lab
</div>
""", unsafe_allow_html=True)