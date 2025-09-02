"""
Streamlit UI for Concrete Truck Optimizer.
Fast internal interface for testing and tweaking before production UI.
"""

import streamlit as st
import sys
import os
from datetime import datetime, date, time
from typing import Optional

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.components import (
    render_missions_table, render_optimization_metrics, render_truck_capacity_chart,
    render_gantt_chart, render_trucks_table, render_items_table, render_site_materials_table,
    get_missions, add_mission, get_trucks, add_truck, get_items, add_item,
    get_site_materials, add_site_material, optimize_routes, get_route_links
)

# Page configuration
st.set_page_config(
    page_title="Truck Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = date.today()

if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None

if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = 'http://localhost:8000'

# Main title
st.title("🚚 Concrete Truck Optimizer")
st.caption("Internal testing interface for route optimization")

# Create tabs
TABS = ["📥 Daily Missions", "🚚 Add Catalog", "🛠️ Edit / Site Inventory"]
tab1, tab2, tab3 = st.tabs(TABS)

# Tab 1: Daily Missions
with tab1:
    st.header("📥 Daily Missions")
    # Sidebar form for adding missions
    with st.sidebar:
        st.subheader("Add Mission")
        
        with st.form("add_mission_form", clear_on_submit=True):
            mission_date = st.date_input("Date", value=st.session_state.selected_date)
            location_name = st.text_input("Location", placeholder="e.g., Construction Site Alpha")
            address = st.text_input("Address", placeholder="123 Main St, City", help="Used if location not known")
            action = st.selectbox("Action", ["pickup", "drop"])
            items = st.text_input("Items", placeholder="rebar:5; big drill:1", 
                                help="semicolon-separated item:qty")
            priority = st.number_input("Priority", min_value=0, max_value=9, value=1, step=1)
            
            col1, col2 = st.columns(2)
            with col1:
                earliest = st.time_input("Earliest", value=None)
            with col2:
                latest = st.time_input("Latest", value=None)
            
            notes = st.text_area("Notes", placeholder="Optional notes...")
            
            # Form buttons
            col1, col2 = st.columns(2)
            with col1:
                submit_mission = st.form_submit_button("Add Mission", type="primary")
            with col2:
                optimize_today = st.form_submit_button("Optimize Today", type="secondary")
        
        # Handle form submissions
        if submit_mission:
            mission_data = {
                "date": mission_date.isoformat(),
                "location_name": location_name,
                "address": address if address else None,
                "action": action,
                "items": items,
                "priority": priority,
                "earliest": earliest.isoformat() if earliest else None,
                "latest": latest.isoformat() if latest else None,
                "notes": notes if notes else None
            }
            
            if add_mission(mission_data):
                st.success("Mission added successfully!")
                st.rerun()
            else:
                st.error("Failed to add mission")
        
        if optimize_today:
            with st.spinner("Optimizing routes..."):
                result = optimize_routes(st.session_state.selected_date.isoformat())
                if result:
                    st.session_state.optimization_result = result
                    st.success("Optimization completed!")
                    st.rerun()
                else:
                    st.error("Optimization failed")
    
    # Main panel
    st.subheader("Missions for Selected Date")
    
    # Date picker for viewing missions
    col1, col2 = st.columns([2, 1])
    with col1:
        missions_date = st.date_input("View missions for date:", value=st.session_state.selected_date)
        if missions_date != st.session_state.selected_date:
            st.session_state.selected_date = missions_date
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", help="Refresh missions list"):
            st.rerun()
    
    # Missions table
    missions = get_missions(missions_date.isoformat())
    render_missions_table(missions, missions_date.isoformat())
    
    # Optimization results
    if st.session_state.optimization_result:
        st.subheader("Optimization Result (read-only)")
        render_optimization_metrics(st.session_state.optimization_result)
        
        # Charts
        routes = st.session_state.optimization_result.get('routes', [])
        if routes:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Capacity Utilization")
                render_truck_capacity_chart(routes)
            with col2:
                st.subheader("Schedule Timeline")
                render_gantt_chart(routes)
        
        # Download options
        st.subheader("Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Missions CSV"):
                # TODO: Implement CSV download
                st.info("CSV download not yet implemented")
        
        with col2:
            if st.button("📥 Download Routes JSON"):
                # TODO: Implement JSON download
                st.info("JSON download not yet implemented")
        
        with col3:
            if st.button("📥 Download Google Maps Links"):
                links = get_route_links(missions_date.isoformat())
                if links:
                    # TODO: Format and download links
                    st.info("Links download not yet implemented")

# Tab 2: Add Catalog
with tab2:
    st.header("🚚 Add Catalog")
    # Sidebar form for adding trucks/items
    with st.sidebar:
        st.subheader("Add Truck / Machine / Material")
        
        # Truck section
        st.subheader("Add Truck")
        with st.form("add_truck_form", clear_on_submit=True):
            truck_name = st.text_input("Truck Name", placeholder="e.g., Big Truck C")
            max_weight_lb = st.number_input("Max Weight (lbs)", min_value=0, value=5000)
            bed_len_ft = st.number_input("Bed Length (ft)", min_value=0.0, value=8.0, step=0.5)
            bed_width_ft = st.number_input("Bed Width (ft)", min_value=0.0, value=5.5, step=0.5)
            height_limit_ft = st.number_input("Height Limit (ft)", min_value=0.0, value=8.0, step=0.5)
            large_capable = st.checkbox("Large Capable")
            
            if st.form_submit_button("Add Truck", type="primary"):
                truck_data = {
                    "name": truck_name,
                    "max_weight_lb": max_weight_lb,
                    "bed_len_ft": bed_len_ft,
                    "bed_width_ft": bed_width_ft,
                    "height_limit_ft": height_limit_ft,
                    "large_capable": large_capable
                }
                
                if add_truck(truck_data):
                    st.success("Truck added successfully!")
                    st.rerun()
                else:
                    st.error("Failed to add truck")
        
        st.divider()
        
        # Item section
        st.subheader("Add Item")
        with st.form("add_item_form", clear_on_submit=True):
            item_name = st.text_input("Item Name", placeholder="e.g., concrete block")
            category = st.selectbox("Category", ["machine", "equipment", "material", "fuel"])
            weight_lb_per_unit = st.number_input("Weight per Unit (lbs)", min_value=0.0, value=50.0)
            dims_lwh_ft = st.text_input("Dimensions (L,W,H)", placeholder="6,3,4 or leave blank")
            requires_large_truck = st.checkbox("Requires Large Truck")
            
            if st.form_submit_button("Add Item", type="primary"):
                item_data = {
                    "name": item_name,
                    "category": category,
                    "weight_lb_per_unit": weight_lb_per_unit,
                    "requires_large_truck": requires_large_truck
                }
                
                # Parse dimensions if provided
                if dims_lwh_ft.strip():
                    try:
                        dims = [float(x.strip()) for x in dims_lwh_ft.split(',')]
                        if len(dims) == 3:
                            item_data["dims_lwh_ft"] = dims
                    except:
                        st.warning("Invalid dimensions format, ignoring")
                
                if add_item(item_data):
                    st.success("Item added successfully!")
                    st.rerun()
                else:
                    st.error("Failed to add item")
    
    # Main panel
    st.subheader("Fleet")
    trucks = get_trucks()
    render_trucks_table(trucks)
    
    st.subheader("Item Catalog")
    items = get_items()
    render_items_table(items)

# Tab 3: Edit / Site Inventory
with tab3:
    st.header("🛠️ Edit / Site Inventory")
    # Sidebar form
    with st.sidebar:
        st.subheader("Site Materials")
        
        with st.form("add_site_material_form", clear_on_submit=True):
            # Get locations and items for dropdowns
            locations = []  # TODO: Get from API
            materials = [item for item in get_items() if item.get('category') == 'material']
            
            site_selector = st.selectbox("Job Site", 
                                       options=[loc.get('name', 'Unknown') for loc in locations] if locations else ["No locations available"])
            material_name = st.selectbox("Material", 
                                       options=[mat.get('name', 'Unknown') for mat in materials] if materials else ["No materials available"])
            qty = st.number_input("Quantity", min_value=0, value=1, step=1)
            material_notes = st.text_area("Notes", placeholder="Optional notes about this material...")
            
            if st.form_submit_button("Add/Update Material at Site", type="primary"):
                material_data = {
                    "site_name": site_selector,
                    "material_name": material_name,
                    "qty": qty,
                    "notes": material_notes if material_notes else None
                }
                
                if add_site_material(material_data):
                    st.success("Site material updated!")
                    st.rerun()
                else:
                    st.error("Failed to update site material")
    
    # Main panel
    st.subheader("Current Site Inventories")
    site_materials = get_site_materials()
    render_site_materials_table(site_materials)

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🚚 Truck Optimizer v0.1.0")

with col2:
    # API status indicator
    try:
        from ui.components import api_call
        health = api_call('GET', '/health')
        if health:
            st.caption("🟢 API Connected")
        else:
            st.caption("🔴 API Disconnected")
    except:
        st.caption("🔴 API Disconnected")

with col3:
    # Settings
    with st.popover("⚙️ Settings"):
        new_api_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
        if st.button("Update API URL"):
            st.session_state.api_base_url = new_api_url
            st.success("API URL updated!")
