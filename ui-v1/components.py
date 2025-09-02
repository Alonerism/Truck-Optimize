"""
Streamlit components for the Truck Optimizer UI.
Reusable widgets and visualization functions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import httpx


def render_missions_table(missions: List[Dict], date: str):
    """Render missions table with edit/delete actions."""
    if not missions:
        st.info("No missions for this date")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(missions)
    
    # Format columns for display
    display_columns = ['id', 'location_name', 'action', 'items_display', 'priority', 'earliest', 'latest', 'notes']
    if 'items_display' not in df.columns:
        df['items_display'] = df.get('items', '')
    
    # Add action buttons
    for idx, mission in enumerate(missions):
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            st.write(f"**{mission['location_name']}** ({mission['action']}) - {mission.get('items_display', '')}")
            if mission.get('notes'):
                st.caption(mission['notes'])
        
        with col2:
            if st.button("✏️", key=f"edit_{mission['id']}", help="Edit mission"):
                st.session_state[f"editing_mission_{mission['id']}"] = True
                st.rerun()
        
        with col3:
            if st.button("🗑️", key=f"delete_{mission['id']}", help="Delete mission"):
                if delete_mission(mission['id']):
                    st.success("Mission deleted")
                    st.rerun()
                else:
                    st.error("Failed to delete mission")


def render_optimization_metrics(result: Dict):
    """Render optimization result metrics."""
    if not result:
        st.info("No optimization result available")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_drive = sum(route.get('total_drive_minutes', 0) for route in result.get('routes', []))
        st.metric("Total Drive Time", f"{total_drive:.1f} min")
    
    with col2:
        total_service = sum(route.get('total_service_minutes', 0) for route in result.get('routes', []))
        st.metric("Total Service Time", f"{total_service:.1f} min")
    
    with col3:
        total_overtime = sum(route.get('overtime_minutes', 0) for route in result.get('routes', []))
        st.metric("Overtime", f"{total_overtime:.1f} min")
    
    with col4:
        trucks_used = len([r for r in result.get('routes', []) if r.get('stops')])
        st.metric("Trucks Used", trucks_used)
    
    # Per-truck details
    st.subheader("Route Details")
    
    for route in result.get('routes', []):
        if not route.get('stops'):
            continue
            
        truck_name = route.get('truck', {}).get('name', 'Unknown Truck')
        
        with st.expander(f"🚚 {truck_name} ({len(route['stops'])} stops)"):
            # Route summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Drive Time:** {route.get('total_drive_minutes', 0):.1f} min")
            with col2:
                st.write(f"**Service Time:** {route.get('total_service_minutes', 0):.1f} min")
            with col3:
                st.write(f"**Weight:** {route.get('total_weight_lb', 0):.0f} lbs")
            
            # Stops table
            if route.get('stops'):
                stops_df = pd.DataFrame(route['stops'])
                st.dataframe(stops_df[['stop_order', 'location_name', 'estimated_arrival', 'estimated_departure']], 
                           use_container_width=True)
            
            # Google Maps button
            if route.get('google_maps_url'):
                st.link_button("🗺️ Open in Google Maps", route['google_maps_url'])


def render_truck_capacity_chart(routes: List[Dict]):
    """Render truck capacity utilization chart."""
    if not routes:
        return
    
    truck_names = []
    used_weights = []
    max_weights = []
    
    for route in routes:
        if route.get('stops'):
            truck = route.get('truck', {})
            truck_names.append(truck.get('name', 'Unknown'))
            used_weights.append(route.get('total_weight_lb', 0))
            max_weights.append(truck.get('max_weight_lb', 1))
    
    if not truck_names:
        return
    
    fig = go.Figure()
    
    # Used capacity
    fig.add_trace(go.Bar(
        name='Used Capacity',
        x=truck_names,
        y=used_weights,
        marker_color='lightblue'
    ))
    
    # Max capacity (as line)
    fig.add_trace(go.Scatter(
        name='Max Capacity',
        x=truck_names,
        y=max_weights,
        mode='markers+lines',
        marker=dict(color='red', size=8),
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Truck Capacity Utilization",
        xaxis_title="Truck",
        yaxis_title="Weight (lbs)",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_gantt_chart(routes: List[Dict]):
    """Render Gantt chart timeline of truck activities."""
    if not routes:
        return
    
    tasks = []
    
    for route in routes:
        if not route.get('stops'):
            continue
            
        truck_name = route.get('truck', {}).get('name', 'Unknown')
        
        for stop in route['stops']:
            # Parse times
            try:
                start_time = datetime.fromisoformat(stop['estimated_arrival'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(stop['estimated_departure'].replace('Z', '+00:00'))
                
                tasks.append(dict(
                    Task=truck_name,
                    Start=start_time,
                    Finish=end_time,
                    Resource=stop['location_name']
                ))
            except:
                continue
    
    if not tasks:
        return
    
    df = pd.DataFrame(tasks)
    
    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Task", 
        color="Resource",
        title="Truck Schedule Timeline"
    )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=400 + len(set(df['Task'])) * 50)
    
    st.plotly_chart(fig, use_container_width=True)


def render_trucks_table(trucks: List[Dict]):
    """Render trucks table with delete actions."""
    if not trucks:
        st.info("No trucks in fleet")
        return
    
    for truck in trucks:
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.write(f"**{truck['name']}** - {truck['max_weight_lb']:,} lbs")
            st.caption(f"Bed: {truck['bed_len_ft']}×{truck['bed_width_ft']} ft, Large capable: {truck.get('large_capable', False)}")
        
        with col2:
            if st.button("🗑️", key=f"delete_truck_{truck['id']}", help="Delete truck"):
                if delete_truck(truck['id']):
                    st.success("Truck deleted")
                    st.rerun()
                else:
                    st.error("Failed to delete truck")


def render_items_table(items: List[Dict]):
    """Render items table with delete actions."""
    if not items:
        st.info("No items in catalog")
        return
    
    for item in items:
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.write(f"**{item['name']}** ({item['category']}) - {item['weight_lb_per_unit']} lbs/unit")
            if item.get('requires_large_truck'):
                st.caption("🚛 Requires large truck")
        
        with col2:
            if st.button("🗑️", key=f"delete_item_{item['id']}", help="Delete item"):
                if delete_item(item['id']):
                    st.success("Item deleted")
                    st.rerun()
                else:
                    st.error("Failed to delete item")


def render_site_materials_table(site_materials: List[Dict]):
    """Render site materials table with edit/delete actions."""
    if not site_materials:
        st.info("No site materials tracked")
        return
    
    for material in site_materials:
        col1, col2, col3 = st.columns([4, 1, 1])
        
        with col1:
            st.write(f"**{material['site_name']}** - {material['material_name']}: {material['qty']} units")
            if material.get('notes'):
                st.caption(material['notes'])
        
        with col2:
            if st.button("✏️", key=f"edit_material_{material['id']}", help="Edit material"):
                st.session_state[f"editing_material_{material['id']}"] = True
                st.rerun()
        
        with col3:
            if st.button("🗑️", key=f"delete_material_{material['id']}", help="Delete material"):
                if delete_site_material(material['id']):
                    st.success("Material deleted")
                    st.rerun()
                else:
                    st.error("Failed to delete material")


# API helper functions
def get_api_base_url():
    """Get API base URL from config or default."""
    return st.session_state.get('api_base_url', 'http://localhost:8000')


def api_call(method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
    """Make API call with error handling."""
    try:
        base_url = get_api_base_url()
        url = f"{base_url}{endpoint}"
        
        with httpx.Client() as client:
            if method.upper() == 'GET':
                response = client.get(url)
            elif method.upper() == 'POST':
                response = client.post(url, json=data)
            elif method.upper() == 'PATCH':
                response = client.patch(url, json=data)
            elif method.upper() == 'DELETE':
                response = client.delete(url)
            else:
                st.error(f"Unsupported HTTP method: {method}")
                return None
        
        if response.status_code >= 400:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
        
        return response.json() if response.content else {}
    
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


# CRUD operations
def get_missions(date: str) -> List[Dict]:
    """Get missions for a specific date."""
    result = api_call('GET', f'/jobs?date={date}')
    return result if result else []


def add_mission(mission_data: Dict) -> bool:
    """Add a new mission."""
    result = api_call('POST', '/jobs/quick_add', mission_data)
    return result is not None


def delete_mission(mission_id: int) -> bool:
    """Delete a mission."""
    result = api_call('DELETE', f'/jobs/{mission_id}')
    return result is not None


def get_trucks() -> List[Dict]:
    """Get all trucks."""
    result = api_call('GET', '/catalog/trucks')
    return result if result else []


def add_truck(truck_data: Dict) -> bool:
    """Add a new truck."""
    result = api_call('POST', '/catalog/trucks', truck_data)
    return result is not None


def delete_truck(truck_id: int) -> bool:
    """Delete a truck."""
    result = api_call('DELETE', f'/catalog/trucks/{truck_id}')
    return result is not None


def get_items() -> List[Dict]:
    """Get all items."""
    result = api_call('GET', '/catalog/items')
    return result if result else []


def add_item(item_data: Dict) -> bool:
    """Add a new item."""
    result = api_call('POST', '/catalog/items', item_data)
    return result is not None


def delete_item(item_id: int) -> bool:
    """Delete an item."""
    result = api_call('DELETE', f'/catalog/items/{item_id}')
    return result is not None


def get_site_materials() -> List[Dict]:
    """Get all site materials."""
    result = api_call('GET', '/site_materials')
    return result if result else []


def add_site_material(material_data: Dict) -> bool:
    """Add or update site material."""
    result = api_call('POST', '/site_materials', material_data)
    return result is not None


def delete_site_material(material_id: int) -> bool:
    """Delete site material."""
    result = api_call('DELETE', f'/site_materials/{material_id}')
    return result is not None


def optimize_routes(date: str, single_truck: bool = False, seed: int = 42) -> Optional[Dict]:
    """Run route optimization."""
    params = f"?date={date}&single_truck={1 if single_truck else 0}&seed={seed}"
    result = api_call('POST', f'/optimize{params}')
    return result


def get_route_links(date: str) -> Optional[Dict]:
    """Get Google Maps links for routes."""
    result = api_call('GET', f'/links/{date}')
    return result
