import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json


st.set_page_config(page_title="Washington State EV Dashboard", layout="wide")

# Approximate county centroids (latitude, longitude) for Washington State
county_centroids = {
    "King": (47.6062, -122.3321),  # Seattle
    "Snohomish": (47.9783, -122.2021),  # Everett
    "Thurston": (47.0379, -122.9007),  # Olympia
    "Pierce": (47.2529, -122.4443),  # Tacoma
    "Kitsap": (47.6732, -122.6970),  # Bremerton
    "Spokane": (47.6588, -117.4260),  # Spokane
    "Clark": (45.6278, -122.6704),  # Vancouver
    "Whatcom": (48.7519, -122.4787),  # Bellingham
    "Benton": (46.2087, -119.1208),  # Kennewick
    "Skagit": (48.4242, -122.3342),  # Mount Vernon
    "Yakima": (46.6021, -120.5059),  # Yakima
    "Franklin": (46.2804, -119.2752),  # Pasco
    "Island": (48.1517, -122.6706),  # Oak Harbor
    "Lewis": (46.5775, -122.3928),  # Chehalis
    "Cowlitz": (46.1458, -122.9332),  # Kelso
}

# Minimal GeoJSON for Washington State counties (subset for key counties)
wa_counties_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "King"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.5, 47.8], [-122.0, 47.8], [-122.0, 47.3], [-122.5, 47.3], [-122.5, 47.8]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Snohomish"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.4, 48.3], [-121.9, 48.3], [-121.9, 47.8], [-122.4, 47.8], [-122.4, 48.3]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Thurston"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-123.0, 47.2], [-122.5, 47.2], [-122.5, 46.7], [-123.0, 46.7], [-123.0, 47.2]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Pierce"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.8, 47.4], [-122.3, 47.4], [-122.3, 46.9], [-122.8, 46.9], [-122.8, 47.4]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Kitsap"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.9, 47.8], [-122.4, 47.8], [-122.4, 47.3], [-122.9, 47.3], [-122.9, 47.8]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Spokane"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-117.8, 47.9], [-117.3, 47.9], [-117.3, 47.4], [-117.8, 47.4], [-117.8, 47.9]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Clark"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.8, 46.0], [-122.3, 46.0], [-122.3, 45.5], [-122.8, 45.5], [-122.8, 46.0]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Whatcom"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-123.0, 49.0], [-122.5, 49.0], [-122.5, 48.5], [-123.0, 48.5], [-123.0, 49.0]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Benton"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-119.5, 46.5], [-119.0, 46.5], [-119.0, 46.0], [-119.5, 46.0], [-119.5, 46.5]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Skagit"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.7, 48.7], [-122.2, 48.7], [-122.2, 48.2], [-122.7, 48.2], [-122.7, 48.7]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Yakima"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-121.0, 46.8], [-120.5, 46.8], [-120.5, 46.3], [-121.0, 46.3], [-121.0, 46.8]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Franklin"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-119.5, 46.8], [-119.0, 46.8], [-119.0, 46.3], [-119.5, 46.3], [-119.5, 46.8]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Island"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.9, 48.4], [-122.4, 48.4], [-122.4, 47.9], [-122.9, 47.9], [-122.9, 48.4]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Lewis"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-123.0, 46.8], [-122.5, 46.8], [-122.5, 46.3], [-123.0, 46.3], [-123.0, 46.8]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Cowlitz"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-123.0, 46.4], [-122.5, 46.4], [-122.5, 45.9], [-123.0, 45.9], [-123.0, 46.4]]]
            }
        },
    ]
}

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("D:/JU Internship/Project/Electric_Vehicle_Population_Data.csv",encoding="ISO-8859-1")
    # Clean data
    df["Electric Range"] = df["Electric Range"].fillna(0)
    df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"] = df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].fillna("Unknown")
    df["Electric Utility"] = df["Electric Utility"].fillna("Unknown")
    return df

df = load_data()

# Split Electric Utility into individual providers
def split_utilities(df):
    utility_pairs = []
    for idx, row in df.iterrows():
        utilities = row["Electric Utility"].split("||") if row["Electric Utility"] != "Unknown" else ["Unknown"]
        for utility in utilities:
            utility_pairs.append((idx, utility.strip()))
    utility_df = pd.DataFrame(utility_pairs, columns=["index", "Electric Utility"])
    expanded_df = df.drop("Electric Utility", axis=1).merge(utility_df, left_index=True, right_on="index")
    return expanded_df

utility_expanded_df = split_utilities(df)

# Initialize session state for filters
if "county_filter" not in st.session_state:
    st.session_state.county_filter = ["King", "Snohomish", "Thurston"]
if "year_filter" not in st.session_state:
    st.session_state.year_filter = (2015, 2024)
if "make_filter" not in st.session_state:
    st.session_state.make_filter = ["TESLA", "NISSAN", "CHEVROLET"]
if "ev_type_filter" not in st.session_state:
    st.session_state.ev_type_filter = list(df["Electric Vehicle Type"].unique())
if "cafv_filter" not in st.session_state:
    st.session_state.cafv_filter = list(df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].unique())
if "make_dist_filter" not in st.session_state:
    st.session_state.make_dist_filter = ["TESLA"]
if "utility_dist_filter" not in st.session_state:
    st.session_state.utility_dist_filter = ["PUGET SOUND ENERGY INC"]
if "search_field" not in st.session_state:
    st.session_state.search_field = "VIN (1-10)"
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# Sidebar filters
st.sidebar.header("Filters")

# Reset button
def reset_filters():
    st.session_state.county_filter = ["King", "Snohomish", "Thurston"]
    st.session_state.year_filter = (2015, 2024)
    st.session_state.make_filter = ["TESLA", "NISSAN", "CHEVROLET"]
    st.session_state.ev_type_filter = list(df["Electric Vehicle Type"].unique())
    st.session_state.cafv_filter = list(df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].unique())
    st.session_state.make_dist_filter = ["TESLA"]
    st.session_state.utility_dist_filter = ["PUGET SOUND ENERGY INC"]
    st.session_state.search_field = "VIN (1-10)"
    st.session_state.search_query = ""

st.sidebar.button("Reset Filters", on_click=reset_filters)

# General filters
county_filter = st.sidebar.multiselect("Select County", options=sorted(df["County"].unique()), default=st.session_state.county_filter, key="county_filter")
year_filter = st.sidebar.slider("Model Year", min_value=int(df["Model Year"].min()), max_value=int(df["Model Year"].max()), value=st.session_state.year_filter, key="year_filter")
make_filter = st.sidebar.multiselect("Select Make (General Filter)", options=sorted(df["Make"].unique()), default=st.session_state.make_filter, key="make_filter")
ev_type_filter = st.sidebar.multiselect("Select EV Type", options=sorted(df["Electric Vehicle Type"].unique()), default=st.session_state.ev_type_filter, key="ev_type_filter")
cafv_filter = st.sidebar.multiselect("Select CAFV Eligibility", options=sorted(df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].unique()), default=st.session_state.cafv_filter, key="cafv_filter")
make_dist_filter = st.sidebar.multiselect("Select Make for County Distribution", options=sorted(df["Make"].unique()), default=st.session_state.make_dist_filter, key="make_dist_filter")
utility_dist_filter = st.sidebar.multiselect("Select Electric Utility for County Distribution", options=sorted(utility_expanded_df["Electric Utility"].unique()), default=st.session_state.utility_dist_filter, key="utility_dist_filter")

# Search bar in sidebar
st.sidebar.subheader("Search Records")
search_field = st.sidebar.selectbox("Search by", options=["VIN (1-10)", "Model", "City"], index=["VIN (1-10)", "Model", "City"].index(st.session_state.search_field), key="search_field")
search_query = st.sidebar.text_input("Enter search term", value=st.session_state.search_query, key="search_query")

# Apply general filters to utility_expanded_df
filtered_utility_df = utility_expanded_df[
    (utility_expanded_df["County"].isin(county_filter) if county_filter else True) &
    (utility_expanded_df["Model Year"].between(year_filter[0], year_filter[1])) &
    (utility_expanded_df["Make"].isin(make_filter) if make_filter else True) &
    (utility_expanded_df["Electric Vehicle Type"].isin(ev_type_filter) if ev_type_filter else True) &
    (utility_expanded_df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].isin(cafv_filter) if cafv_filter else True)
]

# Apply general filters to original df
filtered_df = df[
    (df["County"].isin(county_filter) if county_filter else True) &
    (df["Model Year"].between(year_filter[0], year_filter[1])) &
    (df["Make"].isin(make_filter) if make_filter else True) &
    (df["Electric Vehicle Type"].isin(ev_type_filter) if ev_type_filter else True) &
    (df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].isin(cafv_filter) if cafv_filter else True)
]

# Apply search filter for detailed data table
if search_query:
    filtered_search_df = filtered_df[filtered_df[search_field].str.contains(search_query, case=False, na=False)]
else:
    filtered_search_df = filtered_df

# Create new DataFrame for count of each Make
make_counts_df = filtered_df["Make"].value_counts().reset_index()
make_counts_df.columns = ["Make", "Count"]

# Create DataFrame for county distribution of selected Make(s)
county_make_dist_df = filtered_df[filtered_df["Make"].isin(make_dist_filter)][["County", "Make"]].groupby(["County", "Make"]).size().reset_index(name="Count")

# Create DataFrame for count of each Electric Utility
utility_counts_df = filtered_utility_df["Electric Utility"].value_counts().reset_index()
utility_counts_df.columns = ["Electric Utility", "Count"]

# Group small providers into "Other" for pie chart
top_n_pie = 7
top_utilities_pie = utility_counts_df.head(top_n_pie)["Electric Utility"].tolist()
utility_counts_df_pie = utility_counts_df.copy()
utility_counts_df_pie["Electric Utility"] = utility_counts_df_pie["Electric Utility"].apply(lambda x: x if x in top_utilities_pie else "Other")
utility_counts_df_pie = utility_counts_df_pie.groupby("Electric Utility")["Count"].sum().reset_index()

# Create DataFrame for top 10 utilities (for bar chart)
top_n_bar = 10
utility_counts_df_bar = utility_counts_df.head(top_n_bar)

# Create DataFrame for county distribution of selected Electric Utility(s)
county_utility_dist_df = filtered_utility_df[filtered_utility_df["Electric Utility"].isin(utility_dist_filter)][["County", "Electric Utility"]].groupby(["County", "Electric Utility"]).size().reset_index(name="Count")

# Create DataFrame for EV type breakdown per Electric Utility
utility_ev_type_df = filtered_utility_df.groupby(["Electric Utility", "Electric Vehicle Type"]).size().reset_index(name="Count")
top_n_ev_type = 7
top_utilities_ev_type = utility_counts_df.head(top_n_ev_type)["Electric Utility"].tolist()
utility_ev_type_df = utility_ev_type_df[utility_ev_type_df["Electric Utility"].isin(top_utilities_ev_type)]

# Create DataFrame for average electric range by Electric Vehicle Type
ev_type_range_df = filtered_df[filtered_df["Electric Range"] > 0][["Electric Vehicle Type", "Electric Range"]].groupby("Electric Vehicle Type")["Electric Range"].mean().reset_index()
ev_type_range_df.columns = ["Electric Vehicle Type", "Average Range"]

# Create DataFrame for most popular model by Make
model_counts_df = filtered_df.groupby(["Make", "Model"]).size().reset_index(name="Count")
popular_models_df = model_counts_df.loc[model_counts_df.groupby("Make")["Count"].idxmax()].reset_index(drop=True)
popular_models_df = popular_models_df[["Make", "Model", "Count"]].sort_values("Make")
popular_models_df.columns = ["Make", "Most Popular Model", "Count"]

# Create DataFrame for bubble map (county-level EV counts with centroids)
county_counts = filtered_df["County"].value_counts().reset_index()
county_counts.columns = ["County", "EV Count"]
county_bubble_df = county_counts[county_counts["County"].isin(county_centroids.keys())].copy()
county_bubble_df["Latitude"] = county_bubble_df["County"].apply(lambda x: county_centroids[x][0])
county_bubble_df["Longitude"] = county_bubble_df["County"].apply(lambda x: county_centroids[x][1])
county_bubble_df["GeoJSON_ID"] = county_bubble_df["County"]
# Map counties to numeric IDs for color
county_bubble_df["County_ID"] = pd.factorize(county_bubble_df["County"])[0]

# Main title and KPIs
st.title("Washington Electric Vehicle Population Data Dashboard")
# st.markdown("Explore EV adoption trends, geographic distribution, and vehicle characteristics in Washington State.")

st.markdown("---")
st.subheader("Quick Insights")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total EVs registered", len(filtered_df))
col2.metric("BEV %", f"{(filtered_df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)').mean() * 100:.1f}%")
col3.metric("Avg Electric Range in miles", f"{filtered_df[filtered_df['Electric Range'] > 0]['Electric Range'].mean():.1f}" if filtered_df[filtered_df['Electric Range'] > 0].shape[0] > 0 else "N/A")
col4.metric("Top County using EV", filtered_df["County"].mode()[0] if not filtered_df.empty else "N/A")

# Section: Geographic Distribution
st.markdown("---")
st.subheader("Geographic Distribution of EVs")
col_map, col_bar = st.columns([2, 2])

# Bubble Map with County Outlines
with col_map:
    if not county_bubble_df.empty:
        fig = go.Figure()

        # Add county outlines (transparent choropleth)
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=wa_counties_geojson,
                locations=county_bubble_df["GeoJSON_ID"],
                z=[0] * len(county_bubble_df),  # Dummy z for no fill
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],  # Transparent
                marker_line_width=1.5,
                marker_line_color="rgba(0,0,0,0)",
                showscale=False,
                hoverinfo="none",
                featureidkey="properties.name"  # Match GeoJSON property
            )
        )

        # Add bubble map
        fig.add_trace(
            go.Scattermapbox(
                lat=county_bubble_df["Latitude"],
                lon=county_bubble_df["Longitude"],
                mode="markers",
                marker=dict(
                    size=county_bubble_df["EV Count"] / county_bubble_df["EV Count"].max() * 200,  # Larger bubbles
                    sizemode="area",
                    sizeref=0.1,
                    sizemin=5,
                    color=county_bubble_df["County_ID"],  # Numeric IDs
                    colorscale="jet",
                    showscale=True,
                    # line=dict(color="red", width=1)  # Red border for bubbles
                ),
                text=county_bubble_df["County"] + "<br>EV Count: " + county_bubble_df["EV Count"].astype(str),
                hoverinfo="text"
            )
        )

        fig.update_layout(
            title="County wise EV Distribution using Bubble Map showing physical proximity of clusters",
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=47.5, lon=-120.5),
                zoom=6
            ),
            height=400,
            margin={"r":0,"t":50,"l":0,"b":0},
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No geographic data available for the selected counties.")

# Top Counties Bar Chart
with col_bar:
    county_counts_bar = filtered_df["County"].value_counts().head(5)
    fig_bar = px.bar(x=county_counts_bar.index, y=county_counts_bar.values, labels={"x": "County", "y": "EV Count"}, title="Top 5 Counties by EV Count")
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

# Section: Market Share and Adoption Trends
st.markdown("---")
st.subheader("Market Share and Adoption Trends")
col_pie, col_line = st.columns([2, 2])

# Pie Chart for Make Counts
with col_pie:
    make_counts_df = filtered_df["Make"].value_counts().head(10)
    fig_pie = px.pie(values=make_counts_df.values, names=make_counts_df.index, title="Market Share by Manufacturer")
    fig_pie.update_layout(height=600)
    
    st.plotly_chart(fig_pie, use_container_width=True)

# Registrations by Model Year
with col_line:
    year_counts = filtered_df.groupby(["Model Year", "Electric Vehicle Type"]).size().unstack().fillna(0)
    fig_line = go.Figure()
    for ev_type in year_counts.columns:
        fig_line.add_trace(go.Scatter(x=year_counts.index, y=year_counts[ev_type], mode="lines+markers", name=ev_type))
    fig_line.update_layout(title="EV Registrations by Model Year", xaxis_title="Model Year", yaxis_title="EV Count", height=600)
    fig_line.update_layout(legend=dict(
    yanchor="bottom",
    y=-10.00,
    xanchor="left",
    x=0.01
    ))
    st.plotly_chart(fig_line, use_container_width=True)

# Section: Electric Range and CAFV Eligibility
st.markdown("---")
st.subheader("Electric Range, Average Range and CAFV Eligibility")
col_hist, col_range_bar, col_donut = st.columns([2, 2, 2])

# Electric Range Histogram
with col_hist:
    range_data = filtered_df[filtered_df["Electric Range"] > 0]
    fig_hist = px.histogram(range_data, x="Electric Range", nbins=30, title="Distribution of Electric Range", labels={"Electric Range": "Range (miles)"})
    fig_hist.update_layout(height=600)
    st.plotly_chart(fig_hist, use_container_width=True)

# Bar Chart for Average Range by EV Type
with col_range_bar:
    if not ev_type_range_df.empty:
        fig_range_bar = px.bar(
            ev_type_range_df,
            x="Electric Vehicle Type",
            y="Average Range",
            title="Average Range by EV Type",
            labels={"Average Range": "Average Range (miles)"},
            text="Average Range"
        )
        fig_range_bar.update_traces(texttemplate="%{text:.1f}", textposition="auto")
        fig_range_bar.update_layout(height=600, xaxis={"tickangle": 90})
        st.plotly_chart(fig_range_bar, use_container_width=True)
    else:
        st.warning("No valid range data available for the selected filters.")

# CAFV Eligibility Donut Chart
with col_donut:
    cafv_counts = filtered_df["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].value_counts()
    fig_donut = px.pie(values=cafv_counts.values, names=cafv_counts.index, title="CAFV Eligibility Breakdown", hole=0.4)
    fig_donut.update_layout(height=600,showlegend=True)
    fig_donut.update_layout(legend=dict(
    yanchor="middle",
    y=-10.00,
    xanchor="left",
    x=0.01
))
    # fig_donut.update_traces(textposition='outside',textinfo='percent+label')
    st.plotly_chart(fig_donut, use_container_width=True)

# Section: Electric Utility Insights
st.markdown("---")
st.subheader("Electric Utility Insights")


# Horizontal Bar Chart for Utility Counts
fig_utility_bar = px.bar(
    utility_counts_df_bar,
    y="Electric Utility",
    x="Count",
    orientation="h",
    title="Top 10 Electric Utility Providers by EV Count",
    labels={"Count": "Number of EVs"}
)
fig_utility_bar.update_layout(height=600, yaxis={"tickangle": 0})
st.plotly_chart(fig_utility_bar, use_container_width=True)

st.markdown("""---""")
# Pie Chart for Utility Market Share
fig_utility_pie = px.pie(
    utility_counts_df_pie,
    values="Count",
    names="Electric Utility",
    title="Market Share of Electric Utility Providers"
)
fig_utility_pie.update_layout(height=600)
fig_utility_pie.update_layout(legend=dict(
yanchor="bottom",
y=-10.00,
xanchor="left",
x=0.01
))
st.plotly_chart(fig_utility_pie, use_container_width=True)

st.markdown("""---""")

if not county_utility_dist_df.empty:
    fig_utility_county_bar = px.bar(
        county_utility_dist_df,
        x="County",
        y="Count",
        color="Electric Utility",
        barmode="group",
        title="Distribution of Selected Utility Provider(s) by County",
        labels={"Count": "Vehicle Count"}
    )
    fig_utility_county_bar.update_layout(height=600)
    st.plotly_chart(fig_utility_county_bar, use_container_width=True)
else:
    st.warning("Select Electric Utility Provider(s) from Side Menu to show its distribution across Counties.")


st.markdown("""---""")
# Stacked Bar Graph for EV Type Breakdown
fig_utility_ev_type = px.bar(
    utility_ev_type_df,
    x="Electric Utility",
    y="Count",
    color="Electric Vehicle Type",
    title="EV Type Breakdown by Utility Provider",
    labels={"Count": "Vehicle Count", "Electric Vehicle Type": "EV Type","Electric Utility":"Electric Utility Provider"}
)
fig_utility_ev_type.update_layout(height=600, xaxis={"tickangle": 30})
st.plotly_chart(fig_utility_ev_type, use_container_width=True)

# Section: Make and Model Insights
st.markdown("---")
st.subheader("Make and Model Insights")

# County Distribution by Selected Make
if not county_make_dist_df.empty:
    fig_make_county_bar = px.bar(
        county_make_dist_df,
        x="County",
        y="Count",
        color="Make",
        barmode="group",
        title="Distribution of Selected Make(s) by County",
        labels={"Count": "Vehicle Count"}
    )
    fig_make_county_bar.update_layout(height=400)
    st.plotly_chart(fig_make_county_bar, use_container_width=True)
else:
    st.warning("Select EV Make(s) from Side Menu to show its distribution across Counties.")

st.markdown("---")
# Vehicle Counts by Make
col_make_county, col_popular_model = st.columns([1, 1])

# Bar Graph for County Distribution of Selected Utilities
with col_make_county:
    st.subheader("Number of Vehicles by each Make")
    st.dataframe(make_counts_df, height=300)

# Most Popular Models by Make
with col_popular_model:
    st.subheader("Most Popular Models by each Make")
    if not popular_models_df.empty:
        st.dataframe(popular_models_df, height=300)
    else:
        st.warning("No models available for the selected filters.")

# Section: Detailed Data
st.markdown("---")
st.subheader("Detailed Data")
if not filtered_search_df.empty:
    st.dataframe(
        filtered_search_df[[
            "VIN (1-10)", "County", "City", "Model Year", "Make", "Model",
            "Electric Vehicle Type", "Electric Range", "Clean Alternative Fuel Vehicle (CAFV) Eligibility"
        ]],
        height=400,
    )
else:
    st.warning("No records match the search criteria.")
st.markdown("**Select Filters from the Side Menu to Search for a specific data.**")

st.markdown("---")
st.markdown("Created by [Sneha Lahiri](https://github.com/Sana9909)", unsafe_allow_html=True)