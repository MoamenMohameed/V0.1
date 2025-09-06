import streamlit as st
import geopandas as gpd
import numpy as np
import zipfile
import os
import tempfile
import folium
import json
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import datetime
import ee
import logging
from folium.plugins import Draw

import base64

st.set_page_config(
    page_title="GeoLand Analyzer",  # عنوان الصفحة
    page_icon="🌍",                  # أيقونة الصفحة (يمكنك تغييرها)
    layout="wide"                    # التخطيط بعرض كامل
)

# =========================
# Logging setup
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Earth Engine Setup
# =========================
cred_path = os.path.join(os.getcwd(), "ee_credentials.json")
b64_path = os.path.join(os.getcwd(), "service_account.b64")

# التأكد من وجود ملف Base64
if not os.path.exists(b64_path):
    st.error("❌ ملف service_account.b64 غير موجود في مجلد المشروع!")
    st.stop()

# قراءة Base64 وفك التشفير
with open(b64_path, "r") as f:
    service_account_b64 = f.read()

with open(cred_path, "wb") as f:
    f.write(base64.b64decode(service_account_b64))

# تهيئة Earth Engine باستخدام Service Account
try:
    credentials = ee.ServiceAccountCredentials(
        "project000-466321@appspot.gserviceaccount.com",  # ضع هنا اسم حساب الخدمة الصحيح
        key_file=cred_path
    )
    ee.Initialize(credentials)
    st.success("🌍 Earth Engine تم تهيئته بنجاح!")
except Exception as e:
    st.error(f"❌ فشل في تهيئة Earth Engine: {e}")
    st.stop()

# ====================================
# Function to upload SHP/ZIP
# ====================================
def fileupload(file):
    with tempfile.TemporaryDirectory() as tempdir:
        tempath = os.path.join(tempdir, file.name)
        with open(tempath, "wb") as f:
            f.write(file.getbuffer())

        if file.name.lower().endswith(".zip"):
            with zipfile.ZipFile(tempath, "r") as zf:
                zf.extractall(tempdir)

            shp_files = []
            for root, dirs, files in os.walk(tempdir):
                for f_name in files:
                    if f_name.lower().endswith(".shp"):
                        shp_files.append(os.path.join(root, f_name))

            if not shp_files:
                st.error("No SHP file found inside ZIP.")
                return None

            gdf = gpd.read_file(shp_files[0])
        else:
            gdf = gpd.read_file(tempath)

        return gdf

# ====================================
# Session state setup
# ====================================
if "page" not in st.session_state:
    st.session_state.page = "Upload SHP"
if "df" not in st.session_state:
    st.session_state.df = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "drawn_features" not in st.session_state:
    st.session_state.drawn_features = []
if "map_key" not in st.session_state:
    st.session_state.map_key = 0

# ====================================
# Sidebar
# ====================================
st.sidebar.title("📉 Navigation")
if st.sidebar.button("Go to Upload Page"):
    st.session_state.page = "Upload SHP"
if st.sidebar.button("Go to Map"):
    if st.session_state.df is not None:
        st.session_state.page = "Visualization"
    else:
        st.sidebar.warning("Please upload SHP file first!")
if st.sidebar.button("Go to Layout"):
    st.session_state.page = "Layout"
if st.sidebar.button("Go to Aanlysis"):
    if st.session_state.df is not None:
        st.session_state.page = "Dashboard"
    else:
        st.sidebar.warning("Please upload SHP file first!")
if st.sidebar.button("Go to Save on DB"):   # ✅ new button
    if st.session_state.df is not None:
        st.session_state.page = "Save on DB"
    else:
        st.sidebar.warning("Please upload SHP file first!")


# ====================================
# Pages
# ====================================

# --------------------------------------------------------------------------------------------------
# Upload Page
# --------------------------------------------------------------------------------------------------
if st.session_state.page == "Upload SHP":
    st.header("📤 Upload SHP Page")
    file = st.file_uploader("Upload Your ShapeFile (ZIP or SHP)", type=["zip", "shp"])
    
    if file:
        df = fileupload(file)
        if df is not None:
            st.session_state.df = df  
            st.session_state.file_name = file.name   # ✅ store file name
            max_rows = len(df)
            rows = st.slider("Choose number of rows", min_value=1, max_value=max_rows, step=1)
            cols = st.multiselect("Choose columns", df.columns.to_list(), default=df.columns.to_list())
            st.write(df[:rows][cols])

# --------------------------------------------------------------------------------------------------
# Visualization Page
# --------------------------------------------------------------------------------------------------
elif st.session_state.page == "Visualization":
    st.header("🗺 Map Page")
    df = st.session_state.df

    if df is not None:
        st.subheader("📋 Full Data View:")

        # Add selection column
        df_display = df.drop(columns="geometry").copy()
        df_display["Select"] = False

        # Editable table with checkboxes
        edited_df = st.data_editor(
            df_display,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor"
        )

        # Selected rows
        selected_rows = edited_df[edited_df["Select"] == True]

        # Create map
        m = folium.Map(location=[30, 31], zoom_start=8)

        # Satellite background
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Satellite",
            name="Satellite",
            overlay=False,
            control=True
        ).add_to(m)

        # Add all layers
        geojson_data = df.to_json()
        popup_columns = [col for col in df.columns if col != "geometry"][:5]

        folium.GeoJson(
            json.loads(geojson_data),
            name="Layer",
            popup=folium.GeoJsonPopup(
                fields=popup_columns,
                aliases=popup_columns
            ),
            style_function=lambda feature: {
                "fillColor": "blue",
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.4
            }
        ).add_to(m)

        # If user selected a row
        if not selected_rows.empty:
            selected_index = selected_rows.index[0]  # first selected row
            selected_geom = df.loc[selected_index].geometry

            if selected_geom.geom_type == "Point":
                lat, lon = selected_geom.y, selected_geom.x
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Row {selected_index}",
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(m)
                m.location = [lat, lon]
                m.zoom_start = 14
            else:
                geojson = folium.GeoJson(
                    json.loads(gpd.GeoSeries([selected_geom]).to_json()),
                    name="Selected Feature",
                    style_function=lambda feature: {
                        "fillColor": "red",
                        "color": "red",
                        "weight": 3,
                        "fillOpacity": 0.5
                    }
                )
                geojson.add_to(m)
                m.fit_bounds([[selected_geom.bounds[1], selected_geom.bounds[0]],
                              [selected_geom.bounds[3], selected_geom.bounds[2]]])

        # Show map
        folium.LayerControl().add_to(m)
        st_folium(m, width=1500, height=600)

    else:
        st.info("No data available for visualization.")

# --------------------------------------------------------------------------------------------------
# Layout Page
# --------------------------------------------------------------------------------------------------
elif st.session_state.page == "Layout":
    st.header("🖼 Layout Page")
    
    df = st.session_state.df
    if df is None:
        st.info("Please upload SHP file first.")
    else:
        import contextily as ctx
        from matplotlib.patches import Polygon as MplPolygon
        import arabic_reshaper
        from bidi.algorithm import get_display
        from matplotlib_scalebar.scalebar import ScaleBar
        import io

        # Function to handle Arabic text
        def arabic_text(text):
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)
            return bidi_text

        # Convert CRS for Web Mercator
        gdf_poly_web = df.to_crs(epsg=3857)
        geom_scale = 0.9
        image_files = []

        # Initial disabled download button
        zip_buffer = io.BytesIO()
        st.download_button(
            label="⬇️ Download All Images as ZIP",
            data=zip_buffer,
            file_name="all_polygons.zip",
            mime="application/zip",
            disabled=True
        )

        num_cols = 3
        cols = st.columns(num_cols)
        col_index = 0

        for idx, row in gdf_poly_web.iterrows():
            geom = row.geometry
            name_col = None
            for col in row.index:
                if 'name' in col.lower() or 'اسم' in col.lower():
                    name_col = col
                    break
            
            name = row[name_col] if name_col else f"Polygon_{idx}"
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            gpd.GeoDataFrame(geometry=[geom], crs=gdf_poly_web.crs).plot(
                ax=ax, facecolor="none", edgecolor="red", linewidth=2
            )
            
            minx, miny, maxx, maxy = geom.bounds
            width = maxx - minx
            height = maxy - miny
            frame_width = width / geom_scale
            frame_height = height / geom_scale
            x_margin = (frame_width - width) / 2
            y_margin = (frame_height - height) / 2
            ax.set_xlim(minx - x_margin, maxx + x_margin)
            ax.set_ylim(miny - y_margin, maxy + y_margin)
            
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=14)
            for txt in ax.texts:
                txt.remove()
            
            if geom.geom_type == 'Polygon':
                poly_coords = list(geom.exterior.coords)
                mpl_poly = MplPolygon(poly_coords, transform=ax.transData)
                for im in ax.get_images():
                    im.set_clip_path(mpl_poly)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            
            plt.title(arabic_text(str(name)), fontsize=14)
            
            scalebar = ScaleBar(1, units="m", location="lower right")
            ax.add_artist(scalebar)
            
            ax.annotate('↑\nN', xy=(0.95, 0.95), xycoords='axes fraction',
                        ha='center', va='center', fontsize=14, fontweight='bold', color='black')
            
            buf = io.BytesIO()
            plt.savefig(buf, dpi=300, bbox_inches='tight', format='png')
            plt.close(fig)
            buf.seek(0)
            image_files.append((f"{idx}_{name}.png", buf))
            
            with cols[col_index]:
                st.image(buf, caption=str(name), use_container_width=True)
                st.download_button(
                    label=f"Download {name}",
                    data=buf,
                    file_name=f"{idx}_{name}.png",
                    mime="image/png"
                )
            col_index = (col_index + 1) % num_cols

        if image_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for fname, img_buf in image_files:
                    img_buf.seek(0)
                    zf.writestr(fname, img_buf.read())
            zip_buffer.seek(0)

            st.download_button(
                label="⬇️ Download All Images as ZIP",
                data=zip_buffer,
                file_name="all_polygons.zip",
                mime="application/zip"
            )
        else:
            st.warning("No polygons found in the shapefile.")

# --------------------------------------------------------------------------------------------------
# Dashboard Page
# --------------------------------------------------------------------------------------------------
elif st.session_state.page == "Dashboard":
    st.header("📊 Aanlysis Page")
    
    df = st.session_state.df
    
    if df is None:
        st.info("Please upload an SHP file first.")
    else:
        st.subheader("Select Date Range")
        
        end_date = datetime.date.today() - datetime.timedelta(weeks=1)
        start_date = st.date_input(
            "Select start date:",
            value=end_date - datetime.timedelta(weeks=4),
            max_value=end_date
        )
        
        st.write(f"Date range: {start_date} to {end_date}")
        
        m = folium.Map(location=[30, 31], zoom_start=12)
        
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Satellite",
            name="Satellite",
            overlay=False,
            control=True
        ).add_to(m)
        
        gdf_poly = df[df.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not gdf_poly.empty:
            geojson_data = json.loads(gdf_poly.to_json())
            folium.GeoJson(
                geojson_data,
                name="SHP Layer",
                style_function=lambda feature: {
                    "fillColor": "blue",
                    "color": "black",
                    "weight": 2,
                    "fillOpacity": 0.6
                },
                popup=folium.GeoJsonPopup(
                    fields=[col for col in gdf_poly.columns if col != "geometry"][:5],
                    aliases=[col for col in gdf_poly.columns if col != "geometry"][:5]
                )
            ).add_to(m)
        
        draw = Draw(
            export=False,
            draw_options={
                "polyline": False,
                "rectangle": True,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "polygon": True
            }
        )
        draw.add_to(m)
        
        folium.LayerControl().add_to(m)
        
        map_data = st_folium(
            m, 
            width=1200, 
            height=600, 
            key=f"map_{st.session_state.map_key}",
            returned_objects=["all_drawings"]
        )
        
        if map_data and map_data.get("all_drawings"):
            drawn_features = map_data["all_drawings"]
            
            if drawn_features and drawn_features != st.session_state.drawn_features:
                st.session_state.drawn_features = drawn_features
                
                for i, feature in enumerate(drawn_features):
                    if feature["geometry"]["type"] == "Polygon":
                        coords = feature["geometry"]["coordinates"]
                        ee_polygon = ee.Geometry.Polygon(coords)
                        
                        st.subheader(f"Analysis for Drawn Polygon {i+1}")
                        
                        try:
                            s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR")
                                            .filterBounds(ee_polygon)
                                            .filterDate(str(start_date), str(end_date))
                                            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
                            
                            ndvi = s2_collection.mean().normalizedDifference(["B8", "B4"]).rename("NDVI")
                            
                            ndbi = s2_collection.mean().normalizedDifference(["B11", "B8"]).rename("NDBI")
                            
                            ndvi_mean = ndvi.reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=ee_polygon,
                                scale=10
                            ).get("NDVI").getInfo()
                            
                            ndbi_mean = ndbi.reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=ee_polygon,
                                scale=10
                            ).get("NDBI").getInfo()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Mean NDVI", f"{ndvi_mean:.3f}" if ndvi_mean else "N/A")
                            with col2:
                                st.metric("Mean NDBI", f"{ndbi_mean:.3f}" if ndbi_mean else "N/A")
                                
                            ndvi_url = ndvi.getThumbURL({
                                'min': -1, 
                                'max': 1,
                                'palette': ['blue', 'white', 'green'],
                                'region': ee_polygon,
                                'dimensions': 512
                            })
                            
                            ndbi_url = ndbi.getThumbURL({
                                'min': -1, 
                                'max': 1,
                                'palette': ['blue', 'white', 'brown'],
                                'region': ee_polygon,
                                'dimensions': 512
                            })
                            
                            st.image(ndvi_url, caption=f"NDVI for Polygon {i+1}")
                            st.image(ndbi_url, caption=f"NDBI for Polygon {i+1}")
                            
                        except Exception as e:
                            st.error(f"Error calculating indices: {str(e)}")
        
        if st.button("Clear Drawings"):
            st.session_state.drawn_features = []
            st.session_state.map_key += 1
            st.experimental_rerun()

# --------------------------------------------------------------------------------------------------
# Save on DB Page
# --------------------------------------------------------------------------------------------------
elif st.session_state.page == "Save on DB":
    st.header("💾 Save Shapefile to Database")

    st.subheader("🛠 Database Settings")
    db_name = st.text_input("Database Name")
    db_user = st.text_input("User Name")
    db_pass = st.text_input("Password", type="password")
    db_host = st.text_input("Host Address")
    db_port = st.text_input("Port")

    df = st.session_state.df

    if df is None:
        st.warning("⚠️ Please upload a SHP file first before saving to database.")
    else:
        if st.button("🚀 Save Data to Database"):
            try:
                from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
                from sqlalchemy.orm import declarative_base, relationship, sessionmaker
                from geoalchemy2 import Geometry
                from datetime import datetime
                import json as _json

                # --- Database connection ---
                engine = create_engine(
                    f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
                )
                Base = declarative_base()

                # --- Shapefile table ---
                class Shapefile(Base):
                    __tablename__ = "shapefiles"
                    id = Column(Integer, primary_key=True, autoincrement=True)
                    name = Column(String, nullable=False)
                    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
                    polygons = relationship("Polygon", back_populates="shapefile")

                # --- Polygon table ---
                class Polygon(Base):
                    __tablename__ = "polygons"
                    id = Column(Integer, primary_key=True, autoincrement=True)
                    shapefile_id = Column(Integer, ForeignKey("shapefiles.id"))
                    attributes = Column(String)
                    geom = Column(Geometry("POLYGON", srid=4326))
                    shapefile = relationship("Shapefile", back_populates="polygons")

                Base.metadata.create_all(engine)
                Session = sessionmaker(bind=engine)
                session = Session()

                # Use file name from session_state
                shapefile_name = st.session_state.get("file_name", "Uploaded_Shapefile")

                # --- Save Shapefile record ---
                shapefile_record = Shapefile(
                    name=shapefile_name,
                    upload_date=datetime.utcnow()
                )
                session.add(shapefile_record)
                session.commit()

                # --- Prepare GeoDataFrame ---
                gdf_save = st.session_state.df.copy()

                # Ensure CRS is EPSG:4326
                try:
                    if gdf_save.crs is None:
                        st.warning("⚠️ CRS is not defined in the shapefile, EPSG:4326 will be assumed.")
                        gdf_save.set_crs(epsg=4326, inplace=True)
                    else:
                        if gdf_save.crs.to_epsg() != 4326:
                            gdf_save = gdf_save.to_crs(epsg=4326)
                except Exception as _crs_err:
                    st.error(f"❌ Error converting CRS to EPSG:4326: {_crs_err}")
                    raise

                # Explode MultiPolygon into individual Polygons
                try:
                    gdf_save = gdf_save.explode(index_parts=False, ignore_index=True)
                except TypeError:
                    # For compatibility with older GeoPandas versions
                    gdf_save = gdf_save.explode().reset_index(drop=True)

                # Counters
                saved_count = 0
                skipped_count = 0

                # --- Save polygons ---
                for idx, row in gdf_save.iterrows():
                    geom = row.geometry

                    if geom is None or geom.is_empty:
                        skipped_count += 1
                        continue

                    if geom.geom_type != "Polygon":
                        skipped_count += 1
                        continue

                    # Attributes (excluding geometry)
                    try:
                        attrs_dict = row.drop(labels="geometry").to_dict()
                    except Exception:
                        attrs_dict = {}

                    attrs_json = _json.dumps(attrs_dict, ensure_ascii=False, default=str)

                    poly = Polygon(
                        shapefile_id=shapefile_record.id,
                        attributes=attrs_json,
                        geom=f"SRID=4326;{geom.wkt}"
                    )
                    session.add(poly)
                    saved_count += 1

                session.commit()

                # --- User messages ---
                st.success(f"✅ {saved_count} polygon(s) from shapefile '{shapefile_name}' saved into the database.")
                if skipped_count > 0:
                    st.info(f"ℹ️ {skipped_count} record(s) skipped because they were not Polygon or had empty geometry.")

            except Exception as e:
                st.error(f"❌ Error while saving: {str(e)}")
