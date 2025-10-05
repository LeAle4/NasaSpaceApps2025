import sys
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QApplication
from vispy import app, scene
from vispy.scene import visuals
import vispy.geometry as geometry

# Make VisPy use PyQt5 backend
app.use_app('pyqt5')

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
M_SUN = 1.9891e30  # kg
R_SUN = 6.9634e8  # meters
R_EARTH = 6.371e6  # meters
AU = 1.495978707e11
DAY = 86400
TWOPI = 2.0 * np.pi

# Solar System data (Simplified orbital elements)
SOLAR_SYSTEM_PLANETS = {
    'Mercury': {
        'semi_major_axis': 0.387 * AU,
        'eccentricity': 0.206,
        'inclination': 7.0,
        'period': 87.97 * DAY,
        'radius': 2439.7e3,  # meters
        'color': (0.6, 0.6, 0.6, 1.0),  # Gray
        'distance_from_star': 0.387
    },
    'Venus': {
        'semi_major_axis': 0.723 * AU,
        'eccentricity': 0.007,
        'inclination': 3.4,
        'period': 224.7 * DAY,
        'radius': 6051.8e3,
        'color': (1.0, 0.9, 0.4, 1.0),  # Yellowish
        'distance_from_star': 0.723
    },
    'Earth': {
        'semi_major_axis': 1.0 * AU,
        'eccentricity': 0.017,
        'inclination': 0.0,
        'period': 365.25 * DAY,
        'radius': R_EARTH,
        'color': (0.4, 0.6, 0.9, 1.0),  # Blue
        'distance_from_star': 1.0
    },
    'Mars': {
        'semi_major_axis': 1.524 * AU,
        'eccentricity': 0.094,
        'inclination': 1.8,
        'period': 686.98 * DAY,
        'radius': 3389.5e3,
        'color': (0.8, 0.4, 0.2, 1.0),  # Reddish
        'distance_from_star': 1.524
    },
    'Jupiter': {
        'semi_major_axis': 5.204 * AU,
        'eccentricity': 0.049,
        'inclination': 1.3,
        'period': 4332.59 * DAY,
        'radius': 69911e3,
        'color': (0.9, 0.7, 0.4, 1.0),  # Orange-brown
        'distance_from_star': 5.204
    },
    'Saturn': {
        'semi_major_axis': 9.583 * AU,
        'eccentricity': 0.057,
        'inclination': 2.5,
        'period': 10759.22 * DAY,
        'radius': 58232e3,
        'color': (0.9, 0.9, 0.6, 1.0),  # Pale yellow
        'distance_from_star': 9.583
    },
    'Uranus': {
        'semi_major_axis': 19.201 * AU,
        'eccentricity': 0.046,
        'inclination': 0.8,
        'period': 30688.5 * DAY,
        'radius': 25362e3,
        'color': (0.4, 0.9, 0.9, 1.0),  # Cyan
        'distance_from_star': 19.201
    },
    'Neptune': {
        'semi_major_axis': 30.047 * AU,
        'eccentricity': 0.010,
        'inclination': 1.8,
        'period': 60182 * DAY,
        'radius': 24622e3,
        'color': (0.2, 0.4, 0.9, 1.0),  # Deep blue
        'distance_from_star': 30.047
    }
}

# Utilities
def kepler_E_from_M(M, e, tol=1e-9, maxiter=80):
    M = np.mod(M, 2*np.pi)
    E = M if e < 0.8 else np.pi
    for _ in range(maxiter):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f/fp
        E += dE
        if abs(dE) < tol:
            break
    return E

def true_anomaly_from_E(E, e):
    sinv = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e*np.cos(E))
    cosv = (np.cos(E) - e) / (1 - e*np.cos(E))
    return np.arctan2(sinv, cosv)

def rotation_matrix_z(angle_rad):
    c = np.cos(angle_rad); s = np.sin(angle_rad)
    return np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])

def rotation_matrix_x(angle_rad):
    c = np.cos(angle_rad); s = np.sin(angle_rad)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

# Realistic star color approximation
def temp_to_rgb(T):
    T = np.clip(T, 1000, 40000)
    T /= 100.0
    # Red
    r = 255 if T <= 66 else np.clip(329.698727446 * ((T-60) ** -0.1332047592),0,255)
    # Green
    g = 99.4708025861*np.log(T)-161.1195681661 if T<=66 else np.clip(288.1221695283*((T-60)**-0.0755148492),0,255)
    # Blue
    b = 0 if T<=19 else 255 if T>=66 else np.clip(138.5177312231*np.log(T-10)-305.0447927307,0,255)
    return (r/255, g/255, b/255)

def create_glowing_star(radius, temperature, parent_scene):
    """
    Creates a glowing star effect using multiple sphere layers.
    
    Args:
        radius (float): Base radius of the star
        temperature (float): Temperature in Kelvin
        parent_scene: Parent scene node
        
    Returns:
        tuple: (main_star, glow_layers) for animation control
    """
    star_color = temp_to_rgb(temperature)
    
    # Temperature-based brightness factor
    temp_factor = np.clip((temperature - 3000) / (40000 - 3000), 0.2, 2.0)
    
    # Main star core - brightest layer
    core_star = visuals.Sphere(radius=radius, 
                              parent=parent_scene,
                              method='latitude', 
                              subdivisions=64,
                              color=star_color + (1.0,))
    core_star.transform = scene.transforms.MatrixTransform()
    
    # Create multiple glow layers for realistic effect - REDUCED sizes to avoid covering planets
    glow_layers = []
    
    # Inner glow - bright, star-colored (reduced from 1.3 to 1.1)
    inner_glow = visuals.Sphere(radius=radius * 1.1,
                               parent=parent_scene,
                               method='latitude',
                               subdivisions=32,
                               color=star_color + (0.4 * temp_factor,))  # Reduced opacity
    inner_glow.transform = scene.transforms.MatrixTransform()
    glow_layers.append(inner_glow)
    
    # Middle glow - softer, slightly cooler color (reduced from 1.8 to 1.3)
    middle_color = np.array(star_color) * 0.8 + np.array([0.2, 0.2, 0.3]) * 0.2
    middle_glow = visuals.Sphere(radius=radius * 1.3,
                                parent=parent_scene,
                                method='latitude',
                                subdivisions=24,
                                color=tuple(middle_color) + (0.2 * temp_factor,))  # Reduced opacity
    middle_glow.transform = scene.transforms.MatrixTransform()
    glow_layers.append(middle_glow)
    
    # Outer corona - very soft, cooler (reduced from 2.5 to 1.6)
    corona_color = np.array(star_color) * 0.6 + np.array([0.3, 0.3, 0.4]) * 0.4
    corona = visuals.Sphere(radius=radius * 1.6,
                           parent=parent_scene,
                           method='latitude',
                           subdivisions=16,
                           color=tuple(corona_color) + (0.1 * temp_factor,))  # Reduced opacity
    corona.transform = scene.transforms.MatrixTransform()
    glow_layers.append(corona)
    
    # For very hot stars (> 10000K), add blue-white corona (reduced from 3.0 to 1.8)
    if temperature > 10000:
        hot_corona = visuals.Sphere(radius=radius * 1.8,
                                   parent=parent_scene,
                                   method='latitude',
                                   subdivisions=12,
                                   color=(0.8, 0.9, 1.0, 0.05 * temp_factor))  # Much reduced opacity
        hot_corona.transform = scene.transforms.MatrixTransform()
        glow_layers.append(hot_corona)
    
    # Position all elements at origin
    core_star.transform.translate((0, 0, 0))
    for glow in glow_layers:
        glow.transform.translate((0, 0, 0))
    
    return core_star, glow_layers

def orbital_position_vector(a_m, e, i_deg, omega_deg, Omega_deg, M0, t, P):
    n = TWOPI / P
    M = M0 + n*t
    E = kepler_E_from_M(M, e)
    nu = true_anomaly_from_E(E, e)
    r = a_m * (1 - e*np.cos(E))
    x_orb = r*np.cos(nu); y_orb = r*np.sin(nu); z_orb = 0
    vec = np.array([x_orb, y_orb, z_orb])
    R = rotation_matrix_z(np.deg2rad(Omega_deg)) @ rotation_matrix_x(np.deg2rad(i_deg)) @ rotation_matrix_z(np.deg2rad(omega_deg))
    return R @ vec

def add_starfield(view, num_stars=8000, radius=10000.0):
    """Generate a starfield as random points on a very large sphere surrounding the scene"""
    phi = np.random.uniform(0, 2*np.pi, num_stars)
    costheta = np.random.uniform(-1, 1, num_stars)

    theta = np.arccos(costheta)
    # All stars are on the sphere surface (constant radius)
    r = radius

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    positions = np.vstack([x, y, z]).T
    
    # Create more varied star colors and brightness
    colors = np.ones((num_stars, 4))
    # Add some color variation (white to slightly bluish/yellowish stars)
    color_variation = np.random.uniform(0.8, 1.0, (num_stars, 3))
    colors[:,:3] = color_variation
    colors[:,3] = np.random.uniform(0.3, 1.0, num_stars)  # varying brightness

    stars = visuals.Markers(parent=view.scene)
    stars.set_data(pos=positions, face_color=colors, size=2.5)
    
    # Prevent stars from being culled when zooming in
    # Set render properties to always show stars
    stars.order = -1000  # Render stars first (background)
    stars.set_gl_state(depth_test=False)  # Don't depth test stars
    
    return stars

def create_solar_system_planets(parent_scene, scale):
    """Create visual representations of solar system planets with labels"""
    planet_objects = {}
    
    for name, data in SOLAR_SYSTEM_PLANETS.items():
        # Calculate visual radius - use more appropriate scaling for unified system
        # Scale planets to be visible but not too large compared to the star and exoplanet
        planet_vis_radius = max(0.005, (data['radius'] * scale * 20))
        
        # Create planet sphere
        planet = visuals.Sphere(radius=planet_vis_radius,
                              parent=parent_scene,
                              method='ico',
                              subdivisions=2,
                              color=data['color'])
        planet.transform = scene.transforms.MatrixTransform()
        
        # Calculate orbital position (initial position at t=0)
        a_m = data['semi_major_axis']
        e = data['eccentricity']
        i_deg = data['inclination']
        omega_deg = 0.0  # Simplified
        Omega_deg = 0.0  # Simplified
        M0 = np.random.uniform(0, 2*np.pi)  # Random initial position
        
        # Position planet at initial orbital position
        pos = orbital_position_vector(a_m, e, i_deg, omega_deg, Omega_deg, M0, 0.0, data['period'])
        scaled_pos = pos * scale
        planet.transform.translate(scaled_pos)
        
        # Create orbit path for this planet (simplified circular orbit for visibility)
        orbit_angles = np.linspace(0, 2*np.pi, 64)
        orbit_positions = []
        for angle in orbit_angles:
            orbit_t = (angle / (2*np.pi)) * data['period']
            orbit_pos = orbital_position_vector(a_m, e, i_deg, omega_deg, Omega_deg, 0, orbit_t, data['period'])
            orbit_positions.append(orbit_pos * scale)
        
        # Create orbit line
        orbit_line = scene.visuals.Line(pos=np.array(orbit_positions),
                                      color=data['color'][:3] + (0.3,),  # Semi-transparent orbit
                                      width=1,
                                      parent=parent_scene)
        
        # Create text label for planet name - positioned upper right of planet
        label_offset = np.array([planet_vis_radius * 3, planet_vis_radius * 2, planet_vis_radius])  # Upper right offset
        label = scene.visuals.Text(text=name,
                                 pos=scaled_pos + label_offset,
                                 color='white',
                                 font_size=16,  # Bigger text
                                 parent=parent_scene)
        
        planet_objects[name] = {
            'visual': planet,
            'data': data,
            'initial_M': M0,
            'label': label,
            'orbit_line': orbit_line  # Store orbit line for visibility control
        }
        
    return planet_objects

def create_habitable_zone(parent_scene, scale, inner_au, outer_au, exoplanet_incl_deg):
    """Create two 2D habitable zone annuli (rings):
        1. Base system plane (z=0) representing Solar System reference plane.
        2. Exoplanet orbital plane inclined by exoplanet_incl_deg about the X axis.

    Each ring has inner/outer boundary lines plus a translucent fill mesh.
    Returns dict of visuals for toggling visibility.
    """
    inner_r = inner_au * AU * scale
    outer_r = outer_au * AU * scale
    if outer_r < inner_r:
        inner_r, outer_r = outer_r, inner_r

    segments = 256
    ang = np.linspace(0, 2*np.pi, segments, endpoint=True)
    base_inner = np.stack([inner_r*np.cos(ang), inner_r*np.sin(ang), np.zeros_like(ang)], axis=1)
    base_outer = np.stack([outer_r*np.cos(ang), outer_r*np.sin(ang), np.zeros_like(ang)], axis=1)

    # Function to build annulus mesh faces between two rings
    def build_annulus_mesh(inner_pts, outer_pts):
        verts = np.vstack([inner_pts, outer_pts])
        faces = []
        n = inner_pts.shape[0]
        for i in range(n - 1):
            a = i
            b = i + 1
            c = i + n
            d = i + 1 + n
            faces.append([a, c, b])
            faces.append([b, c, d])
        # close loop
        a = n - 1; b = 0; c = a + n; d = n
        faces.append([a, c, b])
        faces.append([b, c, d])
        return verts, np.array(faces, dtype=np.uint32)

    # Base plane visuals
    base_inner_line = scene.visuals.Line(pos=base_inner, color=(0.25, 0.95, 0.25, 0.85), width=2, parent=parent_scene)
    base_outer_line = scene.visuals.Line(pos=base_outer, color=(0.15, 0.7, 0.2, 0.6), width=2, parent=parent_scene)
    v_base, f_base = build_annulus_mesh(base_inner, base_outer)
    base_mesh = scene.visuals.Mesh(vertices=v_base, faces=f_base, color=(0.2, 0.8, 0.3, 0.12), parent=parent_scene)
    base_mesh.set_gl_state(depth_test=False, blend=True)
    base_mesh.order = -55

    # Inclined plane: rotate points about X axis by exoplanet inclination
    incl_rad = np.deg2rad(exoplanet_incl_deg)
    R_incl = rotation_matrix_x(incl_rad)
    incl_inner = (R_incl @ base_inner.T).T
    incl_outer = (R_incl @ base_outer.T).T
    incl_inner_line = scene.visuals.Line(pos=incl_inner, color=(0.3, 0.6, 1.0, 0.9), width=2, parent=parent_scene)
    incl_outer_line = scene.visuals.Line(pos=incl_outer, color=(0.2, 0.45, 0.9, 0.6), width=2, parent=parent_scene)
    v_incl, f_incl = build_annulus_mesh(incl_inner, incl_outer)
    incl_mesh = scene.visuals.Mesh(vertices=v_incl, faces=f_incl, color=(0.25, 0.5, 1.0, 0.10), parent=parent_scene)
    incl_mesh.set_gl_state(depth_test=False, blend=True)
    incl_mesh.order = -54

    return {
        'base_inner_line': base_inner_line,
        'base_outer_line': base_outer_line,
        'base_mesh': base_mesh,
        'incl_inner_line': incl_inner_line,
        'incl_outer_line': incl_outer_line,
        'incl_mesh': incl_mesh
    }

def render_koi_orbit(df, row_index=0, speed=1.0, show_solar_system=False, show_habitable_zone=False):
    row = df.loc[row_index] if row_index in df.index else df.iloc[row_index]

    # KOI data
    P_days = float(row.get('koi_period',1.0)); P_sec = P_days*DAY
    M_star = float(row.get('koi_smass',1.0))*M_SUN
    a_m = float(row.get('koi_sma',np.nan)) * AU if not pd.isna(row.get('koi_sma',np.nan)) else (G*M_star*P_sec**2/(4*np.pi**2))**(1/3)
    e = float(row.get('koi_eccen',0.0) or 0.0)
    i_deg = float(row.get('koi_incl',90.0) or 90.0)
    omega_deg = float(row.get('koi_longp',0.0) or 0.0)
    Omega_deg = 0.0
    M0 = 0.0
    star_r_m = float(row.get('koi_srad',1.0) or 1.0)*R_SUN
    planet_r_m = float(row.get('koi_prad',1.0) or 1.0)*R_EARTH
    teff = float(row.get('koi_steff',5778.0) or 5778.0)
    star_rgb = temp_to_rgb(teff)
    # --- Habitable zone estimation ---
    star_radius_rel = (star_r_m / R_SUN)
    star_temp_rel = (teff / 5778.0)
    luminosity_rel = (star_radius_rel ** 2) * (star_temp_rel ** 4)
    inner_hz_au = np.sqrt(luminosity_rel / 1.1)
    outer_hz_au = np.sqrt(luminosity_rel / 0.53)
    # ---------------------------------------------------------------

    # Automatic scaling
    scale = 1.0 / (1.5 * a_m)
    star_vis_radius = max(0.02, star_r_m*scale*5)
    planet_vis_radius = max(0.005, planet_r_m*scale*50)

    # Orbit points
    ts = np.linspace(0,P_sec,512)
    path = np.array([orbital_position_vector(a_m,e,i_deg,omega_deg,Omega_deg,M0,t,P_sec) for t in ts])
    path_units = path*scale

    # VisPy canvas with PyQt5 backend
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(1000,700))
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(fov=45, distance=3.0)
    
    # Create PyQt5 UI overlay widgets
    from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QCheckBox, QWidget, QFrame
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    
    # Get the native widget to add PyQt5 overlays
    native_widget = canvas.native
    
    # Data values for display
    kepid = row.get('kepid', '')
    if kepid and kepid != 'Unknown':
        exoplanet_name = f"Exoplanet {kepid}"
    else:
        exoplanet_name = "Exoplanet"
    
    star_mass_value = float(row.get('koi_smass',1.0) or 1.0)
    star_radius_value = float(row.get('koi_srad',1.0) or 1.0)
    planet_radius_value = float(row.get('koi_prad',1.0) or 1.0)
    
    # Create data table widget (upper left)
    data_frame = QFrame(native_widget)
    data_frame.setStyleSheet("""
        QFrame {
            background-color: rgba(0, 0, 0, 150);
            border: 2px solid cyan;
            border-radius: 10px;
            padding: 15px;
        }
        QLabel {
            color: white;
            font-size: 12px;
            margin: 3px 0px;
            border: none;
        }
    """)
    data_layout = QVBoxLayout(data_frame)
    data_layout.setSpacing(3)  # Tighter vertical spacing
    
    # Data table title
    title_label = QLabel("System Data")
    title_label.setStyleSheet("color: cyan; font-size: 14px; font-weight: bold; margin-bottom: 5px;")
    data_layout.addWidget(title_label)
    
    # Data rows
    period_label = QLabel(f"Period: {P_days:.1f} days")
    star_mass_label = QLabel(f"Star Mass: {star_mass_value:.2f} M☉")
    star_radius_label = QLabel(f"Star Radius: {star_radius_value:.2f} R☉")
    planet_radius_label = QLabel(f"Planet Radius: {planet_radius_value:.2f} R⊕")
    temp_label = QLabel(f"Temperature: {teff:.0f} K")
    
    data_layout.addWidget(period_label)
    data_layout.addWidget(star_mass_label)
    data_layout.addWidget(star_radius_label)
    data_layout.addWidget(planet_radius_label)
    data_layout.addWidget(temp_label)
    # Let it size to content (we'll position later)
    data_frame.adjustSize()
    data_frame.show()
    
    # Create controls widget (upper right)
    controls_frame = QFrame(native_widget)
    controls_frame.setStyleSheet("""
        QFrame {
            background-color: rgba(0, 0, 0, 150);
            border: 2px solid cyan;
            border-radius: 10px;
            padding: 15px;
        }
        QLabel {
            color: white;
            font-size: 12px;
            margin: 3px 0px;
            border: none;
        }
        QCheckBox {
            color: white;
            font-size: 12px;
            margin: 5px 0px;
            border: none;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
        }
        QCheckBox::indicator:unchecked {
            background-color: white;
            border: 1px solid gray;
        }
        QCheckBox::indicator:checked {
            background-color: green;
            border: 1px solid gray;
        }
    """)
    controls_layout = QVBoxLayout(controls_frame)
    controls_layout.setSpacing(3)  # Tighter spacing
    
    # Controls title
    controls_title = QLabel("Controls")
    controls_title.setStyleSheet("color: cyan; font-size: 14px; font-weight: bold; margin-bottom: 5px;")
    controls_layout.addWidget(controls_title)
    
    # Solar system checkbox
    solar_system_checkbox = QCheckBox("Show Solar System")
    solar_system_checkbox.setChecked(show_solar_system)
    
    def on_solar_system_toggle(checked):
        nonlocal show_solar_system, solar_system_planets
        show_solar_system = checked
        
        if show_solar_system and not solar_system_planets:
            # Create solar system planets
            print("Adding solar system planets...")
            solar_system_planets = create_solar_system_planets(view.scene, scale)
        elif not show_solar_system and solar_system_planets:
            # Hide solar system planets and orbit lines
            print("Hiding solar system planets...")
            for planet_data in solar_system_planets.values():
                planet_data['visual'].visible = False
                planet_data['label'].visible = False
                planet_data['orbit_line'].visible = False
        elif show_solar_system and solar_system_planets:
            # Show existing solar system planets and orbit lines
            print("Showing solar system planets...")
            for planet_data in solar_system_planets.values():
                planet_data['visual'].visible = True
                planet_data['label'].visible = True
                planet_data['orbit_line'].visible = True
    
    solar_system_checkbox.stateChanged.connect(on_solar_system_toggle)
    controls_layout.addWidget(solar_system_checkbox)

    # Habitable Zone checkbox
    habitable_zone_checkbox = QCheckBox("Habitable Zone")
    habitable_zone_checkbox.setChecked(show_habitable_zone)

    def on_hz_toggle(checked):
        nonlocal habitable_zone_objects, show_habitable_zone
        show_habitable_zone = checked
        if checked and habitable_zone_objects is None:
            print("Creating habitable zone visuals...")
            habitable_zone_objects = create_habitable_zone(view.scene, scale, inner_hz_au, outer_hz_au, i_deg)
        elif habitable_zone_objects is not None:
            for v in habitable_zone_objects.values():
                v.visible = checked
    habitable_zone_checkbox.stateChanged.connect(on_hz_toggle)
    controls_layout.addWidget(habitable_zone_checkbox)
    # Size to content (position later)
    controls_frame.adjustSize()
    controls_frame.show()

    # ================= Responsive Corner Positioning (improved) =================
    from PyQt5.QtCore import QTimer
    CORNER_MARGIN = 6  # tighter margin per request

    def reposition_ui():
        # Only adjust size if content width changed to avoid flicker
        prev_w_data = data_frame.width()
        prev_w_ctrl = controls_frame.width()
        data_frame.adjustSize()
        controls_frame.adjustSize()

        w = native_widget.width() or canvas.size[0]
        # Fallback if width is zero during resize init
        if w <= 0:
            return

        # Anchor top-left
        data_frame.move(CORNER_MARGIN, CORNER_MARGIN)

        # Anchor top-right
        right_x = w - controls_frame.width() - CORNER_MARGIN
        if right_x < CORNER_MARGIN:
            right_x = CORNER_MARGIN
        controls_frame.move(right_x, CORNER_MARGIN)

        # Overlap resolution (narrow window)
        if (controls_frame.x() < data_frame.x() + data_frame.width() + 4 and
            controls_frame.y() < data_frame.y() + data_frame.height()):
            controls_frame.move(CORNER_MARGIN, data_frame.y() + data_frame.height() + 4)

    # Connect also to VisPy canvas resize (emitted by backend)
    try:
        canvas.events.resize.disconnect(reposition_ui)  # ensure no duplicates
    except Exception:
        pass
    canvas.events.resize.connect(lambda ev: reposition_ui())

    # Wrap native Qt resizeEvent only once
    if not hasattr(native_widget, '_orig_resize_evt2'):  # sentinel
        native_widget._orig_resize_evt2 = native_widget.resizeEvent
        def _wrapped_resize(ev):
            native_widget._orig_resize_evt2(ev)
            reposition_ui()
        native_widget.resizeEvent = _wrapped_resize

    # Initial deferred positioning
    QTimer.singleShot(0, reposition_ui)
    # ============================================================================

    # Add starfield on a very large sphere background
    starfield_radius = 10000.0
    add_starfield(view, num_stars=8000, radius=starfield_radius)

    # Orbit line
    scene.visuals.Line(pos=path_units[:,:3], color=(0.7,0.7,1.0,0.8), width=2, parent=view.scene)

    # Create glowing star with multiple layers
    print(f"Creating glowing star with T={teff:.0f}K...")
    main_star, glow_layers = create_glowing_star(radius=star_vis_radius,
                                               temperature=teff,
                                               parent_scene=view.scene)
    
    # Store all star components for potential animation
    star_components = [main_star] + glow_layers

    # Planet with more realistic appearance
    planet_color = (0.4, 0.6, 0.9, 1.0)  # Earth-like blue
    planet = visuals.Sphere(radius=planet_vis_radius, 
                          parent=view.scene,
                          method='ico', 
                          subdivisions=3, 
                          color=planet_color)
    planet.transform = scene.transforms.MatrixTransform()
    
    # Add subtle atmosphere glow around planet
    planet_atmosphere = visuals.Sphere(radius=planet_vis_radius * 1.2,
                                     parent=view.scene, 
                                     method='ico',
                                     subdivisions=2,
                                     color=(0.5, 0.7, 1.0, 0.2))  # Atmospheric blue glow
    planet_atmosphere.transform = scene.transforms.MatrixTransform()
    
    # Create exoplanet label - positioned upper right of exoplanet
    exoplanet_label_offset = np.array([planet_vis_radius * 3, planet_vis_radius * 2, planet_vis_radius])
    exoplanet_label = scene.visuals.Text(text=exoplanet_name,
                                       pos=exoplanet_label_offset,  # Will be updated in animation
                                       color='yellow',
                                       font_size=18,  # Even bigger for exoplanet
                                       parent=view.scene)
    exoplanet_label.transform = scene.transforms.MatrixTransform()

    # Initialize solar system planets (create if initially enabled)
    solar_system_planets = {}
    habitable_zone_objects = None
    if show_solar_system:
        print("Adding solar system planets...")
        # Use the SAME scale as the exoplanet system for spatial consistency
        solar_system_planets = create_solar_system_planets(view.scene, scale)
    if show_habitable_zone:
        print(f"Adding habitable zone: {inner_hz_au:.2f}-{outer_hz_au:.2f} AU (L={luminosity_rel:.2f} Lsun)")
        habitable_zone_objects = create_habitable_zone(view.scene, scale, inner_hz_au, outer_hz_au, i_deg)

    # Camera distance with maximum zoom-out limit to keep stars as backdrop
    maxr = np.max(np.linalg.norm(path_units,axis=1))
    initial_distance = max(1.0, maxr*3.0)
    view.camera.distance = initial_distance
    
    # Set up zoom limits manually via event handling
    max_zoom_distance = starfield_radius * 0.7  # Can't zoom further than 70% of star sphere
    min_zoom_distance = 0.05
    
    # Store original camera wheel event
    original_viewbox_mouse_event = view.camera.viewbox_mouse_event
    
    def limited_zoom_camera_event(event):
        """Custom mouse event handler to limit zoom range"""
        if event.type == 'mouse_wheel':
            # Get current distance before processing
            current_distance = view.camera.distance
            
            # Let the original handler process the event
            original_viewbox_mouse_event(event)
            
            # Clamp the distance after the zoom
            if view.camera.distance > max_zoom_distance:
                view.camera.distance = max_zoom_distance
            elif view.camera.distance < min_zoom_distance:
                view.camera.distance = min_zoom_distance
        else:
            # For all other events, use original handler
            original_viewbox_mouse_event(event)
    
    # Replace the camera's mouse event handler
    view.camera.viewbox_mouse_event = limited_zoom_camera_event

    # Animation with unified time scale
    time_sim = 0.0
    def update(ev):
        nonlocal time_sim
        dt = ev.dt * speed * DAY  # Same time scaling for all objects
        time_sim += dt
        
        # Update exoplanet orbital position
        exo_pos = orbital_position_vector(a_m,e,i_deg,omega_deg,Omega_deg,M0,time_sim,P_sec)
        scaled_pos = tuple(exo_pos * scale)
        
        # Update planet position
        planet.transform.reset()
        planet.transform.translate(scaled_pos)
        
        # Update atmosphere position (follows planet)
        planet_atmosphere.transform.reset()
        planet_atmosphere.transform.translate(scaled_pos)
        
        # Update exoplanet label position (upper right of exoplanet)
        exoplanet_label_pos = np.array(scaled_pos) + np.array([planet_vis_radius * 3, planet_vis_radius * 2, planet_vis_radius])
        exoplanet_label.transform.reset()
        exoplanet_label.transform.translate(exoplanet_label_pos)
        
        # Update solar system planets if showing and visible - using SAME time scale
        if show_solar_system and solar_system_planets:
            for planet_name, planet_data in solar_system_planets.items():
                if not planet_data['visual'].visible:
                    continue  # Skip if planet is hidden
                    
                planet_obj = planet_data['visual']
                label_obj = planet_data['label']
                data = planet_data['data']
                
                # Use the orbital elements for proper animation with SAME time scale
                a_m_solar = data['semi_major_axis']
                e_solar = data['eccentricity']
                i_deg_solar = data['inclination']
                omega_deg_solar = 0.0  # Simplified
                Omega_deg_solar = 0.0  # Simplified
                M0_solar = planet_data['initial_M']
                P_sec_solar = data['period']
                
                # Calculate new orbital position using SAME time_sim (unified time scale)
                solar_pos_3d = orbital_position_vector(a_m_solar, e_solar, i_deg_solar, 
                                                     omega_deg_solar, Omega_deg_solar, 
                                                     M0_solar, time_sim, P_sec_solar)
                solar_pos = solar_pos_3d * scale  # Use same scale as exoplanet
                
                # Update planet position
                planet_obj.transform.reset()
                planet_obj.transform.translate(solar_pos)
                
                # Update label position (upper right offset from planet)
                solar_planet_radius = data['radius'] * scale * 50  # Same scaling as creation
                label_offset = solar_pos + np.array([solar_planet_radius * 3, solar_planet_radius * 2, solar_planet_radius])
                label_obj.pos = label_offset
        
        canvas.update()

    timer = app.Timer(interval=1/60.0, connect=update, start=True)
    print(f"Rendering KOI {row.get('kepid','Unknown')} with P={P_days:.2f}d, a={a_m/AU:.3f} AU, e={e:.3f}, i={i_deg:.1f}°")
    print(f"Star: T={teff:.0f}K, R={star_r_m/R_SUN:.2f} R☉ (with layered glow effects)")
    print(f"Planet: R={planet_r_m/R_EARTH:.2f} R⊕ (with atmosphere)")
    
    if show_solar_system:
        print("Solar System: 8 planets with orbital animation and labels")
        for name, planet_obj in solar_system_planets.items():
            data = planet_obj['data']
            print(f"  {name}: {data['distance_from_star']:.1f} AU, Period: {data['period']/DAY:.0f} days")
    if show_habitable_zone:
        print(f"Habitable Zone: {inner_hz_au:.2f}-{outer_hz_au:.2f} AU (Luminosity {luminosity_rel:.2f} Lsun)")
    
    print("Visual effects: multi-layer glowing star, atmospheric planet, habitable zone annulus, starfield background")
    print("Controls: Mouse drag to rotate, scroll to zoom")
    app.run()

def create_instance(df, row_index=0, speed=1.0, show_solar_system=False, show_habitable_zone=False):
    render_koi_orbit(df, row_index=row_index, speed=speed, show_solar_system=show_solar_system, show_habitable_zone=show_habitable_zone)

# Example usage
if __name__=='__main__':
    sample = {'kepid':'KOI-0001','koi_period':54.4183827,'koi_time0bk':162.51384,
              'koi_smass':1.0,'koi_srad':1.0,'koi_prad':1.00,'koi_sma':0.2734,
              'koi_eccen':0.05,'koi_incl':89.57,'koi_longp':90.0,'koi_steff':5778.0}
    df_sample = pd.DataFrame([sample])
    

    render_koi_orbit(df_sample, row_index=0, speed=5.0, show_solar_system=True, show_habitable_zone=True)
