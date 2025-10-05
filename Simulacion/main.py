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
    
    # Create multiple glow layers for realistic effect
    glow_layers = []
    
    # Inner glow - bright, star-colored
    inner_glow = visuals.Sphere(radius=radius * 1.3,
                               parent=parent_scene,
                               method='latitude',
                               subdivisions=32,
                               color=star_color + (0.6 * temp_factor,))
    inner_glow.transform = scene.transforms.MatrixTransform()
    glow_layers.append(inner_glow)
    
    # Middle glow - softer, slightly cooler color
    middle_color = np.array(star_color) * 0.8 + np.array([0.2, 0.2, 0.3]) * 0.2
    middle_glow = visuals.Sphere(radius=radius * 1.8,
                                parent=parent_scene,
                                method='latitude',
                                subdivisions=24,
                                color=tuple(middle_color) + (0.3 * temp_factor,))
    middle_glow.transform = scene.transforms.MatrixTransform()
    glow_layers.append(middle_glow)
    
    # Outer corona - very soft, cooler
    corona_color = np.array(star_color) * 0.6 + np.array([0.3, 0.3, 0.4]) * 0.4
    corona = visuals.Sphere(radius=radius * 2.5,
                           parent=parent_scene,
                           method='latitude',
                           subdivisions=16,
                           color=tuple(corona_color) + (0.15 * temp_factor,))
    corona.transform = scene.transforms.MatrixTransform()
    glow_layers.append(corona)
    
    # For very hot stars (> 10000K), add blue-white corona
    if temperature > 10000:
        hot_corona = visuals.Sphere(radius=radius * 3.0,
                                   parent=parent_scene,
                                   method='latitude',
                                   subdivisions=12,
                                   color=(0.8, 0.9, 1.0, 0.1 * temp_factor))
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

def add_starfield(view, num_stars=2000, radius=15.0):
    """Generate a starfield as random points surrounding the scene"""
    phi = np.random.uniform(0, 2*np.pi, num_stars)
    costheta = np.random.uniform(-1, 1, num_stars)
    u = np.random.uniform(0, 1, num_stars)

    theta = np.arccos(costheta)
    r = radius * (u ** (1/3))  # uniform in volume

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    positions = np.vstack([x, y, z]).T
    colors = np.ones((num_stars, 4))
    colors[:,3] = np.random.uniform(0.7, 1.0, num_stars)  # varying brightness

    stars = visuals.Markers(parent=view.scene)
    stars.set_data(pos=positions, face_color=colors, size=1.5)
    return stars

def create_solar_system_planets(parent_scene, scale):
    """Create visual representations of solar system planets with labels"""
    planet_objects = {}
    
    for name, data in SOLAR_SYSTEM_PLANETS.items():
        # Calculate visual radius (scaled for visibility) - make them bigger
        planet_vis_radius = max(0.008, (data['radius'] * scale * 50))
        
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
        
        # Create text label for planet name
        label = scene.visuals.Text(text=name,
                                 pos=(scaled_pos[0], scaled_pos[1] + planet_vis_radius * 2, scaled_pos[2]),
                                 color='white',
                                 font_size=12,
                                 parent=parent_scene)
        
        planet_objects[name] = {
            'visual': planet,
            'data': data,
            'initial_M': M0,
            'label': label
        }
        
    return planet_objects

def render_koi_orbit(df, row_index=0, speed=1.0, show_solar_system=False):
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

    # Automatic scaling
    scale = 1.0 / (1.5 * a_m)
    star_vis_radius = max(0.02, star_r_m*scale*5)
    planet_vis_radius = max(0.005, planet_r_m*scale*50)

    # Orbit points
    ts = np.linspace(0,P_sec,512)
    path = np.array([orbital_position_vector(a_m,e,i_deg,omega_deg,Omega_deg,M0,t,P_sec) for t in ts])
    path_units = path*scale

    # VisPy canvas
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(1000,700))
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(fov=45, distance=3.0)

    # Add starfield
    add_starfield(view, num_stars=2000, radius=5.0)

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

    # Create solar system planets if requested
    solar_system_planets = {}
    if show_solar_system:
        print("Adding solar system planets...")
        # Adjust scale for solar system - much smaller scale to fit in view but avoid star overlap
        solar_scale = 0.8 / (10 * AU)  # Scale so inner planets are visible outside the star glow
        solar_system_planets = create_solar_system_planets(view.scene, solar_scale)

    # Camera distance
    maxr = np.max(np.linalg.norm(path_units,axis=1))
    view.camera.distance = max(1.0, maxr*3.0)

    # Animation
    time_sim = 0.0
    def update(ev):
        nonlocal time_sim
        dt = ev.dt * speed * DAY
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
        
        # Update solar system planets if showing
        if show_solar_system:
            for planet_name, planet_data in solar_system_planets.items():
                planet_obj = planet_data['visual']
                label_obj = planet_data['label']
                data = planet_data['data']
                
                # Use the orbital elements for proper animation
                a_m_solar = data['semi_major_axis']
                e_solar = data['eccentricity']
                i_deg_solar = data['inclination']
                omega_deg_solar = 0.0  # Simplified
                Omega_deg_solar = 0.0  # Simplified
                M0_solar = planet_data['initial_M']
                P_sec_solar = data['period']
                
                # Calculate new orbital position using proper orbital mechanics
                solar_pos_3d = orbital_position_vector(a_m_solar, e_solar, i_deg_solar, 
                                                     omega_deg_solar, Omega_deg_solar, 
                                                     M0_solar, time_sim, P_sec_solar)
                solar_pos = solar_pos_3d * solar_scale
                
                # Update planet position
                planet_obj.transform.reset()
                planet_obj.transform.translate(solar_pos)
                
                # Update label position (slightly offset from planet)
                label_offset = solar_pos + np.array([0, 0, 0.05])  # Smaller offset
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
    
    print("Visual effects: multi-layer glowing star, atmospheric planet, starfield background")
    print("Controls: Mouse drag to rotate, scroll to zoom")
    app.run()

# Example usage
if __name__=='__main__':
    sample = {'kepid':'KOI-0001','koi_period':54.4183827,'koi_time0bk':162.51384,
              'koi_smass':0.919,'koi_srad':0.927,'koi_prad':2.83,'koi_sma':0.2734,
              'koi_eccen':0.05,'koi_incl':89.57,'koi_longp':90.0,'koi_steff':10000.0}
    df_sample = pd.DataFrame([sample])
    
    # Example 1: Show only the KOI system
    # render_koi_orbit(df_sample, row_index=0, speed=5.0)
    
    # Example 2: Show KOI system with solar system planets for comparison
    render_koi_orbit(df_sample, row_index=0, speed=5.0, show_solar_system=True)
