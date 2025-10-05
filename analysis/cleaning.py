import dataio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataio
from imblearn.over_sampling import SMOTENC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

EMPTY_THRESHOLD = 0.12

class SkewAwareImputer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.imputers = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)  # ensure DataFrame for column-wise ops
        self.imputers = {}
        for col in X.columns:
            skew = X[col].dropna().skew()
            if abs(skew) < self.threshold:
                strategy = "mean"
            else:
                strategy = "median"
            imputer = SimpleImputer(strategy=strategy)
            imputer.fit(X[[col]])
            self.imputers[col] = imputer
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, imputer in self.imputers.items():
            X[col] = imputer.transform(X[[col]])
        return X.values

def build_preprocessing_pipeline(df, nan_col_threshold=0.8):
    """
    Builds preprocessing pipeline:
    - Numeric features: impute and scale
    - Flag features: pass through unchanged (no imputation, no encoding)
    """
    # 1. Drop columns with too many NaNs
    df = df.loc[:, df.isnull().mean() < nan_col_threshold]

    # 2. Separate numeric and flag features
    flag_features = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    numeric_features = [col for col in df.columns if col not in flag_features]

    # 3. Pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SkewAwareImputer(threshold=1.0)),
        ('scaler', StandardScaler())
    ])

    # Pass through flags unchanged
    from sklearn.preprocessing import FunctionTransformer
    flag_pipeline = FunctionTransformer(lambda x: x)

    # 4. Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('flag', flag_pipeline, flag_features)
        ]
    )

    return preprocessor, numeric_features, flag_features

DISCARDS = [
  "kepler_name",
  "kepid",
  "kepoi_name",
  "koi_pdisposition",
  #"koi_disposition",
  "koi_comment",
  "koi_disp_prov",
  "koi_parm_prov",
  "koi_sparprov",
  "koi_vet_date",
  "koi_datalink_dvr",
  "koi_datalink_dvs",
  "koi_quarters",
  "koi_fittype",
  "koi_dikco_msky",
  "koi_dikco_mra",
  "koi_sma",
  "koi_score",
  "koi_max_sngle_ev",
  "koi_hmag",
  "koi_trans_mod",
  "koi_model_dof",
  "koi_delivname",
  "koi_vet_stat",
  "koi_limbdark_mod",
  "koi_sparprov",
  "koi_tce_delivname",
  "koi_time0bk",
  "koi_fwm_sra",
  "koi_fwm_sdec",
  "koi_rmag",
  "koi_kmag",
  "koi_zmag",
  "koi_zmag",
  "koi_ror",
  "koi_jmag",
  "koi_kepmag",
  "koi_kmag",
  "koi_rmag",
  "koi_period",
  "koi_period",
  "koi_period",
  "koi_zmag",
  "koi_rmag",
  "koi_zmag",
  "koi_zmag",
  "koi_kepmag",
  "koi_zmag",
  "koi_ldm_coeff2",
  "koi_jmag",
  "koi_imag",
  "koi_zmag",
  "koi_imag",
  "koi_kepmag",
  "koi_kepmag",
  "koi_rmag",
  "koi_rmag",
  "koi_dikco_mdec"
]
DISCARDSXTRA = [  
  "koi_slogg", #
  "koi_smet",
  "koi_time0",
  "koi_srad",
  "koi_steff",
  "koi_smass",
  "koi_gmag",
  "koi_tce_plnt_num",
  "koi_fwm_prao",
  "koi_fwm_pdeco",
  "koi_ldm_coeff1",
  "koi_srho"
]

# ========================================
# FUNCIONES DE CONVERSIÓN Y UTILIDAD
# ========================================

def string_to_ms(texto: str) -> int:
    """
    Convierte un literal de duración como '47d21h15m30.5s' a milisegundos.
    
    Args:
        texto (str): Cadena con formato de duración (ej: '47d21h15m30.5s')
        
    Returns:
        int: Total en milisegundos
        
    Raises:
        ValueError: Si el formato es inválido
    """
    import re
    
    patron = re.compile(
        r'^(?:(?P<d>\d+)d)?'
        r'(?:(?P<h>\d+)h)?'
        r'(?:(?P<m>\d+)m)?'
        r'(?:(?P<s>\d+(?:\.\d+)?)s)?$'
    )
    m = patron.match(texto.strip())
    if not m or not any(m.group(g) for g in ('d','h','m','s')):
        raise ValueError(f"Formato inválido: {texto}")
        
    dias = int(m.group('d')) if m.group('d') else 0
    horas = int(m.group('h')) if m.group('h') else 0
    minutos = int(m.group('m')) if m.group('m') else 0
    segundos = float(m.group('s')) if m.group('s') else 0.0

    total_ms = (
        dias * 86400000 +
        horas * 3600000 +
        minutos * 60000 +
        int(round(segundos * 1000))
    )
    return total_ms

def status_to_int(status: str) -> int:
    """
    Convierte un estado de exoplaneta a un entero.
    
    Args:
        status (str): Estado del exoplaneta
        
    Returns:
        int: 1 para CONFIRMED, 0 para CANDIDATE, -1 para FALSE POSITIVE
        
    Raises:
        ValueError: Si el estado no es válido
    """
    mapping = {
        'CONFIRMED': 1,
        'CANDIDATE': 0,
        'FALSE POSITIVE': -1
    }
    status_upper = status.strip().upper()
    if status_upper not in mapping:
        raise ValueError(f"Estado inválido: {status}")
    return mapping[status_upper]

# ========================================
# FUNCIONES DE LIMPIEZA Y FILTRADO
# ========================================

def clean_dataframe(df, discards_list=None):
    """
    Limpia el DataFrame eliminando columnas no deseadas y procesando tipos de datos.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        tuple: (correlation_df, cleaned_df) - DataFrames procesados
    """
    print(f"DataFrame original: {df.shape[1]} columnas")
    
    # Crear una copia para trabajar
    cleaned_df = df.copy()
    
    # Obtener columnas a descartar que realmente existen en el DataFrame
    existing_discard_cols = [col for col in discards_list if col in cleaned_df.columns]
    print(f"Columnas a eliminar encontradas: {len(existing_discard_cols)}")
    
    # Eliminar columnas especificadas
    if existing_discard_cols:
        cleaned_df = cleaned_df.drop(columns=existing_discard_cols)
        print(f"Después de eliminar columnas especificadas: {cleaned_df.shape[1]} columnas")
    
    # Eliminar columnas que terminan en '_str'
    str_columns = [col for col in cleaned_df.columns if col.endswith('_str')]
    if str_columns:
        cleaned_df = cleaned_df.drop(columns=str_columns)
        print(f"Después de eliminar columnas '_str': {cleaned_df.shape[1]} columnas")
    
    # Crear DataFrame para análisis de correlación (sin columnas de error)
    correlation_df = cleaned_df.copy()
    
    # Eliminar columnas de error para el análisis de correlación
    error_columns = [col for col in correlation_df.columns 
                    if col.endswith('_err1') or col.endswith('_err2') or col.endswith('_err')]
    if error_columns:
        correlation_df = correlation_df.drop(columns=error_columns)
        print(f"Después de eliminar columnas de error: {correlation_df.shape[1]} columnas")
    
    # Convertir disposición a entero
    if 'koi_disposition' in correlation_df.columns:
        correlation_df = correlation_df.copy()  # Evitar SettingWithCopyWarning
        correlation_df['koi_disposition'] = correlation_df['koi_disposition'].apply(status_to_int)
        print("Columna 'koi_disposition' convertida a entero")
    
    # Eliminar columnas con un solo valor único
    unique_counts = correlation_df.nunique()
    single_value_cols = unique_counts[unique_counts <= 1].index.tolist()
    if single_value_cols:
        correlation_df = correlation_df.drop(columns=single_value_cols)
        print(f"Después de eliminar columnas con un solo valor: {correlation_df.shape[1]} columnas")
    
    # Eliminar columnas completamente vacías
    empty_cols = correlation_df.columns[correlation_df.isnull().all()].tolist()
    if empty_cols:
        correlation_df = correlation_df.drop(columns=empty_cols)
        print(f"Después de eliminar columnas completamente vacías: {correlation_df.shape[1]} columnas")
    
    print(f"DataFrame final para correlación: {correlation_df.shape}")
    print(f"DataFrame limpio con errores: {cleaned_df.shape}")
    
    return correlation_df, cleaned_df

def filter_rows_by_missing_data(df, threshold=0.12):
    """
    Filtra filas basándose en el porcentaje de datos faltantes.
    
    Args:
        df (pd.DataFrame): DataFrame a filtrar
        threshold (float): Umbral de datos faltantes permitidos (0-1)
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    print(f"Forma original: {df.shape}")
    filtered_df = df[df.isnull().mean(axis=1) < threshold]
    print(f"Forma después del filtro: {filtered_df.shape}")
    return filtered_df

def analyze_correlation(df, top_n=30, threshold=0.9):
    """
    Analiza la correlación entre variables y encuentra pares altamente correlacionados.
    
    Args:
        df (pd.DataFrame): DataFrame para analizar
        top_n (int): Número de pares más correlacionados a mostrar
        threshold (float): Umbral para considerar alta colinealidad
        
    Returns:
        tuple: (correlation_table, top_n_pairs, high_colinear)
    """
    # Calcular tabla de correlación
    correlation_table = df.corr().abs()
    
    # Crear pares de correlación (evitar duplicados)
    corr_pairs = (
        correlation_table
            .stack()
            .reset_index()
            .rename(columns={'level_0': 'var1', 'level_1': 'var2', 0: 'corr'})
    )
    corr_pairs = corr_pairs[corr_pairs.var1 < corr_pairs.var2]
    
    # Ordenar de forma descendente
    corr_pairs_sorted = corr_pairs.sort_values('corr', ascending=False)
    
    # Top N pares
    top_n_pairs = corr_pairs_sorted.head(top_n)
    
    # Pares con alta colinealidad
    high_colinear = corr_pairs_sorted[corr_pairs_sorted['corr'] >= threshold]
    
    return correlation_table, top_n_pairs, high_colinear

def print_correlation_analysis(top_n_pairs, high_colinear, top_n=30, threshold=0.9):
    """
    Imprime los resultados del análisis de correlación.
    
    Args:
        top_n_pairs (pd.DataFrame): Top N pares correlacionados
        high_colinear (pd.DataFrame): Pares con alta colinealidad
        top_n (int): Número de pares mostrados
        threshold (float): Umbral de colinealidad
    """
    print(f"Top {top_n} pares de variables correlacionadas (correlación absoluta):")
    print(top_n_pairs.to_string(index=False))
    
    print(f"\nPares con correlación >= {threshold}: (count={len(high_colinear)})")
    print(high_colinear.to_string(index=False))

def make_correlation_heatmap(correlation_table, figsize=(18, 18)):
    """
    Create a correlation heatmap and return a Graph wrapper (figure + axes).

    Args:
        correlation_table (pd.DataFrame): Absolute-value correlation matrix
        figsize (tuple): Figure size

    Returns:
        Graph: a Graph object containing the figure and axes for further use or saving
    """
    # Create a figure and axis explicitly so we can return them wrapped in Graph
    fig, ax = plt.subplots(figsize=figsize)

    # Use imshow for a compact heatmap; ensure the matrix is 2D and square
    im = ax.imshow(correlation_table.values, cmap='viridis', interpolation='nearest', aspect='auto')
    cbar = fig.colorbar(im, ax=ax)

    # Set ticks and labels
    labels = list(correlation_table.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Matriz de Correlación")

    # Tighten layout and return Graph
    try:
        from analysis.visualization import Graph
    except Exception:
        # fallback to local import path used by other modules
        from visualization import Graph

    g = Graph.from_figure(fig)
    return g

# ========================================
# FUNCIONES DE PREPARACIÓN DE DATOS
# ========================================

def separate_confirmed_and_candidates(df):
    """
    Separa los datos en confirmados y candidatos.
    
    Args:
        df (pd.DataFrame): DataFrame con columna 'koi_disposition'
        
    Returns:
        tuple: (clean_df, candidates_df) - confirmados/falsos positivos y candidatos
    """
    clean_df = df[df['koi_disposition'] != 0]  # Confirmados y falsos positivos
    candidates_df = df[df['koi_disposition'] == 0]  # Candidatos
    
    print(f"Datos confirmados/falsos positivos: {clean_df.shape}")
    print(f"Datos candidatos: {candidates_df.shape}")
    
    return clean_df, candidates_df

def prepare_features_and_target(df):
    """
    Separa las características (X) del objetivo (Y).
    
    Args:
        df (pd.DataFrame): DataFrame con todas las variables
        
    Returns:
        tuple: (X, Y) - características y variable objetivo
    """
    Y = df['koi_disposition']
    X = df.drop('koi_disposition', axis=1)
    
    print(f"Características: {X.columns.tolist()}")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de Y: {Y.shape}")
    
    return X, Y

def add_noise_to_features(X, df_with_errors):
    """
    Añade ruido gaussiano a las características basándose en sus errores.
    
    Args:
        X (pd.DataFrame): DataFrame de características
        df_with_errors (pd.DataFrame): DataFrame original con columnas de error
        
    Returns:
        pd.DataFrame: DataFrame con ruido añadido
    """
    import numpy as np
    
    X_noised = X.copy()
    
    for col in X.columns:
        # Buscar columnas de error correspondientes
        if col + "_err1" in df_with_errors.columns:
            sigma = (df_with_errors[col + "_err1"] + df_with_errors[col + "_err2"]) / 2
            X_noised[col] += np.random.normal(0, sigma)
        elif col + "_err" in df_with_errors.columns:
            sigma = df_with_errors[col + "_err"]
            X_noised[col] += np.random.normal(0, sigma)
    
    return X_noised

def apply_smote_balancing(X_processed, Y, flag_indices):
    """
    Aplica SMOTENC para balancear los datos, considerando características categóricas.
    
    Args:
        X_processed (np.array): Características procesadas
        Y (pd.Series): Variable objetivo
        flag_indices (list): Índices de las características categóricas
        
    Returns:
        tuple: (X_balanced, y_balanced) - datos balanceados
    """
    
    print("Antes de SMOTE: ", X_processed.shape, Y.shape)
    print("Índices de flags para SMOTENC:", flag_indices)
    
    # Usar SMOTENC - flags son categóricas
    smote = SMOTENC(categorical_features=flag_indices, random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_processed, Y)
    
    print("Después de SMOTE:", X_balanced.shape, y_balanced.shape)
    
    return X_balanced, y_balanced

def preprocess_and_balance(X, Y):
    """Preprocess X (fit transformer) and balance classes using SMOTENC.

    Returns balanced DataFrames and the fitted preprocessor.
    """
    preprocessor, num_cols, flag_cols = build_preprocessing_pipeline(X)

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Determine categorical column indices for SMOTENC
    feature_names = preprocessor.get_feature_names_out()
    cat_indices = [i for i, name in enumerate(feature_names) 
                   if any(name.startswith(prefix) for prefix in ['bin__', 'cat__'])]

    # Apply SMOTENC
    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_balanced, y_balanced = smote_nc.fit_resample(X_processed, Y)

    # Convert back to DataFrame
    X_balanced_df = pd.DataFrame(X_balanced, columns=feature_names)
    Y_balanced_df = pd.DataFrame(y_balanced, columns=[Y.name] if hasattr(Y, 'name') else ['target'])

    return X_balanced_df, Y_balanced_df, preprocessor

def save_data(X_balanced, y_balanced, feature_names, target_name):
    """
    Guarda los datos procesados en archivos CSV.
    
    Args:
        X_balanced (np.array): Características balanceadas
        y_balanced (np.array): Objetivo balanceado
        feature_names (list): Nombres de las características
        target_name (str): Nombre de la variable objetivo
    """
    import pandas as pd
    
    # Convertir a DataFrames
    X_balanced_df = pd.DataFrame(X_balanced, columns=feature_names)
    Y_balanced_df = pd.DataFrame(y_balanced, columns=[target_name])
    
    # Guardar archivos
    X_balanced_df.to_csv('data/parameters.csv', index=False)
    Y_balanced_df.to_csv('data/labels.csv', index=False)

    print("Datos guardados en 'data/parameters.csv' y 'data/labels.csv'")

def pipeline_full_cleaning_and_balancing(df, discards_list=DISCARDS + DISCARDSXTRA):
    # 1. Limpieza inicial
    #Se descartan los elementos de la lista DISCARD y DISCARDXTRA
    correlation_df, cleaned_df = clean_dataframe(df, discards_list=discards_list)
    
    # 2. Filtrado de filas con muchos NaNs
    filtered_df = filter_rows_by_missing_data(cleaned_df, threshold=EMPTY_THRESHOLD)
    
    # 3. Análisis de correlación
    corr_table, top_n_pairs, high_colinear = analyze_correlation(correlation_df, top_n=30, threshold=0.9)
    print_correlation_analysis(top_n_pairs, high_colinear, top_n=30, threshold=0.9)
    corrGraph = make_correlation_heatmap(corr_table)
    
    # 4. Separar confirmados y candidatos
    clean_df, candidates_df = separate_confirmed_and_candidates(filtered_df)
    
    # 5. Preparar características y objetivo
    X, Y = prepare_features_and_target(clean_df)
    
    # 6. Añadir ruido basado en errores
    X_noised = add_noise_to_features(X, cleaned_df)
    
    # 7. Preprocesamiento y balanceo
    X_balanced_df, Y_balanced_df, preprocessor = preprocess_and_balance(X_noised, Y)
    
    # 8. Guardar datos
    save_data(X_balanced_df.values, Y_balanced_df.values, X_balanced_df.columns.tolist(), Y.name if hasattr(Y, 'name') else 'target')
    
    return X_balanced_df, Y_balanced_df, preprocessor, corrGraph