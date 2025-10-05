import dataio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from . import visualization

DISCARDBASE = [
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
DISCARDXTRA = [  
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

DISCARDLIST = DISCARDBASE + DISCARDXTRA
EMPTY_THRESHOLD = 0.12  # Filtrar filas con más del 12% de datos faltantes
CORRELATION_THRESHOLD = 0.9  # Umbral para alta colinealidad

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

# --------------------------
# Main preprocessing function
# --------------------------
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

# %%
result, status, message = dataio.loadcsvfile("data/koi_exoplanets.csv")
if(status == 0):
    raise Exception(message)
print(result)

# %%
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

# %%
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

# %%
# ========================================
# FUNCIONES DE ANÁLISIS DE CORRELACIÓN
# ========================================

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

def create_correlation_heatmap(correlation_table, figsize=(18, 18)):
    """
    Crea un mapa de calor de la matriz de correlación.
    
    Args:
        correlation_table (pd.DataFrame): Tabla de correlación
        figsize (tuple): Tamaño de la figura
    """
    import matplotlib.pyplot as plt

    # Convert to numpy array if DataFrame passed
    if hasattr(correlation_table, 'values'):
        mat = correlation_table.values
        labels = list(correlation_table.columns)
    else:
        mat = np.asarray(correlation_table)
        labels = [str(i) for i in range(mat.shape[1])]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.matshow(mat, fignum=fig.number)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Matriz de Correlación", pad=20)

    # Return a Graph wrapper instead of showing
    try:
        return visualization.Graph.from_axes(ax)
    except Exception:
        # Fallback: return the figure object if visualization isn't available
        return fig

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

# %%
# ========================================
# FUNCIONES DE BALANCEO DE DATOS
# ========================================

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
    from imblearn.over_sampling import SMOTENC
    
    print("Antes de SMOTE: ", X_processed.shape, Y.shape)
    print("Índices de flags para SMOTENC:", flag_indices)
    
    # Usar SMOTENC - flags son categóricas
    smote = SMOTENC(categorical_features=flag_indices, random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_processed, Y)
    
    print("Después de SMOTE:", X_balanced.shape, y_balanced.shape)
    
    return X_balanced, y_balanced

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

# %%
# ========================================
# FUNCIONES DE PROCESAMIENTO SEPARADO
# ========================================

def separate_and_save_candidates_vs_others(correlation_df, save_raw=True):
    """
    Separa candidatos del resto de datos y guarda en CSVs separados.
    
    Args:
        correlation_df (pd.DataFrame): DataFrame completo
        save_raw (bool): Si guardar los datos raw separados
        
    Returns:
        tuple: (non_candidates_df, candidates_df)
    """
    import os
    
    # Separar candidatos (0) del resto (-1, 1)
    candidates_df = correlation_df[correlation_df['koi_disposition'] == 0].copy()
    non_candidates_df = correlation_df[correlation_df['koi_disposition'] != 0].copy()
    
    print(f"Candidatos (koi_disposition == 0): {candidates_df.shape}")
    print(f"No candidatos (koi_disposition != 0): {non_candidates_df.shape}")
    
    if save_raw:
        candidates_file = 'data/candidates_raw.csv'
        non_candidates_file = 'data/non_candidates_raw.csv'
        
        candidates_df.to_csv(candidates_file, index=False)
        non_candidates_df.to_csv(non_candidates_file, index=False)
    
    return non_candidates_df, candidates_df

def process_non_candidates_with_smote(non_candidates_df, df_werror):
    """
    Procesa solo los datos que NO son candidatos con SMOTENC.
    
    Args:
        non_candidates_df (pd.DataFrame): Datos que no son candidatos
        df_werror (pd.DataFrame): DataFrame con errores para añadir ruido
        save_processed (bool): Si guardar los datos procesados
        
    Returns:
        tuple: (X_balanced, y_balanced, feature_names)
    """
    import os
    import pandas as pd
    
    # Preparar características y objetivo solo para no candidatos
    Y_non_candidates = non_candidates_df['koi_disposition']
    X_non_candidates = non_candidates_df.drop('koi_disposition', axis=1)
    
    print(f"Procesando datos no candidatos: {X_non_candidates.shape}")
    
    # Añadir ruido basado en errores
    X_noised = add_noise_to_features(X_non_candidates, df_werror)
    
    # Preprocesamiento
    preprocessor, num_cols, flag_cols = build_preprocessing_pipeline(X_noised)
    X_processed = preprocessor.fit_transform(X_noised)
    
    # Índices de flags para SMOTENC
    flag_indices = list(range(len(num_cols), len(num_cols) + len(flag_cols)))
    
    # Aplicar SMOTENC solo a datos no candidatos
    X_balanced, y_balanced = apply_smote_balancing(X_processed, Y_non_candidates, flag_indices)
    
    # Nombres de características
    all_cols = num_cols + flag_cols
    
    return X_balanced, y_balanced, all_cols

def process_candidates_without_smote(candidates_df, df_werror):
    """
    Procesa los candidatos sin aplicar SMOTENC (solo preprocesamiento).
    
    Args:
        candidates_df (pd.DataFrame): Datos de candidatos
        df_werror (pd.DataFrame): DataFrame con errores para añadir ruido
        save_processed (bool): Si guardar los datos procesados
        
    Returns:
        tuple: (X_processed, y_candidates, feature_names)
    """
    import os
    import pandas as pd
    
    # Preparar características y objetivo para candidatos
    Y_candidates = candidates_df['koi_disposition']
    X_candidates = candidates_df.drop('koi_disposition', axis=1)
    
    print(f"Procesando candidatos: {X_candidates.shape}")
    
    # Añadir ruido basado en errores
    X_noised = add_noise_to_features(X_candidates, df_werror)
    
    # Preprocesamiento (sin SMOTENC)
    preprocessor, num_cols, flag_cols = build_preprocessing_pipeline(X_noised)
    X_processed = preprocessor.fit_transform(X_noised)
    
    # Nombres de características
    all_cols = num_cols + flag_cols
    
    return X_processed, Y_candidates.values, all_cols

def preprocess_and_balance(X, Y):
    preprocessor, num_cols, bin_cols, cat_cols = build_preprocessing_pipeline(X)

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

def preprocess_pipeline(dataset: pd.DataFrame, discards_list=DISCARDLIST, empty_threshold=EMPTY_THRESHOLD):
    """
    Función placeholder para futuras implementaciones de pipeline de preprocesamiento.
    """
    # Filtrar columnas según la lista de descartes
    correlation_df_basic, df_werror_basic = clean_dataframe(dataset, discards_list=discards_list)
    correlation_df_basic.corr().abs()

    correlation_df_basic_filtered = filter_rows_by_missing_data(correlation_df_basic, threshold=empty_threshold)

    # Separar candidatos vs no candidatos
    non_candidates_basic, candidates_basic = separate_and_save_candidates_vs_others(
        correlation_df_basic_filtered
    )


    X_balanced_basic, y_balanced_basic, feature_names_basic = process_non_candidates_with_smote(
        non_candidates_basic, df_werror_basic
    )


    X_candidates_basic, y_candidates_basic, _ = process_candidates_without_smote(
        candidates_basic, df_werror_basic
)

    correlation_table, top_n_pairs, high_colinear = analyze_correlation(correlation_df, top_n=30, threshold=CORRELATION_THRESHOLD)

    # Mostrar el mapa de calor
    g = create_correlation_heatmap(correlation_table)

    clean_df, candidates_df = separate_confirmed_and_candidates(correlation_df)

    X, Y = prepare_features_and_target(correlation_df)


    X_noised = add_noise_to_features(X, df_werror)

    non_candidates_df, candidates_df = separate_and_save_candidates_vs_others(correlation_df, save_raw=True)

    X_balanced_non_candidates, y_balanced_non_candidates, feature_names = process_non_candidates_with_smote(
        non_candidates_df, df_werror
    )

    X_processed_candidates, y_candidates, _ = process_candidates_without_smote(
        candidates_df, df_werror
    )






# Guardar datos procesados con nombres específicos
import pandas as pd
import os

# Crear directorio si no existe
os.makedirs('data', exist_ok=True)

# Guardar datos básicos
X_balanced_basic_df = pd.DataFrame(X_balanced_basic, columns=feature_names_basic)
y_balanced_basic_df = pd.DataFrame(y_balanced_basic, columns=['koi_disposition'])
X_candidates_basic_df = pd.DataFrame(X_candidates_basic, columns=feature_names_basic)
y_candidates_basic_df = pd.DataFrame(y_candidates_basic, columns=['koi_disposition'])

# Guardar archivos con sufijo '_basic'
X_balanced_basic_df.to_csv('data/non_candidates_features37.csv', index=False)
y_balanced_basic_df.to_csv('data/non_candidates_labels37.csv', index=False)
X_candidates_basic_df.to_csv('data/candidates_features37.csv', index=False)
y_candidates_basic_df.to_csv('data/candidates_labels37.csv', index=False)

# %%
# ========================================
# PROCESAMIENTO CASO 2: DISCARDS + EXTRA_DISCARDS
# ========================================

print("\n" + "="*60)
print("CASO 2: PROCESAMIENTO CON DESCARTE EXTENDIDO (discards + extra_discards)")
print("="*60)

# Combinar listas de descarte
discards_extended = discards + extra_discards
print(f"Total de columnas a descartar: {len(discards_extended)}")
print(f"Columnas adicionales en extra_discards: {extra_discards}")

# Limpieza con descarte extendido
correlation_df_extended, df_werror_extended = clean_dataframe(result, discards_list=discards_extended)

print(f"\nDataFrame después de descarte extendido: {correlation_df_extended.shape}")
print(f"Columnas restantes: {correlation_df_extended.columns.tolist()}")

# Filtrar filas con muchos datos faltantes
correlation_df_extended_filtered = filter_rows_by_missing_data(correlation_df_extended, threshold=0.12)

# Separar candidatos vs no candidatos
non_candidates_extended, candidates_extended = separate_and_save_candidates_vs_others(
    correlation_df_extended_filtered, save_raw=False  # No guardar raw aquí
)

print(f"\nCaso extendido - No candidatos: {non_candidates_extended.shape}")
print(f"Caso extendido - Candidatos: {candidates_extended.shape}")

# Procesar no candidatos con SMOTENC
print("\n--- Procesando NO candidatos (descarte extendido) ---")
X_balanced_extended, y_balanced_extended, feature_names_extended = process_non_candidates_with_smote(
    non_candidates_extended, df_werror_extended
)

# Procesar candidatos sin SMOTENC
print("\n--- Procesando candidatos (descarte extendido) ---")
X_candidates_extended, y_candidates_extended, _ = process_candidates_without_smote(
    candidates_extended, df_werror_extended
)

# Guardar datos procesados con nombres específicos
X_balanced_extended_df = pd.DataFrame(X_balanced_extended, columns=feature_names_extended)
y_balanced_extended_df = pd.DataFrame(y_balanced_extended, columns=['koi_disposition'])
X_candidates_extended_df = pd.DataFrame(X_candidates_extended, columns=feature_names_extended)
y_candidates_extended_df = pd.DataFrame(y_candidates_extended, columns=['koi_disposition'])

# Guardar archivos con sufijo '_extended'
X_balanced_extended_df.to_csv('data/non_candidates_features25.csv', index=False)
y_balanced_extended_df.to_csv('data/non_candidates_labels25.csv', index=False)
X_candidates_extended_df.to_csv('data/candidates_features25.csv', index=False)
y_candidates_extended_df.to_csv('data/candidates_labels25.csv', index=False)



