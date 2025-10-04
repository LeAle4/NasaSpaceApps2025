import pandas as pd
import pyvo as vo

# Función para consultar a un servicio TAP y devuelve un DataFrame de pandas
def query_url(url: str, query: str) -> pd.DataFrame:
    """
    Consulta un servicio TAP y devuelve un DataFrame de pandas.
    
    Args:
        url (str): URL del servicio TAP
        query (str): Consulta ADQL/SQL para ejecutar
        
    Returns:
        pd.DataFrame: DataFrame con los resultados o DataFrame vacío si hay errores
    """
    # Validación inicial de parámetros
    if not url or not isinstance(url, str):
        return pd.DataFrame()
    
    if not query or not isinstance(query, str):
        return pd.DataFrame()
    
    # Validación básica de formato URL
    if not (url.startswith('http://') or url.startswith('https://')):
        return pd.DataFrame()
    
    try:
        # Intentar crear el servicio TAP
        service = vo.dal.TAPService(url)
        
        # Verificar si el servicio está disponible
        if not hasattr(service, 'search'):
            return pd.DataFrame()
        
        # Ejecutar la consulta
        result = service.search(query)
        
        # Verificar si hay resultados
        if result is None:
            return pd.DataFrame()
        
        # Convertir a pandas DataFrame
        df = result.to_table().to_pandas()
        
        return df
        
    except (vo.dal.DALServiceError, vo.dal.DALQueryError, Exception):
        # En caso de cualquier error, retornar DataFrame vacío
        return pd.DataFrame()