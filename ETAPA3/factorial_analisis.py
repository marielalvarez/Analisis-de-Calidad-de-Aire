import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bartlett, levene
from statsmodels.tsa.seasonal import seasonal_decompose
from factor_analyzer import FactorAnalyzer


from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd

def asignar_etapa(df):
    """
    Crea una columna 'etapa' basada en la fecha:
    0 = antes de la construcción
    1 = etapa 1
    2 = etapa 2
    """
    inicio_construccion = pd.to_datetime('2022-08-30')
    inicio_construccion2 = pd.to_datetime('2024-02-24')
    
    df = df.copy()
    df['date'] = pd.to_datetime(df["fecha"])
    
    df['etapa'] = 0
    df.loc[df['date'] >= inicio_construccion, 'etapa'] = 1
    df.loc[df['date'] >= inicio_construccion2, 'etapa'] = 2
    
    return df


from scipy.stats import mannwhitneyu, wilcoxon

def pruebas_no_parametricas(df, contaminantes, tipo='independiente', etapa_col='etapa'):
    """
    Aplica pruebas no paramétricas por contaminante.
    
    df: dataframe con columnas de contaminantes y columna de etapa
    contaminantes: lista de nombres de columnas ['CO', 'PM10', ...]
    tipo: 'independiente' = Mann-Whitney, 'pareado' = Wilcoxon
    etapa_col: nombre de columna que indica la etapa
    
    Retorna un dataframe con estadístico y p-valor por contaminante y comparacion
    """
    resultados = []
    
    for c in contaminantes:
        # Comparaciones de etapa: 0 vs 1, 0 vs 2, 1 vs 2
        etapas = df[etapa_col].unique()
        etapas.sort()
        
        for i in range(len(etapas)):
            for j in range(i+1, len(etapas)):
                grupo1 = df[df[etapa_col] == etapas[i]][c].dropna()
                grupo2 = df[df[etapa_col] == etapas[j]][c].dropna()
                
                if tipo == 'independiente':
                    stat, p = mannwhitneyu(grupo1, grupo2, alternative='two-sided')
                elif tipo == 'pareado':
                    # Wilcoxon requiere series del mismo tamaño, hacer merge por fecha si necesario
                    min_len = min(len(grupo1), len(grupo2))
                    stat, p = wilcoxon(grupo1.iloc[:min_len], grupo2.iloc[:min_len])
                else:
                    raise ValueError("tipo debe ser 'independiente' o 'pareado'")
                
                resultados.append({
                    'Contaminante': c,
                    'Comparacion': f'{etapas[i]} vs {etapas[j]}',
                    'Estadistico': stat,
                    'p-valor': p
                })
    
    return pd.DataFrame(resultados)

def analisis_factorial_por_etapa(df, variables, etapa_col='etapa', n_factors=None):
    """
    Realiza análisis factorial por etapa y grafica la matriz de cargas factoriales.
    
    df: DataFrame con datos
    variables: lista de columnas a analizar
    etapa_col: nombre de columna que indica la etapa (0, 1, 2)
    n_factors: número de factores a extraer (si None, se calcula automáticamente con eigenvalues>1)
    """
    etapas = df[etapa_col].unique()
    etapas.sort()
    
    resultados = {}
    
    for e in etapas:
        print(f"\n=== Etapa {e} ===")
        data = df[df[etapa_col] == e][variables].dropna()
        
        # Test KMO y Bartlett
        chi_sq, p_value = calculate_bartlett_sphericity(data)
        kmo_all, kmo_model = calculate_kmo(data)
        print(f"Chi-cuadrado Bartlett: {chi_sq:.2f}, p-valor: {p_value:.3f}")
        print(f"KMO global: {kmo_model:.3f}")
        
        if p_value < 0.05 and kmo_model >= 0.6:
            print("=> Datos adecuados para análisis factorial")
            
            # Determinar número de factores si no se da
            if n_factors is None:
                fa_test = FactorAnalyzer(rotation=None)
                fa_test.fit(data)
                ev = fa_test.get_eigenvalues()[0]  # eigenvalues
                n_factors_eig = sum(ev > 1)
                print(f"Número de factores sugerido (eigen>1): {n_factors_eig}")
                n_factors_use = n_factors_eig
            else:
                n_factors_use = n_factors
            
            # Aplicar factor analysis
            fa = FactorAnalyzer(n_factors=n_factors_use, rotation='varimax')
            fa.fit(data)
            cargas = pd.DataFrame(fa.loadings_, index=variables, columns=[f'Factor{i+1}' for i in range(n_factors_use)])
            resultados[e] = cargas
            
            # Graficar cargas factoriales
            plt.figure(figsize=(8,5))
            sns.heatmap(cargas, annot=True, cmap='coolwarm', center=0)
            plt.title(f"Cargas factoriales - Etapa {e}")
            plt.show()
        else:
            print("=> Datos no adecuados para análisis factorial")
    
    return resultados
