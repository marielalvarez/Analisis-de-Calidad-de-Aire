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


def crear_etapas(df, col_fecha='date'):
    """
    Crear columna de etapas basada en fechas de construcción del metro
    """
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    
    # Definir fechas de corte
    inicio_construccion = pd.to_datetime('2022-08-30')
    inicio_construccion2 = pd.to_datetime('2024-02-24')  # ¿O es otra fase de construcción?
    
    # Crear etapas
    condiciones = [
        df[col_fecha] < inicio_construccion,
        (df[col_fecha] >= inicio_construccion) & (df[col_fecha] < inicio_construccion2),
        df[col_fecha] >= inicio_construccion2
    ]
    
    etapas = ['Pre-construcción', 'Etapa1', 'Etapa2']
    df['etapa'] = np.select(condiciones, [0, 1, 2])
    df['etapa_nombre'] = np.select(condiciones, etapas)
    
    # Estadísticas por etapa
    print(" DISTRIBUCIÓN POR ETAPAS:")
    resumen = df.groupby(['etapa', 'etapa_nombre']).size().reset_index(name='registros')
    resumen['fecha_min'] = df.groupby('etapa')[col_fecha].min().values
    resumen['fecha_max'] = df.groupby('etapa')[col_fecha].max().values
    
    for _, row in resumen.iterrows():
        print(f"Etapa {int(row['etapa'])}: {row['etapa_nombre']}")
        print(f"  - Registros: {row['registros']:,}")
        print(f"  - Período: {row['fecha_min'].strftime('%Y-%m-%d')} → {row['fecha_max'].strftime('%Y-%m-%d')}")
        print()
    
    return df

def validar_datos_para_factorial(df, cols_contaminantes):
    """
    Validar que los datos sean adecuados para análisis factorial
    """
    print(" VALIDACIÓN PARA ANÁLISIS FACTORIAL:")
    
    corr_matrix = df[cols_contaminantes].corr()
    print(f"Rango de correlaciones: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.3f} - {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")
    
    # 2. Test de esfericidad de Bartlett (aproximación)
    try:
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
        chi_square, p_value = calculate_bartlett_sphericity(df[cols_contaminantes].dropna())
        print(f"Test de Bartlett: χ² = {chi_square:.2f}, p = {p_value:.4f}")
        if p_value < 0.05:
            print(" Los datos son adecuados para análisis factorial")
        else:
            print("  Los datos pueden no ser adecuados para análisis factorial")
    except:
        print(" Instalar factor_analyzer para test de Bartlett: pip install factor_analyzer")
    
    # 3. KMO (Kaiser-Meyer-Olkin) - aproximación simple
   
    return corr_matrix

def analisis_factorial_por_etapa(df, cols_contaminantes, n_componentes=None):
    """
    Realizar análisis factorial separado por etapa
    """
    resultados = {}
    
    for etapa in sorted(df['etapa'].unique()):
        print(f"\n{'='*50}")
        print(f"  ETAPA {etapa}: {df[df['etapa']==etapa]['etapa_nombre'].iloc[0]}")
        print(f"{'='*50}")
        
        # Filtrar datos de la etapa
        df_etapa = df[df['etapa'] == etapa][cols_contaminantes].copy()
        
        # Manejar valores faltantes
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_etapa), 
            columns=cols_contaminantes,
            index=df_etapa.index
        )
        
        # Estandarizar
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_imputed),
            columns=cols_contaminantes,
            index=df_imputed.index
        )
        
        # PCA para determinar número óptimo de componentes
        pca = PCA()
        pca.fit(df_scaled)
        
        print(f" Varianza explicada por componente: {pca.explained_variance_ratio_[:4]}")
        print(f" Varianza acumulada: {pca.explained_variance_ratio_.cumsum()[:4]}")
        
        # Análisis factorial
        if n_componentes is None:
            # Usar criterio de Kaiser (eigenvalues > 1)
            n_fact = sum(pca.explained_variance_ > 1)
        else:
            n_fact = n_componentes
            
        print(f" Usando {n_fact} factores")
        
        # Factor Analysis
        fa = FactorAnalysis(n_components=n_fact, random_state=42)
        fa.fit(df_scaled)
        
        # Cargas factoriales
        loadings = pd.DataFrame(
            fa.components_.T,
            columns=[f'Factor_{i+1}' for i in range(n_fact)],
            index=cols_contaminantes
        )
        
        print(f"\n CARGAS FACTORIALES:")
        print(loadings.round(3))
        
        # Identificar qué contaminantes cargan en cada factor
        print(f"\n INTERPRETACIÓN DE FACTORES:")
        for i, col in enumerate(loadings.columns):
            cargas_altas = loadings[loadings[col].abs() > 0.5][col].sort_values(key=abs, ascending=False)
            if not cargas_altas.empty:
                print(f"{col}: {', '.join([f'{idx}({val:.2f})' for idx, val in cargas_altas.items()])}")
        
        resultados[etapa] = {
            'loadings': loadings,
            'fa_model': fa,
            'pca_variance': pca.explained_variance_ratio_,
            'scaler': scaler,
            'n_samples': len(df_etapa)
        }
    
    return resultados

def comparar_factores_entre_etapas(resultados):
    """
    Comparar la estructura factorial entre etapas
    """
    print(f"\n{'='*60}")
    print(" COMPARACIÓN ENTRE ETAPAS")
    print(f"{'='*60}")
    
    # Comparar cargas factoriales del Factor 1 entre etapas
    factor1_comparison = pd.DataFrame()
    
    for etapa, res in resultados.items():
        if 'Factor_1' in res['loadings'].columns:
            factor1_comparison[f'Etapa_{etapa}'] = res['loadings']['Factor_1']
    
    if not factor1_comparison.empty:
        print(" CARGAS DEL FACTOR 1 POR ETAPA:")
        print(factor1_comparison.round(3))
        
        # Correlación entre estructuras factoriales
        print(f"\n CORRELACIÓN DE ESTRUCTURAS FACTORIALES:")
        corr_estructuras = factor1_comparison.corr()
        print(corr_estructuras.round(3))

def visualizar_analisis(df, cols_contaminantes, resultados):
    """
    Crear visualizaciones del análisis factorial
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlaciones por etapa
    for i, etapa in enumerate(sorted(df['etapa'].unique())):
        ax = axes[0, i] if i < 2 else axes[1, i-2]
        df_etapa = df[df['etapa'] == etapa][cols_contaminantes]
        corr = df_etapa.corr()
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar=i==0)
        ax.set_title(f'Etapa {etapa}: {df[df["etapa"]==etapa]["etapa_nombre"].iloc[0]}')
    
    # 4. Varianza explicada
    ax = axes[1, 1]
    etapas = sorted(resultados.keys())
    for etapa in etapas:
        variance = resultados[etapa]['pca_variance'][:4]
        ax.plot(range(1, len(variance)+1), np.cumsum(variance), 
               marker='o', label=f'Etapa {etapa}')
    
    ax.set_xlabel('Número de Componentes')
    ax.set_ylabel('Varianza Explicada Acumulada')
    ax.set_title('Varianza Explicada por Etapa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Ejemplo de uso
def analisis_completo_contaminantes(df, col_fecha='date'):
    """
    Función principal que ejecuta todo el análisis
    """
    # Definir columnas de contaminantes
    cols_contaminantes = ['CO', 'PM10', 'PM2.5', 'SO2', 'NO2', 'O3']
    
    # Verificar que las columnas existan
    cols_disponibles = [col for col in cols_contaminantes if col in df.columns]
    if len(cols_disponibles) != len(cols_contaminantes):
        print(f"  Columnas faltantes: {set(cols_contaminantes) - set(cols_disponibles)}")
        cols_contaminantes = cols_disponibles
    
    print(f" Analizando contaminantes: {cols_contaminantes}")
    
    # 1. Crear etapas
    df_con_etapas = crear_etapas(df, col_fecha)
    
    # 2. Validar datos
    corr_matrix = validar_datos_para_factorial(df_con_etapas, cols_contaminantes)
    
    # 3. Análisis factorial por etapa
    resultados = analisis_factorial_por_etapa(df_con_etapas, cols_contaminantes)
    
    # 4. Comparar entre etapas
    comparar_factores_entre_etapas(resultados)
    
    # 5. Visualizar
    fig = visualizar_analisis(df_con_etapas, cols_contaminantes, resultados)
    
    return df_con_etapas, resultados, fig

