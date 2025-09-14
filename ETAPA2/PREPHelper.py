import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def data_overview(df):
    print(f"Dataset shape: {df.shape}")
    
    print("\nData Types and Null Values:")
    print(df.info())
    
    print("\nData Summary (Descriptive Statistics):")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nFirst 5 rows:")
    print(df.head())


def describe_variables(df):
    columns = ['PM10', 'NO2', 'CO', 'SO2', 'O3', 'PM2.5']
    print("\nDescription of variables:\n")
    
    for col in columns:
        print(f"Column: {col}")
        print(f"Description: Air quality pollutant concentration in µg/m³ (for PM10, PM2.5) or µg/m³ (for other gases).")
        print(f"Data Type: {df[col].dtype}")
        print(f"Possible values: Numeric, may include values of zero or negative numbers.")
        print(f"Missing values: {df[col].isnull().sum()} missing entries.")
        print('-' * 50)

def explore_quantitative_variables(df):
    columns = ['PM10', 'NO2', 'CO', 'SO2', 'O3', 'PM2.5']
    
    print("\nMeasures of Central Tendency:")
    for col in columns:
        print(f"{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}, Mode: {df[col].mode()[0]:.2f}")
    
    print("\nMeasures of Dispersion:")
    for col in columns:
        print(f"{col} - Range: {df[col].max() - df[col].min():.2f}, Variance: {df[col].var():.2f}, Std Dev: {df[col].std():.2f}")


sns.set_theme(style="whitegrid", palette="muted") 

sns.set_theme(style="whitegrid")

def visualize_outliers(df):
    columns = ['PM10', 'NO2', 'CO', 'SO2', 'O3', 'PM2.5']
    
    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            x=df[col],
            color="#FFB6C1",         # rosa suave
            fliersize=4,
            linewidth=1.5,
            boxprops=dict(alpha=0.7, edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="darkred", linewidth=2)
        )
        plt.title(f"Boxplot de {col}", fontsize=14, fontweight="bold", color="black")
        plt.xlabel(f"{col} (µg/m³)", fontsize=12, color="black")
        sns.despine()
        plt.show()
        
        # Cálculo de outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        print(f"{col} - Outliers: {outliers.shape[0]} outliers")


def plot_histograms(df):
    columns = ['PM10', 'NO2', 'CO', 'SO2', 'O3', 'PM2.5']
    
    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(
            df[col],
            kde=True,
            color="#FF69B4",  
            alpha=0.6,
            edgecolor="white"
        )
        plt.title(f"Histograma de {col}", fontsize=14, fontweight="bold", color="black")
        plt.xlabel(f"{col} (µg/m³)", fontsize=12, color="black")
        plt.ylabel("Frecuencia", fontsize=12, color="black")
        sns.despine()
        plt.show()

def plot_correlation_heatmap(df):
    corr_matrix = df[['PM10', 'NO2', 'CO', 'SO2', 'O3', 'PM2.5']].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Air Quality Variables")
    plt.show()

from statsmodels.tsa.arima.model import ARIMA


# algunos trials de imputación con forecasting


from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fill_missing_with_forecast(df, column, zoom=False):
    """
    Rellena valores faltantes en una serie temporal usando Exponential Smoothing.
    Grafica antes y después con mejor visibilidad.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con índice de fechas y una columna de serie temporal.
        column (str): Nombre de la columna a procesar.
        zoom (bool): Si True, hace zoom en la zona donde había NaN.
    
    Retorna:
        pd.DataFrame: DataFrame con los valores faltantes rellenados.
    """
    series = df[column]
    
    # Plot original con NaN
    plt.figure(figsize=(12,5))
    plt.plot(series, label="Original (con NaN)", color="red", alpha=0.7)
    plt.title(f"Serie original con valores faltantes: {column}")
    plt.legend()
    plt.show()
    
    # Entrenar modelo (solo datos no nulos)
    train = series.dropna()
    model = ExponentialSmoothing(train, trend="add", seasonal=None)
    fit = model.fit()
    
    # Forecast sobre todo el rango
    forecast = fit.predict(start=series.index[0], end=series.index[-1])
    
    # Rellenar
    filled = series.copy()
    missing_idx = filled[filled.isna()].index
    filled[missing_idx] = forecast[missing_idx]
    
    # Plot mejorado
    plt.figure(figsize=(12,5))
    plt.plot(series, label="Serie original", color="grey", alpha=0.6)
    plt.plot(filled, label="Serie imputada", color="blue", linewidth=1.2, alpha=0.4)
    
    # marcar los puntos imputados
    plt.scatter(missing_idx, filled.loc[missing_idx], 
                color="deeppink", marker="o", s=40, label="Valores imputados")
    
    plt.title(f"Serie después de imputación con forecast: {column}")
    plt.legend()
    
    # Si se pide zoom, mostrar solo donde había NaN
    if zoom and len(missing_idx) > 0:
        plt.xlim(missing_idx.min() - 20, missing_idx.max() + 20)
    
    plt.show()
    
    # Retornar DataFrame modificado
    df_copy = df.copy()
    df_copy[column] = filled
    return df_copy

from statsmodels.tsa.arima.model import ARIMA


def fill_all_gaps_arima(df, columns, order=(1,1,1)):
 
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_index)

    for col in columns:
        series = df[col]
        series_int = series.reset_index(drop=True)
        train = series_int.dropna()
        model = ARIMA(train, order=order)
        fit = model.fit()
        forecast = fit.predict(start=0, end=len(series_int)-1)
        series_int[series_int.isna()] = forecast[series_int.isna()]
        df[col] = pd.Series(series_int.values, index=df.index)
    
    return df

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def impute_with_regression(df, columns):
    df_imputed = df.copy()
    
    # Inicializar el imputador para las características (X) que tengan valores faltantes
    imputer = SimpleImputer(strategy='mean')  # Usamos 'mean' o 'median' según lo que prefieras
    
    for col in columns:
        # Seleccionar los registros donde no hay NaN en la columna objetivo
        train_data = df[df[col].notna()]
        # Seleccionar los registros donde hay NaN en la columna objetivo
        test_data = df[df[col].isna()]
        
        # Definir las características (X) y la variable objetivo (y)
        X_train = train_data.drop(columns=[col])  # Las demás columnas como características
        y_train = train_data[col]  # La columna con valores a predecir
        
        # Imputar los valores faltantes en las características (X) antes de entrenar el modelo
        X_train_imputed = imputer.fit_transform(X_train)
        
        # Entrenar el modelo de regresión
        model = LinearRegression()
        model.fit(X_train_imputed, y_train)

        # Predecir los valores faltantes en el conjunto de test (donde la columna tiene NaN)
        X_test = test_data.drop(columns=[col])  # Las mismas características
        X_test_imputed = imputer.transform(X_test)  # Imputar también las características del conjunto de test
        predicted_values = model.predict(X_test_imputed)
        
        # Asignar los valores predichos a los valores faltantes
        df_imputed.loc[test_data.index, col] = predicted_values
    
    return df_imputed

from sklearn.ensemble import RandomForestRegressor

def impute_with_random_forest(df, columns):
    df_imputed = df.copy()

    for col in columns:
        # Seleccionar los registros donde no hay NaN
        train_data = df[df[col].notna()]
        # Seleccionar los registros donde hay NaN
        test_data = df[df[col].isna()]

        # Definir las características y la variable objetivo
        X_train = train_data.drop(columns=[col])  # Las demás columnas como características
        y_train = train_data[col]  # La columna con valores a predecir

        # Entrenar el modelo de Random Forest
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        # Predecir los valores faltantes
        X_test = test_data.drop(columns=[col])  # Las mismas características
        predicted_values = model.predict(X_test)

        # Asignar los valores predichos a los missing values
        df_imputed.loc[test_data.index, col] = predicted_values

    return df_imputed
