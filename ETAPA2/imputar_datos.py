import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from fancyimpute import IterativeImputer

def imputar_multivariado(df, time_col="date", method="knn"):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).asfreq("H")  # asegurar frecuencia horaria
    
    if method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    elif method == "mice":
        imputer = IterativeImputer(random_state=42, max_iter=10)
    else:
        raise ValueError("Método no soportado. Usa 'knn' o 'mice'.")
    
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    return df_imputed

def evaluar_imputacion(original, imputado, lags=[24, 168]):
    scores = {}
    
    # 1. Distribución
    for stat, func in zip(["mean","var","skew","kurt"],
                          [np.nanmean, np.nanvar, skew, kurtosis]):
        try:
            orig_val = func(original.dropna())
            imp_val  = func(imputado)
            scores[stat] = 1 - abs(orig_val - imp_val) / (abs(orig_val) + 1e-6)
        except:
            scores[stat] = 0
    
    # 2. ACF
    try:
        orig_acf = acf(original.dropna(), nlags=max(lags), fft=True)
        imp_acf  = acf(imputado, nlags=max(lags), fft=True)
        diffs = []
        for lag in lags:
            diffs.append(1 - abs(orig_acf[lag] - imp_acf[lag]))
        scores["acf"] = np.mean(diffs)
    except:
        scores["acf"] = 0
    
    # Score promedio
    return np.mean(list(scores.values()))


def imputar_auto(df, time_col="date", methods=["ffill","linear","knn","mice"]):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).asfreq("H")
    
    imputers_uni = {}
    imputers_multi = {}
    
    # univariados
    if "ffill" in methods:
        imputers_uni["ffill"] = lambda x: x.ffill()
    if "linear" in methods:
        imputers_uni["linear"] = lambda x: x.interpolate(method="time")
    
    # multivariados (sobre todo el df)
    if "knn" in methods:
        imputer = KNNImputer(n_neighbors=5)
        imputers_multi["knn"] = lambda X: pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns, index=X.index
        )
    if "mice" in methods:
        imputer = IterativeImputer(random_state=42, max_iter=10)
        imputers_multi["mice"] = lambda X: pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns, index=X.index
        )

    # aplica multivariados a todo el df una vez
    multi_results = {}
    for name, func in imputers_multi.items():
        try:
            multi_results[name] = func(df)
        except Exception as e:
            continue
    
    # DataFrame final
    df_final = df.copy()
    
    # elegir el mejor método por columna
    for col in df.columns:
        best_score = -np.inf
        best_series = None
        orig_series = df[col]
        
        # probar univariados
        for name, func in imputers_uni.items():
            try:
                imp_series = func(df[[col]])[col]
                score = evaluar_imputacion(orig_series, imp_series)
                if score > best_score:
                    best_score = score
                    best_series = imp_series
            except:
                continue
        
        # probar multivariados
        for name, result in multi_results.items():
            try:
                imp_series = result[col]
                score = evaluar_imputacion(orig_series, imp_series)
                if score > best_score:
                    best_score = score
                    best_series = imp_series
            except:
                continue
        
        df_final[col] = best_series
    
    return df_final
import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.impute import KNNImputer

def imputar_potente(df, time_col="date", method="mice"):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).asfreq("H")
    
    # Paso 1: rellenar gaps cortos con interpolación temporal
    df_interp = df.interpolate(method="time", limit=24)  # hasta 24h seguidas
    
    # Paso 2: aplicar método multivariable para lo que quede
    if method == "mice":
        imputer = IterativeImputer(random_state=42, max_iter=15)
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=7, weights="distance")
    else:
        raise ValueError("Método no soportado: usa 'mice' o 'knn'")
    
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_interp),
        columns=df.columns,
        index=df.index
    )
    return df_imputed

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.arima.model import ARIMA

def imputar_arima(series, order=(2,0,2)):
    series = series.copy()
    
    # verificar si hay nulos
    if series.isna().sum() == 0:
        return series
    
    # entrenar modelo solo con valores observados
    train = series.dropna()
    model = ARIMA(train, order=order)
    fitted = model.fit()
    
    # predecir sobre todo el rango temporal
    pred = fitted.predict(start=series.index[0], end=series.index[-1])
    
    # imputar donde había NA
    series[series.isna()] = pred[series.isna()]
    
    return series
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

class LSTMImputer(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out

def imputar_lstm(df, time_col="date", sequence_length=24, epochs=50, lr=0.001):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).asfreq("H")
    
    # Normalizar los datos
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    # Crear secuencias
    sequences = []
    mask_sequences = []
    for i in range(len(df_scaled) - sequence_length):
        seq = df_scaled.iloc[i:i+sequence_length].values
        mask = ~np.isnan(seq)
        seq[np.isnan(seq)] = 0  # rellenar temporalmente NaNs con 0
        sequences.append(seq)
        mask_sequences.append(mask)
        
    sequences = torch.tensor(sequences, dtype=torch.float32)
    mask_sequences = torch.tensor(mask_sequences, dtype=torch.float32)
    
    # Modelo
    model = LSTMImputer(input_size=df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Entrenamiento
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(sequences)
        loss = loss_fn(output * mask_sequences, sequences * mask_sequences)  # solo comparar datos conocidos
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Imputar los datos faltantes
    with torch.no_grad():
        full_output = model(sequences).numpy()
    
    df_imputed = df_scaled.copy().values
    for i in range(sequence_length, len(df_scaled)):
        for j in range(df.shape[1]):
            if np.isnan(df_imputed[i, j]):
                df_imputed[i, j] = full_output[i - sequence_length, -1, j]  # tomar la predicción más reciente

    # Desnormalizar
    df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=df.columns, index=df.index)
    return df_imputed
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def imputar_random_forest(df, time_col="date", max_iter=5):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).asfreq("H")  # asegurar frecuencia horaria
    
    for it in range(max_iter):
        for col in df.columns:
            if df[col].isna().sum() > 0:
                # datos sin nulos
                train = df[df[col].notna()]
                # datos con nulos
                test = df[df[col].isna()]
                
                if train.shape[0] > 0 and test.shape[0] > 0:
                    X_train = train.drop(columns=[col])
                    y_train = train[col]
                    X_test = test.drop(columns=[col])
                    
                    # entrenar RF
                    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_train, y_train)
                    
                    # predecir faltantes
                    df.loc[df[col].isna(), col] = rf.predict(X_test)
    return df
