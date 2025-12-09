import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Bitcoin Sentinel AI",
    page_icon="₿",
    layout="wide"
)

# --- 2. CONSTANTS & SIDEBAR ---
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.sidebar.title("⚙️ Control Panel")
st.sidebar.info("Dashboard Prediksi Harga Bitcoin berbasis AI")

# Menambah opsi Altcoin
selected_stock = st.sidebar.selectbox(
    "Pilih Aset Kripto:", 
    ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "BNB-USD", "ADA-USD")
)

n_years = st.sidebar.slider("Durasi Prediksi (Tahun):", 1, 4)
period = n_years * 365

# --- 3. FUNCTIONS (BACKEND LOGIC) ---
@st.cache_data
def load_data(ticker):
    # Download data dari Yahoo Finance
    data = yf.download(ticker, START, TODAY)
    
    # FIX 1: Meratakan kolom jika formatnya MultiIndex (Masalah update yfinance baru)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.reset_index(inplace=True)
    return data

# --- 4. MAIN INTERFACE ---
st.title(f"🚀 {selected_stock} AI Prediction Dashboard")

data_load_state = st.text('Sedang memuat data dari pasar global...')
data = load_data(selected_stock)
data_load_state.text('Proses muat data selesai! ✅')

# FIX 2: ERROR HANDLING (Mencegah Crash jika data kosong)
if data.empty:
    st.error(f"⚠️ Maaf, gagal mengambil data {selected_stock} dari Yahoo Finance.")
    st.warning("Penyebab: Server Yahoo Finance mungkin sedang memblokir akses sementara (Rate Limit) atau koneksi timeout.")
    st.info("Solusi: Coba refresh halaman browser ini (F5) beberapa kali.")
    st.stop() # Berhenti di sini, jangan lanjut ke bawah

# Tampilkan Key Metrics
# Menggunakan float() untuk memastikan format angka aman
last_price = data['Close'].iloc[-1] 
prev_price = data['Close'].iloc[-2]
delta = last_price - prev_price
delta_percent = (delta / prev_price) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Harga Terakhir (USD)", 
        value=f"${float(last_price):,.2f}", 
        delta=f"{delta_percent:.2f}%"
    )
with col2:
    st.metric(
        label="Volume Transaksi", 
        value=f"{float(data['Volume'].iloc[-1]):,.0f}"
    )
with col3:
    st.info("Data source: Yahoo Finance")

# --- 5. TABS LAYOUT ---
tab1, tab2 = st.tabs(["📊 Historical Analysis", "🔮 AI Prediction"])

with tab1:
    st.subheader(f"Pergerakan Harga {selected_stock} (2018 - Sekarang)")
    
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open Price", line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='blue')))
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
        
    plot_raw_data()
    
    with st.expander("Lihat Data Mentah (Tabel)"):
        st.write(data.tail())

with tab2:
    st.subheader(f"Prediksi Harga untuk {n_years} Tahun ke Depan")
    
    # Cek apakah data cukup untuk prediksi
    if len(data) < 365:
        st.warning("Data belum cukup untuk melakukan prediksi akurat jangka panjang.")
    else:
        st.write("Model sedang melakukan kalkulasi tren musiman dan regresi...")
        
        # Persiapan Data untuk Prophet
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        
        # Training Model
        m = Prophet()
        m.fit(df_train)
        
        # Prediksi
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        
        # Visualisasi
        st.write("Grafik Prediksi (Garis Biru = Prediksi, Area Biru Muda = Rentang Kemungkinan)")
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.write("Analisis Komponen Tren (Mingguan/Tahunan)")
        fig2 = m.plot_components(forecast)
        st.write(fig2)