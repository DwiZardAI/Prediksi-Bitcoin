import streamlit as st
import pandas as pd
import requests # Pastikan library ini sudah diinstall/ada di requirements.txt
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

# --- 2. CONSTANTS & SIDEBAR INPUTS ---
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.sidebar.title("⚙️ Control Panel")
st.sidebar.info("Prediksi Harga Bitcoin dengan AI")

selected_stock = st.sidebar.selectbox(
    "Pilih Aset Kripto:", 
    ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "BNB-USD", "ADA-USD")
)

n_years = st.sidebar.slider("Durasi Prediksi (Tahun):", 1, 4)
period = n_years * 365

# --- 3. FUNCTIONS (BACKEND LOGIC) ---
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

# --- 4. MAIN INTERFACE (DATA LOADING) ---
st.title(f"🚀 {selected_stock} AI Prediction")

data_load_state = st.text('Sedang memuat data dari pasar global...')
data = load_data(selected_stock)
data_load_state.text('Proses muat data selesai! ✅')

# --- ERROR HANDLING ---
if data.empty:
    st.error(f"⚠️ Maaf, gagal mengambil data {selected_stock}.")
    st.stop()

# --- 5. SIDEBAR EXTRA TOOLS (Ditaruh sini biar aman karena 'data' udah ada) ---
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Extra Tools")

# A. Fear & Greed Index
with st.sidebar.expander("😨 Market Sentiment"):
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url)
        data_fng = response.json()
        value = data_fng['data'][0]['value']
        status = data_fng['data'][0]['value_classification']
        
        st.write(f"Current Index: **{value}**")
        st.write(f"Status: **{status}**")
        
        val_int = int(value)
        if val_int < 25:
            st.progress(val_int, text="Extreme Fear 🥶")
        elif val_int > 75:
            st.progress(val_int, text="Extreme Greed 🤑")
        else:
            st.progress(val_int, text="Neutral 😐")
    except Exception as e:
        st.error("Gagal memuat data sentiment.")

# B. ROI Calculator (Butuh variabel 'data')
with st.sidebar.expander("💰 Hitung Cuan (ROI)"):
    st.write("Simulasi Investasi Rutin")
    invest_amount = st.number_input("Modal Awal (USD)", min_value=10, value=100)
    
    # Ambil harga pertama kali dan harga sekarang
    start_price = data['Close'].iloc[0] 
    current_price = data['Close'].iloc[-1]
    
    coins_owned = invest_amount / start_price
    current_value = coins_owned * current_price
    profit = current_value - invest_amount
    roi = (profit / invest_amount) * 100
    
    st.write(f"Jika beli di awal 2018 ($ {invest_amount}):")
    if profit > 0:
        st.success(f"Jadi: **${current_value:,.2f}** (+{roi:.0f}%)")
    else:
        st.error(f"Jadi: **${current_value:,.2f}** ({roi:.0f}%)")

# --- 6. VISUALIZATION & METRICS ---
last_price = data['Close'].iloc[-1] 
prev_price = data['Close'].iloc[-2]
delta = last_price - prev_price
delta_percent = (delta / prev_price) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Harga Terakhir (USD)", f"${float(last_price):,.2f}", f"{delta_percent:.2f}%")
with col2:
    st.metric("Volume Transaksi", f"{float(data['Volume'].iloc[-1]):,.0f}")
with col3:
    st.info("Data source: Yahoo Finance")

# --- 7. TABS ---
tab1, tab2 = st.tabs(["📊 Historical Analysis", "🔮 AI Prediction"])

with tab1:
    st.subheader(f"Pergerakan Harga {selected_stock}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='blue')))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"Prediksi Harga {n_years} Tahun ke Depan")
    
    if len(data) < 365:
        st.warning("Data belum cukup untuk prediksi.")
    else:
        st.write("Sedang melakukan kalkulasi AI...")
        df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.write("Analisis Komponen Tren")
        fig2 = m.plot_components(forecast)
        st.write(fig2)