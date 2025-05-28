import streamlit as st
import pandas as pd
import pickle

# ฟังก์ชันเตรียมข้อมูล
def input_parameter(df):
    df = df[~df.apply(lambda row: row.astype(str).str.contains('No Good Data For Calculation', case=False, na=False).any(), axis=1)]
    df = df.iloc[1::,]
    df = df.astype(float)
    mw_rg = 41.6466081753843
    df['Total flow EL (Ncum/hr)'] = df['Flow EL'] + df['flow trim EL']
    df['HCl / EL in Feed'] = df['flow HCl'] / df['Total flow EL (Ncum/hr)']
    df['O2 / El in Feed']  = (df['flow O2'] + df['flow trim O2']) / df['Total flow EL (Ncum/hr)']
    df['Rec / El in feed'] = df['flow RG'] / df['Total flow EL (Ncum/hr)']
    df['volume El in RG (Ncum/hr)'] = df['flow RG']/(22.4 *10**-3) * mw_rg /1000 * df['EL in RG']/100 / 28 * (22.4)
    df = df.drop(columns=['flow O2', 'flow trim O2', 'Flow EL', 'flow trim EL', 'flow HCl', 'flow RG', 'EL in RG'])
    df = df.rename(columns={'cat. Inventory' : 'cat. inventory (tone)','temp.':'Temp (C)'})
    df = df[['Total flow EL (Ncum/hr)', 'HCl / EL in Feed', 'O2 / El in Feed',
             'Rec / El in feed', 'volume El in RG (Ncum/hr)', 'Temp (C)', 'P.OVH',
             'cat. inventory (tone)']]
    return df

# ฟังก์ชันทำนาย
def model_predict(df):
    rf = pickle.load(open('DC1201_xg_model_1.sav','rb'))
    result = rf.predict(df)
    name = ['Purity EDC (%ww)', '2-chloro (ppm)', '112 TCE (ppm)', 'CCl4 (ppm)']
    result = pd.DataFrame(result, columns=name)
    result['Purity EDC (%ww)'] = result['2-chloro (ppm)'] * 10000
    result['2-chloro (ppm)'] = result['112 TCE (ppm)'] * 10000
    result['2-chloro (ppm)'] = result['CCl4 (ppm)'] * 10000
    return pd.DataFrame(result, columns=name)

# Streamlit UI
st.title("Prediction Web App for DC1201")

uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        df_input = pd.read_excel(uploaded_file)
        processed_df = input_parameter(df_input)
        predictions = model_predict(processed_df)
        st.success("✅ Prediction Complete!")
        st.write("### Prediction Results")
        st.dataframe(predictions)

        # ให้ดาวน์โหลดผลลัพธ์
        csv = predictions.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", csv, file_name='predictions.csv', mime='text/csv')
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
