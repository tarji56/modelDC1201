import streamlit as st
import pandas as pd
import pickle
from bayes_opt import BayesianOptimization
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
    result['2-chloro (ppm)'] = result['2-chloro (ppm)'] * 10000
    result['112 TCE (ppm)'] = result['112 TCE (ppm)'] * 10000
    result['CCl4 (ppm)'] = result['CCl4 (ppm)'] * 10000
    return pd.DataFrame(result, columns=name)

# Streamlit UI
st.title("Prediction Web App for DC1201")
st.subheader("Prediction EDC")
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
import streamlit as st
import pandas as pd
import pickle
from bayes_opt import BayesianOptimization

def optimization_DC1201_xg(
    Total_flow_EL_lower, Total_flow_EL_upper,
    HCl_per_EL_in_Feed_lower, HCl_per_EL_in_Feed_upper,
    O2_per_EL_in_Feed_lower, O2_per_EL_in_Feed_upper,
    Rec_per_EL_in_feed_lower, Rec_per_EL_in_feed_upper,
    Vol_EL_in_RG_lower, Vol_EL_in_RG_upper,
    Temp_C_lower, Temp_C_upper,
    P_OVH_lower, P_OVH_upper,
    Cat_inventory_ton_lower, Cat_inventory_ton_upper,
    init_points=5, n_iter=100
):
    pbounds = {
        "Total_flow_EL": (Total_flow_EL_lower, Total_flow_EL_upper),
        "HCl_EL_in_Feed": (HCl_per_EL_in_Feed_lower, HCl_per_EL_in_Feed_upper),
        "O2_EL_in_Feed": (O2_per_EL_in_Feed_lower, O2_per_EL_in_Feed_upper),
        "Rec_EL_in_feed": (Rec_per_EL_in_feed_lower, Rec_per_EL_in_feed_upper),
        "Vol_EL_in_RG": (Vol_EL_in_RG_lower, Vol_EL_in_RG_upper),
        "Temp_C": (Temp_C_lower, Temp_C_upper),
        "P_OVH": (P_OVH_lower, P_OVH_upper),
        "Cat_inventory_ton": (Cat_inventory_ton_lower, Cat_inventory_ton_upper),
    }

    def predict_purity(Total_flow_EL, HCl_EL_in_Feed, O2_EL_in_Feed, Rec_EL_in_feed,
                       Vol_EL_in_RG, Temp_C, P_OVH, Cat_inventory_ton):
        input_data = pd.DataFrame([[Total_flow_EL, HCl_EL_in_Feed, O2_EL_in_Feed, Rec_EL_in_feed,
                                    Vol_EL_in_RG, Temp_C, P_OVH, Cat_inventory_ton]],
                                  columns=['Total flow EL (Ncum/hr)', 'HCl / EL in Feed', 'O2 / El in Feed',
                                           'Rec / El in feed', 'volume El in RG (Ncum/hr)',
                                           'Temp (C)', 'P.OVH', 'cat. inventory (tone)'])
        xg = pickle.load(open('DC1201_xg_model_1.sav', 'rb'))
        prediction = xg.predict(input_data)[0]
        return prediction[0] - prediction[1] - prediction[2] - prediction[3]

    optimizer = BayesianOptimization(
        f=predict_purity,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )

    condition = optimizer.max['params']
    optimal_condtition = pd.DataFrame([condition])
    optimal_condtition = optimal_condtition.rename(columns={
        'Cat_inventory_ton': 'cat. inventory (tone)',
        'HCl_EL_in_Feed': 'HCl / EL in Feed',
        'O2_EL_in_Feed': 'O2 / El in Feed',
        'Rec_EL_in_feed': 'Rec / El in feed',
        'Temp_C': 'Temp (C)',
        'P_OVH': 'P.OVH',
        'Total_flow_EL': 'Total flow EL (Ncum/hr)',
        'Vol_EL_in_RG': 'volume El in RG (Ncum/hr)'
    })

    ordered_cols = ['Total flow EL (Ncum/hr)', 'HCl / EL in Feed', 'O2 / El in Feed',
                    'Rec / El in feed', 'volume El in RG (Ncum/hr)', 'Temp (C)', 'P.OVH',
                    'cat. inventory (tone)']
    optimal_condtition = optimal_condtition[ordered_cols]

    xg = pickle.load(open('DC1201_xg_model_1.sav', 'rb'))
    components = xg.predict(optimal_condtition)
    com = pd.DataFrame(components, columns=['Purity EDC (%wt)', '2-chloro (%ww)', '112 TCE (%ww)', 'CCl4 (%ww)'])
    com['2-chloro (ppm)'] = com['2-chloro (%ww)'] * 10000
    com['112 TCE (ppm)'] = com['112 TCE (%ww)'] * 10000
    com['CCl4 (ppm)'] = com['CCl4 (%ww)'] * 10000
    com = com.drop(columns={'2-chloro (%ww)', '112 TCE (%ww)', 'CCl4 (%ww)'})
    return optimal_condtition, com


# === Streamlit UI ===
st.title("🎯 Optimization for DC1201 (XGBoost-based)")
st.text("""Input information:

**Upper**: Upper bound limiting condition
**Lower**: Lower bound limiting condition
**If parameter adjustment is not required, set both upper and lower values to be identical.**

**Total_flow_El**: Ethylene flow rate (Ncum/hr)
**HCl_per_EL_in_Feed**: Volumetric ratio of HCl per El in feed
**O2_per_EL_in_Feed**: Volumetric ratio of O2 per El in feed
**Rec_per_EL_in_feed**: Volumetric ratio of Recycle gas per Ethylene feed
**Vol_EL_in_RG**: Volumetric flow rate of Ethylene in Recycle gas
**Temp_C**: Temperature in reactor (C)
**P_OVH**: Pressure in reactor (kg/cm^2)
**Cat_inventory_ton**: Catalyst inventory (tons)
**n_iter** (default setting = 100): How many steps of Bayesian optimization you want to perform. The more steps, the more likely to find a good maximum you are.
**init_points** (default setting = 5): How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

---

Example:

    Total_flow_EL: (9091, 9269),
    HCl_EL_in_Feed: (1.83, 2),
    O2_EL_in_Feed: (0.5, 1),
    Rec_EL_in_feed: (1.4, 2),
    Vol_EL_in_RG: (535, 1299),
    Temp_C: (239, 248),
    P_OVH: (3.9, 5),
    Cat_inventory_ton: (70, 80)
""")
with st.form("input_form"):
    st.markdown("### 📥 Define variable bounds")

    Total_flow_EL_lower = st.number_input("Total flow EL lower")
    Total_flow_EL_upper = st.number_input("Total flow EL upper")

    HCl_lower = st.number_input("HCl / EL lower")
    HCl_upper = st.number_input("HCl / EL upper")

    O2_lower = st.number_input("O2 / EL lower")
    O2_upper = st.number_input("O2 / EL upper")

    Rec_lower = st.number_input("Recycle / EL lower")
    Rec_upper = st.number_input("Recycle / EL upper")

    Vol_EL_RG_lower = st.number_input("Volume EL in RG lower")
    Vol_EL_RG_upper = st.number_input("Volume EL in RG upper")

    Temp_lower = st.number_input("Reactor Temp lower (°C)")
    Temp_upper = st.number_input("Reactor Temp upper (°C)")

    P_OVH_lower = st.number_input("Reactor Pressure lower")
    P_OVH_upper = st.number_input("Reactor Pressure upper")

    Cat_lower = st.number_input("Catalyst inventory lower (ton)", )
    Cat_upper = st.number_input("Catalyst inventory upper (ton)", value=80.0)

    init_points = st.number_input("Initial exploration points", value=5, min_value=1)
    n_iter = st.number_input("Number of optimization iterations", value=100, min_value=10)

    submitted = st.form_submit_button("Run Optimization")

if submitted:
    with st.spinner("Running optimization... please wait"):
        df_optimal, df_components = optimization_DC1201_xg(
            Total_flow_EL_lower, Total_flow_EL_upper,
            HCl_lower, HCl_upper,
            O2_lower, O2_upper,
            Rec_lower, Rec_upper,
            Vol_EL_RG_lower, Vol_EL_RG_upper,
            Temp_lower, Temp_upper,
            P_OVH_lower, P_OVH_upper,
            Cat_lower, Cat_upper,
            init_points=init_points,
            n_iter=n_iter
        )

    st.success("Optimization complete")
    st.subheader("Optimal Input Conditions")
    st.dataframe(df_optimal)

    st.subheader("Predicted Product Composition")
    st.dataframe(df_components)
