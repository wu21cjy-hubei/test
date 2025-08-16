import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 页面标题
st.set_page_config(page_title="脊柱感染预测模型演示", layout="wide")
st.title("🌟 Random forest model for predicting infectious spondylitis")

# 加载模型与 scaler
model = joblib.load("RF_model.pkl")
scaler = joblib.load("scaler.pkl")

# 定性与定量特征列名
categorical_cols = ['Thoracic', 'Lumbar and Sacrum', 'Number of vertebrae involved',
                    'Extent of vertebral destruction', 'Vertebral intraosseous abscess(≤2 infectious vertebrae = 0; >2 infectious vertebrae = 1)',
                    'Degree of disk destruction', 'Subligamentous spread', 'Skip lesion',
                    'Endplate inflammatory reaction line', 'Paravertebral abscess',
                    'Neurological symptom', 'Fever']

quantitative_cols = ['involved/normal(Involved=1/Not involved=0)', 'ESR', 'CRP', 'A/G', 'WBC(10⁹/L)', 'L%',
                     'Time elapsed to diagnosis of spondylodiscitis (months)', 'The patient's height(m)']

# 输入界面
st.subheader("📝 Please input the characteristic value.")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        for col in quantitative_cols:
            input_data[col] = st.number_input(col, value=0.0, format="%.2f")

    with col2:
        for col in categorical_cols:
            options = [0, 1, 2] if col in ['Extent of vertebral destruction', 'Degree of disk destruction', 'Paravertebral abscess'] else [0, 1]
            input_data[col] = st.selectbox(col, options=options)

    submitted = st.form_submit_button("🚀 开始预测")

if submitted:
    input_df = pd.DataFrame([input_data])
    missing_cols = [col for col in scaler.feature_names_in_ if col not in input_df.columns]
    if missing_cols:
        st.error(f"❌ 缺少特征列：{missing_cols}，请检查列名是否与 scaler 拟合时一致。")
    else:
        input_df_scaled = scaler.transform(input_df[scaler.feature_names_in_])
        input_combined = pd.DataFrame(input_df_scaled, columns=scaler.feature_names_in_)
        input_combined = pd.concat([input_combined, input_df[categorical_cols].reset_index(drop=True)], axis=1)

        prediction = model.predict(input_combined)[0]
        prediction_proba = model.predict_proba(input_combined)[0]

        label_mapping = {0: "1 = Pyogenic spondylitis", 1: "2 = Tuberculous spondylitis", 2: "3 = Brucellar spondylitis", 3: "4 = Fungal spondylitis"}
        st.success(f"✅ 模型预测结果：{label_mapping.get(prediction, prediction)}")

        st.subheader("📊 四个组别预测概率：")
        for i, prob in enumerate(prediction_proba):
            percentage = prob * 100
            st.write(f"{label_mapping.get(i)}: {percentage:.1f}%")
