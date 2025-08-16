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
categorical_cols = ['Thoracic', 'Lumbar and Sacrum', 'Number of vertebrae involved(≤2 infectious vertebrae = 0; >2 infectious vertebrae = 1)',
                    'Extent of vertebral destruction', 'Vertebral intraosseous abscess',
                    'Degree of disk destruction', 'Subligamentous spread', 'Skip lesion',
                    'Endplate inflammatory reaction line', 'Paravertebral abscess',
                    'Neurological symptom', 'Fever']

# 定量特征列名 (Involved=1/Not involved=0)
quantitative_cols = ['involved/normal(Signal ratio between infected vertebrae and normal vertebrae in T2WI)', 'ESR(mm/H)', 'CRP(mg/L)', 'A/G', 'WBC(10⁹/L)', 'L%',
                     'Time elapsed to diagnosis of spondylodiscitis (months)', "The patient's height(m)"]

# 输入界面
st.subheader("📝 Please input the characteristic value.")
st.info("💡 Quantitative Feature Description：Involved=1/Not involved=0")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        for col in quantitative_cols:
            input_data[col] = st.number_input(col, value=0.0, format="%.2f")

    with col2:
        for col in categorical_cols:
            options = [0, 1, 2] if col in ['Extent of vertebral destruction', 'Degree of disk destruction(0 = no height loss；1 = height loss <50%；2 = height loss >50%)', 'Paravertebral abscess(0 = absent；1 = small (<½ vertebral body diameter)；2 = large (≥½ vertebral body diameter))'] else [0, 1]
            input_data[col] = st.selectbox(col, options=options)

    submitted = st.form_submit_button("🚀 开始预测")

if submitted:
    input_df = pd.DataFrame([input_data])
    
    # 创建列名映射：界面显示名称 -> scaler期望名称
    column_mapping = {
        'involved/normal(Signal ratio between infected vertebrae and normal vertebrae in T2WI)': 'involved/normal',
        'CRP(mg/L)': 'CRP',
        'WBC(10⁹/L)': 'WBC',
        'Time elapsed to diagnosis of spondylodiscitis (months)': 'Time elapsed to diagnosis of spondylodiscitis (m)',
        "The patient's height(m)": 'Height(m)'
    }
    
    # 重命名列以匹配scaler期望的名称
    input_df_renamed = input_df.rename(columns=column_mapping)
    
    missing_cols = [col for col in scaler.feature_names_in_ if col not in input_df_renamed.columns]
    if missing_cols:
        st.error(f"❌ 缺少特征列：{missing_cols}，请检查列名是否与 scaler 拟合时一致。")
    else:
        input_df_scaled = scaler.transform(input_df_renamed[scaler.feature_names_in_])
        input_combined = pd.DataFrame(input_df_scaled, columns=scaler.feature_names_in_)
        input_combined = pd.concat([input_combined, input_df_renamed[categorical_cols].reset_index(drop=True)], axis=1)

        prediction = model.predict(input_combined)[0]
        prediction_proba = model.predict_proba(input_combined)[0]

        label_mapping = {0: "1 = Pyogenic spondylitis", 1: "2 = Tuberculous spondylitis", 2: "3 = Brucellar spondylitis", 3: "4 = Fungal spondylitis"}
        st.success(f"✅ 模型预测结果：{label_mapping.get(prediction, prediction)}")

        st.subheader("📊 四个组别预测概率：")
        for i, prob in enumerate(prediction_proba):
            percentage = prob * 100
            st.write(f"{label_mapping.get(i)}: {percentage:.1f}%")
