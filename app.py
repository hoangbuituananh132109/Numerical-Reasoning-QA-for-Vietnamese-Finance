import streamlit as st
import pandas as pd
import json
import os
import re

# =========================
# UTILS
# =========================
def normalize_program(prog):
    if pd.isna(prog):
        return ""

    prog = str(prog).lower().strip()
    prog = prog.replace(" ", "")

    # normalize commutative ops (add, multiply)
    match = re.match(r"(add|multiply)\((.*),(.*)\)", prog)
    if match:
        op, a, b = match.groups()
        args = sorted([a, b])
        return f"{op}({args[0]},{args[1]})"

    return prog


def is_same_program(pred, gold):
    return normalize_program(pred) == normalize_program(gold)


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_test_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        rows.append({
            "id": item["id"],
            "question": item["qa"]["question"],
            "gold_program": item["qa"]["program"],
            "gold_answer_raw": item["qa"]["exe_ans"],
            "context": "\n".join(item["pre_text"]),
            "table": item["table"]
        })

    return pd.DataFrame(rows)


@st.cache_data
def load_model_results(folder):
    model_files = {
        f.replace("_program.csv", ""): os.path.join(folder, f)
        for f in os.listdir(folder) if f.endswith(".csv")
    }

    model_data = {}

    for model_name, path in model_files.items():
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]

            def get_col(names):
                for n in names:
                    if n in df.columns:
                        return n
                return None

            id_col = get_col(["id"])
            prog_col = get_col(["program_step2", "pred_program", "program"])

            df_clean = pd.DataFrame({
                "id": df[id_col],
                "pred_program": df[prog_col] if prog_col else None
            })

            model_data[model_name] = df_clean

        except Exception as e:
            st.warning(f"Lỗi load {model_name}: {e}")

    return model_data


# =========================
# LOAD
# =========================
test_df = load_test_data("data/test.json")
model_data = load_model_results("results/details")

# =========================
# UI
# =========================
st.title("📊 Program Comparison Dashboard (ViNumQA)")

# chọn id
all_ids = test_df["id"].tolist()
selected_id = st.selectbox("Chọn câu hỏi (id)", all_ids)

# lấy dữ liệu gốc
base_row = test_df[test_df["id"] == selected_id].iloc[0]

# =========================
# HIỂN THỊ DATA
# =========================
st.subheader("📌 Question")
st.write(base_row["question"])

st.subheader("📄 Context")
st.text(base_row["context"][:2000] + "...")

st.subheader("📊 Table")
st.dataframe(pd.DataFrame(base_row["table"]))

st.subheader("✅ Gold Program")
st.code(base_row["gold_program"], language="python")


# =========================
# MODEL VIEW
# =========================
st.subheader("🤖 Model Outputs")

results = []

for model_name, df in model_data.items():
    row = df[df["id"] == selected_id]

    if len(row) == 0:
        continue

    row = row.iloc[0]

    pred_prog = row["pred_program"]
    gold_prog = base_row["gold_program"]

    correct = is_same_program(pred_prog, gold_prog)

    results.append({
        "Model": model_name,
        "Correct": "✅" if correct else "❌"
    })

    with st.expander(f"{model_name}"):

        # So sánh side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.write("### 🔵 Predicted Program")
            st.code(pred_prog, language="python")

        with col2:
            st.write("### 🟢 Gold Program")
            st.code(gold_prog, language="python")

        # normalize compare
        st.write("### 🔍 Normalized Compare")
        st.code(
            f"Pred: {normalize_program(pred_prog)}\nGold: {normalize_program(gold_prog)}",
            language="python"
        )

        # FIX BUG ở đây (KHÔNG dùng inline if)
        st.write("### 📊 Result")
        if correct:
            st.success("Correct ✅")
        else:
            st.error("Wrong ❌")


# =========================
# OVERVIEW TABLE
# =========================
st.subheader("📊 So sánh tổng quan")

if results:
    st.dataframe(pd.DataFrame(results))
else:
    st.warning("Không có dữ liệu model cho id này")


# =========================
# GLOBAL METRICS
# =========================
st.subheader("📈 Program Accuracy (All Data)")

metrics = []

for model_name, df in model_data.items():
    merged = df.merge(test_df, on="id")

    correct = 0
    total = len(merged)

    for _, r in merged.iterrows():
        if is_same_program(r["pred_program"], r["gold_program"]):
            correct += 1

    acc = correct / total if total > 0 else 0

    metrics.append({
        "Model": model_name,
        "Program Accuracy": round(acc, 4),
        "Correct": correct,
        "Total": total
    })

st.dataframe(pd.DataFrame(metrics))


# =========================
# ERROR ANALYSIS
# =========================
st.subheader("🔍 Error Analysis")

selected_model = st.selectbox("Chọn model để phân tích lỗi", list(model_data.keys()))

df = model_data[selected_model]
merged = df.merge(test_df, on="id")

errors = []

for _, r in merged.iterrows():
    if not is_same_program(r["pred_program"], r["gold_program"]):
        errors.append({
            "id": r["id"],
            "question": r["question"],
            "pred_program": r["pred_program"],
            "gold_program": r["gold_program"]
        })

error_df = pd.DataFrame(errors)

st.write(f"❌ Total Errors: {len(error_df)}")

if not error_df.empty:
    st.dataframe(error_df.head(50))