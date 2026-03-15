# ======================================
# DeepCSAT 
# ======================================

import sys
import os

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

# ---------- Handle mode (train or app) ----------
if len(sys.argv) > 1 and sys.argv[1] == '--mode' and sys.argv[2] == 'app':
    # -------------------- STREAMLIT APP --------------------
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import plotly.express as px
    import tensorflow as tf
    from tensorflow import keras

    st.set_page_config(page_title="DeepCSAT Predictor", layout="wide")
    st.title("🎯 DeepCSAT – Customer Satisfaction Score Predictor")
    st.markdown("Enter customer interaction details below to predict the CSAT score (1–5).")

    @st.cache_resource
    def load_artifacts():
        model = keras.models.load_model('ann_model.h5')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor

    try:
        model, preprocessor = load_artifacts()
    except Exception as e:
        st.error("Model files not found. Please run training first: python ecom.py")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        channel = st.selectbox("Channel", ["Inbound","Outbound","Email"])
        category = st.selectbox("Category", ["Product Queries","Returns","Order Related","Cancellation","Feedback"])
        subcat = st.selectbox("Sub‑category", ["Installation","Delayed","Missing","Return request","General Enquiry"])
        city = st.selectbox("Customer City", ["Delhi","Mumbai","Bangalore","Chennai","Kolkata","Pune"])
    with col2:
        product_cat = st.selectbox("Product Category", ["Electronics","Mobile","Home","LifeStyle","Furniture","Books"])
        item_price = st.number_input("Item Price (₹)", 0, 500000, 5000, 100)
        handling_time = st.number_input("Handling Time (mins)", 0, 300, 10, 1)
        agent = st.selectbox("Agent Name", ["John","Priya","Raj","Maria","Ahmed","Wei"])
    with col3:
        supervisor = st.selectbox("Supervisor", ["Alex","Neha","Carlos","Mei"])
        manager = st.selectbox("Manager", ["Sarah","Vikram","Chen"])
        tenure = st.selectbox("Tenure Bucket", ["0-30","31-60","61-90",">90","On Job Training"])
        shift = st.selectbox("Agent Shift", ["Morning","Afternoon","Evening","Night","Split"])
        sentiment = st.slider("Sentiment", -1.0, 1.0, 0.0, 0.1)
        response_time = st.number_input("Response Time (mins)", 0, 1000, 30, 5)

    if st.button("Predict CSAT Score"):
        input_dict = {
            'channel_name': channel,
            'category': category,
            'Sub-category': subcat,
            'Customer_City': city,
            'Product_category': product_cat,
            'Item_price': item_price,
            'connected_handling_time': handling_time,
            'Agent_name': agent,
            'Supervisor': supervisor,
            'Manager': manager,
            'Tenure Bucket': tenure,
            'Agent Shift': shift,
            'sentiment': sentiment,
            'response_time_mins': response_time,
        }
        input_df = pd.DataFrame([input_dict])
        input_processed = preprocessor.transform(input_df)
        probs = model.predict(input_processed, verbose=0)[0]
        pred_class = np.argmax(probs) + 1
        confidence = np.max(probs) * 100

        st.success(f"### Predicted CSAT Score: **{pred_class}**")
        st.info(f"Confidence: {confidence:.2f}%")

        prob_df = pd.DataFrame({'Score': [1,2,3,4,5], 'Probability': probs})
        fig = px.bar(prob_df, x='Score', y='Probability', title="Class Probabilities")
        st.plotly_chart(fig)
    sys.exit()

# -------------------- TRAINING SECTION --------------------
print("="*60)
print("DeepCSAT – Training Mode")
print("="*60)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# ---------- 1. Load or generate data (with meaningful patterns) ----------
csv_file = 'eCommerce_Customer_support_data.csv'
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print("Real dataset loaded.")
else:
    print("CSV not found. Generating enhanced synthetic dataset with signal...")
    np.random.seed(42)
    n_rows = 20000   # larger dataset helps learning

    # Base random features
    channel = np.random.choice(['Inbound','Outbound','Email'], size=n_rows)
    category = np.random.choice(['Product Queries','Returns','Order Related','Cancellation','Feedback'], size=n_rows)
    subcat = np.random.choice(['Installation','Delayed','Missing','Return request','General Enquiry'], size=n_rows)
    city = np.random.choice(['Delhi','Mumbai','Bangalore','Chennai','Kolkata','Pune'], size=n_rows)
    product_cat = np.random.choice(['Electronics','Mobile','Home','LifeStyle','Furniture','Books'], size=n_rows)
    agent = np.random.choice(['John','Priya','Raj','Maria','Ahmed','Wei'], size=n_rows)
    supervisor = np.random.choice(['Alex','Neha','Carlos','Mei'], size=n_rows)
    manager = np.random.choice(['Sarah','Vikram','Chen'], size=n_rows)
    tenure = np.random.choice(['0-30','31-60','61-90','>90','On Job Training'], size=n_rows)
    shift = np.random.choice(['Morning','Afternoon','Evening','Night','Split'], size=n_rows)

    # Numerical features with relationships to CSAT
    sentiment = np.random.uniform(-1, 1, size=n_rows)
    handling_time = np.random.uniform(1, 60, size=n_rows)
    response_time = np.random.exponential(30, size=n_rows)
    item_price = np.random.uniform(100, 50000, size=n_rows)

    # Build a synthetic CSAT score based on features
    csat_base = 3.0  # neutral baseline
    # Sentiment effect
    csat_base += sentiment * 1.5
    # Handling time effect (longer handling reduces satisfaction)
    csat_base -= (handling_time / 30) * 0.8
    # Response time effect
    csat_base -= (response_time / 60) * 0.5
    # Price effect: very high price may lower satisfaction slightly
    csat_base -= (item_price / 50000) * 0.3
    # Shift effect: Night shift lower, Morning higher
    shift_effect = {'Morning': 0.3, 'Afternoon': 0.1, 'Evening': -0.1, 'Night': -0.4, 'Split': 0.0}
    csat_base += np.array([shift_effect[s] for s in shift])
    # Category effect
    cat_effect = {'Returns': -0.5, 'Cancellation': -0.6, 'Product Queries': 0.2, 'Order Related': 0.1, 'Feedback': 0.3}
    csat_base += np.array([cat_effect[c] for c in category])
    # Channel effect
    chan_effect = {'Inbound': 0.2, 'Outbound': 0.0, 'Email': -0.2}
    csat_base += np.array([chan_effect[ch] for ch in channel])
    # Tenure effect: more experienced agents give better service
    ten_effect = {'0-30': -0.2, '31-60': 0.0, '61-90': 0.2, '>90': 0.4, 'On Job Training': -0.3}
    csat_base += np.array([ten_effect[t] for t in tenure])
    # Add random noise to make it realistic
    noise = np.random.normal(0, 0.5, size=n_rows)
    csat_score = csat_base + noise

    # Clip and round to 1-5 integer
    csat_score = np.clip(np.round(csat_score), 1, 5).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'CSAT Score': csat_score,
        'channel_name': channel,
        'category': category,
        'Sub-category': subcat,
        'Customer_City': city,
        'Product_category': product_cat,
        'Item_price': item_price.round(),
        'connected_handling_time': handling_time.round(),
        'Agent_name': agent,
        'Supervisor': supervisor,
        'Manager': manager,
        'Tenure Bucket': tenure,
        'Agent Shift': shift,
        'sentiment': sentiment,
        'response_time_mins': response_time.round(),
    })

print("Dataset shape:", df.shape)
print("CSAT distribution:\n", df['CSAT Score'].value_counts().sort_index())

# ---------- 2. Data Cleaning ----------
drop_cols = ['Unique id', 'Order_id', 'Customer Remarks', 'order_date_time',
             'Issue_reported at', 'issue_responded', 'Survey_response_Date']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

df['CSAT Score'] = pd.to_numeric(df['CSAT Score'], errors='coerce')
df.dropna(subset=['CSAT Score'], inplace=True)

X = df.drop('CSAT Score', axis=1)
y = df['CSAT Score'].astype(int) - 1  # convert to 0-4 for NN

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical_cols:
    X[col].fillna('Unknown', inplace=True)
for col in numerical_cols:
    X[col].fillna(X[col].median(), inplace=True)

# Save preprocessor info for later
num_medians = X[numerical_cols].median()
cat_modes = X[categorical_cols].mode().iloc[0] if not X[categorical_cols].mode().empty else pd.Series('Unknown', index=categorical_cols)

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# ---------- 3. Generate 6 Graphs (unchanged) ----------
sns.set_style('whitegrid')
fig = plt.figure(figsize=(18, 12))

# 3.1 CSAT Score distribution
plt.subplot(2, 3, 1)
sns.countplot(x=y+1, palette='viridis')
plt.title('CSAT Score Distribution')
plt.xlabel('Score')
plt.ylabel('Count')

# 3.2 Channel counts
plt.subplot(2, 3, 2)
if 'channel_name' in categorical_cols:
    channel_counts = X['channel_name'].value_counts()
    sns.barplot(x=channel_counts.values, y=channel_counts.index, palette='magma')
    plt.title('Interactions by Channel')
    plt.xlabel('Count')
else:
    plt.text(0.5,0.5,'No channel data',ha='center')

# 3.3 Top 10 Categories
plt.subplot(2, 3, 3)
if 'category' in categorical_cols:
    top_cats = X['category'].value_counts().head(10)
    sns.barplot(x=top_cats.values, y=top_cats.index, palette='plasma')
    plt.title('Top 10 Categories')
    plt.xlabel('Count')
else:
    plt.text(0.5,0.5,'No category data',ha='center')

# 3.4 Handling time vs CSAT
plt.subplot(2, 3, 4)
if 'connected_handling_time' in numerical_cols:
    plot_df = pd.DataFrame({'handling_time': X['connected_handling_time'], 'CSAT': y+1})
    sns.boxplot(x='CSAT', y='handling_time', data=plot_df, palette='Set2')
    plt.title('Handling Time vs CSAT Score')
    plt.ylabel('Handling Time (minutes)')
else:
    plt.text(0.5,0.5,'No handling time',ha='center')

# 3.5 Average CSAT by Agent Shift
plt.subplot(2, 3, 5)
if 'Agent Shift' in categorical_cols:
    df_plot = pd.concat([X['Agent Shift'], pd.Series(y+1, name='CSAT')], axis=1)
    shift_csat = df_plot.groupby('Agent Shift')['CSAT'].mean().sort_values()
    sns.barplot(x=shift_csat.values, y=shift_csat.index, palette='coolwarm')
    plt.title('Average CSAT by Agent Shift')
    plt.xlabel('Average CSAT Score')
else:
    plt.text(0.5,0.5,'No Agent Shift',ha='center')

# 3.6 Tenure Bucket distribution
plt.subplot(2, 3, 6)
if 'Tenure Bucket' in categorical_cols:
    tenure_counts = X['Tenure Bucket'].value_counts()
    plt.pie(tenure_counts.values, labels=tenure_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Tenure Bucket Distribution')
else:
    plt.text(0.5,0.5,'No Tenure Bucket',ha='center')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150)
plt.show()

# ---------- 4. Preprocess for ANN ----------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

X_processed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()
joblib.dump(feature_names, 'feature_columns.pkl')   # optional

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"Number of features after encoding: {X_train.shape[1]}")

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# ---------- 5. Build and train ANN (slightly larger capacity) ----------
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=70,
                    batch_size=256,
                    class_weight=class_weight_dict,
                    callbacks=[early_stop, reduce_lr],
                    verbose=1)

# ---------- 6. Evaluate ----------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1,6), yticklabels=range(1,6))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Plot training history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over epochs')
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

# ---------- 7. Save artifacts ----------
model.save('ann_model.h5')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(num_medians, 'num_medians.pkl')
joblib.dump(cat_modes, 'cat_modes.pkl')
joblib.dump(numerical_cols, 'numerical_cols.pkl')
joblib.dump(categorical_cols, 'categorical_cols.pkl')
print("\nAll artifacts saved.")

# ---------- 8. Test prediction ----------
def predict_csat(input_dict):
    new_data = pd.DataFrame([input_dict])
    for col in categorical_cols:
        if col not in new_data.columns:
            new_data[col] = cat_modes[col] if col in cat_modes.index else 'Unknown'
        else:
            new_data[col] = new_data[col].fillna(cat_modes[col] if col in cat_modes.index else 'Unknown').astype(str)
    for col in numerical_cols:
        if col not in new_data.columns:
            new_data[col] = num_medians[col]
        else:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce').fillna(num_medians[col])
    new_processed = preprocessor.transform(new_data)
    prob = model.predict(new_processed, verbose=0)
    pred = np.argmax(prob, axis=1)[0] + 1
    return pred

sample = X.iloc[0].to_dict()
print("\nSample prediction on first row:", predict_csat(sample))
print("\nConfidence on this sample:", np.max(model.predict(preprocessor.transform(pd.DataFrame([sample])), verbose=0)) * 100)

print("\n" + "="*60)
print("Training complete. To launch the Streamlit app, run:")
print("streamlit run ecom.py -- --mode app")
print("="*60)