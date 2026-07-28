import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Analisis Sentimen BiLSTM", layout="wide")

TOTAL_STEPS = 5

def init_state():
    defaults = {
        'step': 1,
        'df_raw': None,         
        'df_preprocessed': None,
        'df_labeled': None,
        'X_features': None,     
        'y_labels': None,       
        'training_result': None,
        'epoch_logs': None,    
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def ganti_halaman(nomor):
    st.session_state.step = nomor

st.markdown("""
<style>
    .main-title {
        text-align: center; 
        font-size: 32px; 
        font-weight: 800;
        color: var(--text-color); 
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center; 
        font-size: 18px;
        color: var(--text-color);
        opacity: 0.7; 
        margin-bottom: 30px;
    }
    div.stButton > button {
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
        font-weight: 600;
        transition: 0.3s;
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">ANALISIS SENTIMEN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">BiLSTM Deep Learning</div>', unsafe_allow_html=True)

step_labels = ["Upload", "Preprocess", "Embedding", "Training", "Evaluasi"]
nav_cols = st.columns(TOTAL_STEPS)

for i in range(TOTAL_STEPS):
    target = i + 1
    with nav_cols[i]:
        btn_type = "primary" if st.session_state.step == target else "secondary"
        if st.button(step_labels[i], key=f"btn_{target}", use_container_width=True, type=btn_type):
            ganti_halaman(target)
            st.rerun()

st.markdown("---")

if st.session_state.step == 1:
    st.subheader("Upload Dataset")
    st.write("Pilih file CSV yang berisi data teks untuk analisis sentimen.")

    file = st.file_uploader("Upload file CSV", type=["csv"], label_visibility="collapsed")

    if file is not None:
        try:
            df = pd.read_csv(file)

            if 'full_text' not in df.columns:
                st.error("Kolom `full_text` tidak ditemukan. Pastikan CSV kamu memiliki kolom bernama `full_text`.")
            else:
                st.session_state.df_raw = df
                st.success(f"Dataset berhasil di-upload! Total baris: **{len(df):,}**")

                st.metric("Total Baris", f"{len(df):,}")

                st.markdown("**Preview data (10 baris pertama):**")
                st.dataframe(df[['full_text']].head(10), use_container_width=True)

                st.button(
                    "Lanjut ke Preprocessing ",
                    on_click=ganti_halaman, args=(2,),
                    type="primary"
                )
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

    elif st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.info("Dataset sudah di-upload sebelumnya.")
        
        st.metric("Total Baris", f"{len(df):,}")
        st.markdown("**Preview data (10 baris pertama):**")
        st.dataframe(df[['full_text']].head(10), use_container_width=True)
        
        st.button("Lanjut ke Preprocessing ", on_click=ganti_halaman, args=(2,), type="primary")

elif st.session_state.step == 2:
    st.subheader("Preprocessing & Pelabelan Data")
    st.write("Proses: **Case Folding → Hapus URL → Cleansing → Normalisasi → Filter** → **Pelabelan InSet**")

    if st.session_state.df_raw is None:
        st.warning("Belum ada dataset. Silakan upload terlebih dahulu.")
        st.button("Kembali ke Upload", on_click=ganti_halaman, args=(1,))
    else:
        df_raw = st.session_state.df_raw

        st.markdown("#### Tahap 1: Preprocessing Teks")

        if st.session_state.df_preprocessed is not None:
            df_pre = st.session_state.df_preprocessed
            col1, col2, col3 = st.columns(3)
            col1.metric("Data Awal", f"{len(df_raw):,}")
            col2.metric("Data Bersih", f"{len(df_pre):,}")
            col3.metric("Data Dihapus", f"{len(df_raw) - len(df_pre):,}")
            st.success("Preprocessing selesai.")
            
            st.markdown("**Preview hasil preprocessing:**")
            st.dataframe(df_pre[['clean_text']].head(10), use_container_width=True)
        else:
            if st.button("Jalankan Preprocessing"):
                from modules.preprocessing import run_preprocessing

                status_text  = st.empty()
                progress_bar = st.progress(0)

                def update_progress(step, total, pesan):
                    pct = int((step / total) * 100) if total > 0 else 0
                    progress_bar.progress(pct)
                    status_text.info(f"**Tahap {step}/{total}:** {pesan}")

                try:
                    df_result = run_preprocessing(df_raw, progress_callback=update_progress)
                    st.session_state.df_preprocessed = df_result
                    progress_bar.progress(100)
                    status_text.empty()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Data Awal", f"{len(df_raw):,}")
                    col2.metric("Data Bersih", f"{len(df_result):,}")
                    col3.metric("Data Dihapus", f"{len(df_raw) - len(df_result):,}")
                    st.success("Preprocessing selesai!")
                    
                    st.markdown("**Preview hasil preprocessing:**")
                    st.dataframe(df_result[['clean_text']].head(10), use_container_width=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saat preprocessing: {e}")

        if st.session_state.df_preprocessed is not None:
            st.markdown("---")
            st.markdown("#### Tahap 2: Pelabelan Sentimen (Lexicon InSet)")
            st.write("Setiap teks diberi label **Positif / Negatif / Netral** berdasarkan skor lexicon InSet.")

            if st.session_state.df_labeled is not None:
                df_labeled = st.session_state.df_labeled
                dist = df_labeled['label'].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Positif", int(dist.get('Positif', 0)))
                c2.metric("Negatif", int(dist.get('Negatif', 0)))
                c3.metric("Netral",  int(dist.get('Netral',  0)))
                st.success("Pelabelan selesai.")
                
                st.markdown("**Preview hasil pelabelan:**")
                st.dataframe(
                    st.session_state.df_labeled[['clean_text', 'skor_positif', 'skor_negatif', 'total_skor', 'label']].head(10),
                    use_container_width=True
                )
                st.button("Lanjut ke Embedding ", on_click=ganti_halaman, args=(3,), type="primary")
            else:
                if st.button("Jalankan Pelabelan InSet"):
                    from modules.labeling import run_labeling

                    status_lb = st.empty()
                    prog_lb   = st.progress(0)

                    def cb_label(step, total, pesan):
                        prog_lb.progress(int(step / total * 100) if total else 0)
                        status_lb.info(pesan)

                    try:
                        df_labeled = run_labeling(st.session_state.df_preprocessed, progress_callback=cb_label)
                        st.session_state.df_labeled = df_labeled
                        prog_lb.progress(100)
                        status_lb.empty()

                        dist = df_labeled['label'].value_counts()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Positif", int(dist.get('Positif', 0)))
                        c2.metric("Negatif", int(dist.get('Negatif', 0)))
                        c3.metric("Netral",  int(dist.get('Netral',  0)))
                        st.success("Pelabelan selesai!")
                        
                        st.markdown("**Preview hasil pelabelan:**")
                        st.dataframe(
                            df_labeled[['clean_text', 'skor_positif', 'skor_negatif', 'total_skor', 'label']].head(10),
                            use_container_width=True
                        )
                        st.button("Lanjut ke Embedding ", on_click=ganti_halaman, args=(3,), type="primary")
                    except Exception as e:
                        st.error(f"Error pelabelan: {e}")

elif st.session_state.step == 3:
    st.subheader("Ekstraksi Fitur DistilBERT")
    st.write(
        "Model `cahya/distilbert-base-indonesian`, mengubah kata menjadi vektor."
    )

    if st.session_state.df_labeled is None:
        st.warning("Pelabelan belum selesai. Kembali ke halaman Preprocess.")
        st.button("Kembali ke Preprocess", on_click=ganti_halaman, args=(2,))
    else:
        df_labeled = st.session_state.df_labeled

        if st.session_state.X_features is not None:
            X = st.session_state.X_features
            st.success(f"Embedding sudah selesai. Shape matriks: **{X.shape}**")
            st.button("Lanjut ke Training ", on_click=ganti_halaman, args=(4,), type="primary")
        else:
            st.info(f"Memproses **{len(df_labeled):,}** teks.")
            if st.button("▶ Jalankan Ekstraksi Fitur DistilBERT"):
                from modules.embedding import extract_features

                teks_list  = df_labeled['clean_text'].tolist()
                status_emb = st.empty()
                prog_emb   = st.progress(0)

                def cb_emb(step, total, pesan):
                    prog_emb.progress(int(step / total * 100) if total else 0)
                    status_emb.info(pesan)

                try:
                    X = extract_features(teks_list, progress_callback=cb_emb)
                    y = df_labeled['label'].values

                    st.session_state.X_features = X
                    st.session_state.y_labels   = y

                    prog_emb.progress(100)
                    status_emb.empty()

                    st.success(f"Shape matriks: **{X.shape}**")
                    st.button("Lanjut ke Training ", on_click=ganti_halaman, args=(4,), type="primary")
                except Exception as e:
                    st.error(f"Error embedding: {e}")

elif st.session_state.step == 4:
    st.subheader("Training Model BiLSTM")
    if st.session_state.X_features is None:
        st.warning("Embedding belum selesai.")
        st.button("Kembali ke Embedding", on_click=ganti_halaman, args=(3,))
    else:
        X = st.session_state.X_features
        y = st.session_state.y_labels

        epochs = 7
        batch_size = 32
        lr = 0.0001

        if st.session_state.training_result is not None:
            if st.session_state.epoch_logs:
                st.code('\n'.join(st.session_state.epoch_logs))

            result = st.session_state.training_result
            hist   = result['history']
            last_e = len(hist['train_loss'])
            
            st.success(
                f"Training sudah selesai ({last_e} epoch). "
                f"Val Acc terakhir: **{hist['test_acc'][-1]:.4f}**"
            )
            
            col_lanjut, col_ulang = st.columns(2)
            with col_lanjut:
                st.button("Lanjut ke Evaluasi", on_click=ganti_halaman, args=(5,), type="primary", use_container_width=True)
            with col_ulang:
                if st.button("Train Ulang", use_container_width=True):
                    st.session_state.training_result = None
                    st.session_state.epoch_logs = None
                    st.rerun()

        else:
            if st.button("Mulai Training Model"):
                from modules.bilstm import run_training

                log_area   = st.empty()
                prog_train = st.progress(0)
                epoch_logs = []

                def cb_epoch(epoch, total, log_str):
                    epoch_logs.append(log_str)
                    prog_train.progress(int(epoch / total * 100))
                    log_area.code('\n'.join(epoch_logs))

                try:
                    result = run_training(
                        X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=lr,
                        epoch_callback=cb_epoch
                    )
                    
                    st.session_state.training_result = result
                    st.session_state.epoch_logs = epoch_logs
                    
                    prog_train.progress(100)
                    st.success("Training selesai!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error saat training: {e}")

elif st.session_state.step == 5:
    st.markdown('<p class="main-title">Hasil Evaluasi Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Analisis performa klasifikasi sentimen BiLSTM</p>', unsafe_allow_html=True)

    if st.session_state.training_result is None:
        st.warning("Model belum dilatih.")
        st.button("Kembali ke Training", on_click=ganti_halaman, args=(4,))
    else:
        result      = st.session_state.training_result
        report      = result['report']
        cm          = result['confusion_matrix']
        class_names = result['class_names']
        df_labeled  = st.session_state.df_labeled

        col_kiri, col_kanan = st.columns([1, 1.3])

        with col_kiri:
            st.markdown("### **Proporsi Sentimen**")
            if df_labeled is not None:
                dist = df_labeled['label'].value_counts().reset_index()
                dist.columns = ['Label', 'Jumlah']
                color_map = {'Positif': '#10b981', 'Negatif': '#ef4444', 'Netral': '#94a3b8'}
                fig_pie = px.pie(
                    dist, values='Jumlah', names='Label',
                    color='Label', color_discrete_map=color_map,
                    hole=0.4
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='gray')
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_kanan:
            st.markdown("### **Performance evaluation**")
            
            report_data = []
            
            for cls in class_names:
                if cls in report:
                    r = report[cls]
                    report_data.append({
                        'Kelas / Avg': cls,
                        'Precision': f"{r['precision']:.4f}",
                        'Recall': f"{r['recall']:.4f}",
                        'F1-Score': f"{r['f1-score']:.4f}",
                        'Support': str(int(r['support']))
                    })
            
            report_data.append({
                'Kelas / Avg': 'accuracy',
                'Precision': '',
                'Recall': '',
                'F1-Score': f"{report['accuracy']:.4f}",
                'Support': str(int(report['macro avg']['support']))
            })
            
            report_data.append({
                'Kelas / Avg': 'macro avg',
                'Precision': f"{report['macro avg']['precision']:.4f}",
                'Recall': f"{report['macro avg']['recall']:.4f}",
                'F1-Score': f"{report['macro avg']['f1-score']:.4f}",
                'Support': str(int(report['macro avg']['support']))
            })
            
            report_data.append({
                'Kelas / Avg': 'weighted avg',
                'Precision': f"{report['weighted avg']['precision']:.4f}",
                'Recall': f"{report['weighted avg']['recall']:.4f}",
                'F1-Score': f"{report['weighted avg']['f1-score']:.4f}",
                'Support': str(int(report['weighted avg']['support']))
            })

            st.table(pd.DataFrame(report_data))

            st.markdown("### **Confusion Matrix**")
            df_cm = pd.DataFrame(
                cm,
                index=[f"Actual {c}" for c in class_names],
                columns=[f"Pred {c}" for c in class_names]
            )
            st.dataframe(df_cm, use_container_width=True)

        st.markdown("---")
        st.markdown("### **Kurva Training**")
        hist     = result['history']
        ep_range = list(range(1, len(hist['train_loss']) + 1))

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=ep_range, y=hist['train_loss'], name='Train Loss', line=dict(color='blue')))
        fig_loss.add_trace(go.Scatter(x=ep_range, y=hist['test_loss'],  name='Val Loss',  line=dict(color='red')))
        fig_loss.update_layout(title='Kurva Loss Model BiLSTM', xaxis_title='Epoch', yaxis_title='Loss')

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=ep_range, y=hist['train_acc'], name='Train Acc', line=dict(color='blue')))
        fig_acc.add_trace(go.Scatter(x=ep_range, y=hist['test_acc'],  name='Val Acc',  line=dict(color='red')))
        fig_acc.update_layout(title='Kurva Akurasi Model BiLSTM', xaxis_title='Epoch', yaxis_title='Akurasi')

        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_loss, use_container_width=True)
        c2.plotly_chart(fig_acc,  use_container_width=True)

        st.markdown("---")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import io

        df_labeled = st.session_state.df_labeled
        kolom_teks = 'clean_text' if 'clean_text' in df_labeled.columns else df_labeled.columns[0]
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(st.session_state.y_labels)
        
        train_idx, test_idx = train_test_split(
            range(len(df_labeled)),
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
        
        df_test = df_labeled.iloc[test_idx]
        teks_uji = df_test[kolom_teks].values
        y_test_labels = [class_names[int(i)] for i in result['y_test']]
        y_pred_labels = [class_names[int(i)] for i in result['y_pred']]
        
        df_excel = pd.DataFrame({
            'Teks': teks_uji,
            'Hasil Prediksi': y_pred_labels,
            'Hasil Aktual': y_test_labels
        })
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_excel.to_excel(writer, index=False, sheet_name='Hasil Akhir Evaluasi')
        
        st.markdown("""
        <style>
            .div-tombol-berdampingan {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            .div-tombol-berdampingan div {
                display: inline-block;
                width: auto !important;
            }
            .div-tombol-berdampingan button {
                width: auto !important;
                padding: 6px 16px !important;
                font-size: 14px !important;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="div-tombol-berdampingan">', unsafe_allow_html=True)
        
        st.download_button(
            label="Unduh Hasil Prediksi (Excel)",
            data=buffer.getvalue(),
            file_name="hasil_analisis_sentimen_bilstm.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if st.button("Mulai Analisis Baru"):
            for k in ['df_raw', 'df_preprocessed', 'df_labeled', 'X_features', 'y_labels', 'training_result', 'epoch_logs']:
                st.session_state[k] = None
            ganti_halaman(1)
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)