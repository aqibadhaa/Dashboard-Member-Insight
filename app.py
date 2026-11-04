import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN & WARNA BRAND GDGoC
# ==========================================
st.set_page_config(
    page_title="GDGoC UG 2026/2027 — Member Survey Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

GDG_COLORS = ["#4285F4", "#34A853", "#FBBC04", "#EA4335", "#9AA0A6", "#185ABC"]

# ==========================================
# 2. DEFINISI OPSI KANONIKAL (ANTI JEBAKAN KOMA)
# ==========================================
EXPECTATION_OPTIONS = [
    "Learn the fundamentals of a tech field",
    "Improve my skills through classes and workshops",
    "Build real projects for my portfolio",
    "Hear from industry practitioners",
    "Get mentoring and feedback",
    "Join competitions",
    "Meet and collaborate with other members",
]

LEARNING_STYLE_OPTIONS = [
    "Workshop (practice together)",
    "Class (step-by-step lessons)",
    "Self-paced online learning (videos, modules)",
    "Talk or sharing session",
    "Mentoring",
    "Building a project with a team",
]

PATH_OPTIONS = [
    "Web Development",
    "Machine Learning / AI",
    "UI/UX Design",
    "Product Management",
    "Mobile Development",
]

SKILL_OPTIONS_BY_PATH = {
    "UI/UX Design": [
        "Make wireframes and user flows",
        "Design high-fidelity screens in Figma",
        "Build interactive prototypes",
        "Use components, auto layout, and design systems",
        "Run user research or usability tests",
        "None of these yet",
    ],
    "Web Development": [
        "Build pages with HTML and CSS",
        "Add interactivity with JavaScript",
        "Use a frontend framework (React, Vue, etc.)",
        "Build a backend or REST API with a database",
        "Deploy a web app online",
        "None of these yet",
    ],
    "Mobile Development": [
        "Build basic screens (Jetpack Compose, XML, or Flutter)",
        "Handle navigation and state",
        "Fetch data from a REST API",
        "Store data locally or with Firebase",
        "Build a release APK or publish to Play Store",
        "None of these yet",
    ],
    "Product Management": [
        "Write user stories or a PRD",
        "Interview users or do market research",
        "Prioritize features (RICE, MoSCoW, etc.)",
        "Define product metrics or KPIs",
        "Run sprints with Jira, Trello, or Notion",
        "None of these yet",
    ],
    "Machine Learning / AI": [
        "Write Python and analyze data with Pandas",
        "Train models with scikit-learn",
        "Build deep learning models (TensorFlow or PyTorch)",
        "Build apps with LLM APIs (Gemini, etc.)",
        "Deploy a model (API, TFLite, cloud)",
        "None of these yet",
    ],
}

CLASS_CODE_TO_PRODI = {
    "IA": "Informatika",
    "KA": "Sistem Informasi",
    "IB": "Teknik Elektro",
    "ID": "Teknik Industri",
    "IC": "Teknik Mesin",
    "EB": "Akuntansi",
    "EA": "Manajemen",
    "PA": "Psikologi",
    "MA": "Ilmu Komunikasi",
    "SA": "Sastra Inggris",
    "DC": "Teknik Komputer (D3)",
}

# ==========================================
# 3. PIPELINE DATA CLEANING & SANITASI PII
# ==========================================
@st.cache_data
def load_and_clean_data(file_source):
    df = pd.read_csv(file_source)

    # 1. Pastikan kolom sensitif (PII) langsung dihapus jika masih ada
    sensitive_cols = ["Email address", "WhatsApp number", "Instagram username"]
    df = df.drop(
        columns=[c for c in sensitive_cols if c in df.columns],
        errors="ignore",
    )

    # 2. Hapus data dummy / testing tanpa bergantung pada kolom Email
    df = df[
        ~df["Full name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["asda", "test", "dummy"])
    ].copy()
    df = df[df["Full name"].astype(str).str.strip().str.len() > 2].copy()

    # 3. Standarisasi kode Kelas & ekstraksi Tingkat + Jurusan
    def normalize_class(val):
        val = str(val).upper().strip()
        val = re.sub(r"E\.?G\.?\s*", "", val).strip()
        if val == "1T7":
            return "1IA07"
        return val

    df["Class_Clean"] = df["Class"].apply(normalize_class)
    df["Tingkat"] = (
        df["Class_Clean"]
        .str.extract(r"^([1-4])")[0]
        .map(
            {
                "1": "Tingkat 1 (Maba)",
                "2": "Tingkat 2",
                "3": "Tingkat 3",
                "4": "Tingkat 4",
            }
        )
        .fillna("Lainnya")
    )
    df["Kode_Jurusan"] = df["Class_Clean"].str.extract(r"^[1-4]([A-Z]{2})")[0]
    df["Program_Studi"] = (
        df["Kode_Jurusan"]
        .map(CLASS_CODE_TO_PRODI)
        .fillna(df["Study program"].astype(str).str.title())
    )

    # 4. Gabungkan 5 pasangan kolom percabangan per Main Path
    skill_col_map = {
        "UI/UX Design": "What can you already do?",
        "Web Development": "What can you already do?.1",
        "Mobile Development": "What can you already do?.2",
        "Product Management": "What can you already do?.3",
        "Machine Learning / AI": "What can you already do?.4",
    }
    learn_col_map = {
        "UI/UX Design": "What do you want to learn in UI/UX Design?",
        "Web Development": "What do you want to learn in Web Development?",
        "Mobile Development": "What do you want to learn in Mobile Development?",
        "Product Management": "What do you want to learn in Product Management?",
        "Machine Learning / AI": "What do you want to learn in Machine Learning / AI?",
    }

    df["Current_Skills"] = df.apply(
        lambda r: r.get(skill_col_map.get(r["Which one is your main path?"], ""), np.nan),
        axis=1,
    )
    df["Want_To_Learn"] = df.apply(
        lambda r: r.get(learn_col_map.get(r["Which one is your main path?"], ""), np.nan),
        axis=1,
    )

    # 5. Hitung jumlah skill yang sudah dikuasai & status Pemula Murni
    def count_member_skills(row):
        path = row["Which one is your main path?"]
        skills_text = str(row["Current_Skills"])
        if skills_text.strip() == "None of these yet":
            return 0
        opts = [
            o for o in SKILL_OPTIONS_BY_PATH.get(path, []) if o != "None of these yet"
        ]
        return sum(1 for o in opts if o in skills_text)

    df["Skill_Count"] = df.apply(count_member_skills, axis=1)
    df["Level_Kategori"] = np.where(
        df["Skill_Count"] == 0,
        "Pemula Murni (0 Skill)",
        np.where(
            df["Skill_Count"] <= 2,
            "Basic (1-2 Skill)",
            "Intermediate/Advanced (3-5 Skill)",
        ),
    )

    return df


def count_canonical_options(series, canonical_list):
    results = []
    total_resp = len(series.dropna())
    for opt in canonical_list:
        cnt = series.dropna().apply(lambda x: opt in str(x)).sum()
        pct = (cnt / total_resp * 100) if total_resp > 0 else 0.0
        results.append({"Opsi": opt, "Jumlah": cnt, "Persentase": round(pct, 1)})
    return pd.DataFrame(results).sort_values("Jumlah", ascending=True)


def insight_card(points: list[tuple[str, str]]):
    """Render insight cards. Each point is a tuple of (emoji_level, text).
    emoji_level: 'info' | 'warn' | 'danger'
    """
    color_map = {
        "info":   ("#E8F4FD", "#1A73E8", "ℹ️"),
        "warn":   ("#FEF7E0", "#F9AB00", "⚠️"),
        "danger": ("#FCE8E6", "#D93025", "🔴"),
    }
    cards_html = ""
    for level, text in points:
        bg, border, icon = color_map.get(level, color_map["info"])
        cards_html += f"""
        <div style="
            background:{bg};
            border-left: 4px solid {border};
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 8px;
            font-size: 0.9rem;
            line-height: 1.5;
        ">{icon} {text}</div>
        """
    st.markdown(
        f'<div style="margin: 16px 0 20px 0">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def detect_mismatch(df_input):
    """Deteksi member yang ingin mempelajari skill yang sudah mereka kuasai."""
    rows = []
    for _, row in df_input.iterrows():
        path = row["Which one is your main path?"]
        skills_text = str(row.get("Current_Skills", ""))
        want_text   = str(row.get("Want_To_Learn", ""))
        if want_text in ("nan", "") or skills_text in ("nan", ""):
            continue
        opts = [o for o in SKILL_OPTIONS_BY_PATH.get(path, []) if o != "None of these yet"]
        already_have = [o for o in opts if o in skills_text]
        if not already_have:
            continue
        overlap = [o for o in already_have if o.lower()[:15] in want_text.lower()]
        if overlap:
            rows.append({
                "Nama": row["Full name"],
                "Main Path": path,
                "Tingkat": row["Tingkat"],
                "Skill Sudah Dikuasai": ", ".join(overlap),
                "Topik yang Ingin Dipelajari": want_text,
            })
    return pd.DataFrame(rows)


# ==========================================
# 4. SIDEBAR & PEMUATAN DATA
# ==========================================
DEFAULT_CSV = "gdgoc member survey clean.csv"

st.sidebar.title("🎛️ Filter & Kontrol")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Baru (Opsional)", type=["csv"]
)

try:
    source = uploaded_file if uploaded_file is not None else DEFAULT_CSV
    df_all = load_and_clean_data(source)
except FileNotFoundError:
    st.error(
        f"File `{DEFAULT_CSV}` belum ditemukan di direktori project. Pastikan file CSV yang sudah disanitasi berada satu folder dengan `app.py`."
    )
    st.stop()

# Filter global di Sidebar
selected_paths = st.sidebar.multiselect(
    "Filter Main Path:",
    options=PATH_OPTIONS,
    default=PATH_OPTIONS,
)
selected_years = st.sidebar.multiselect(
    "Filter Tingkat Angkatan:",
    options=sorted(df_all["Tingkat"].unique()),
    default=sorted(df_all["Tingkat"].unique()),
)

df = df_all[
    (df_all["Which one is your main path?"].isin(selected_paths))
    & (df_all["Tingkat"].isin(selected_years))
].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data sensitif (Email, WA, IG) telah dihapus.\n\nMenampilkan **{len(df)}** dari **{len(df_all)}** member valid."
)

# ==========================================
# 5. HEADER & KPI CARDS UTAMA
# ==========================================
st.title("GDGoC Universitas Gunadarma 2026/2027 Member Insights")
st.markdown(
    "Dashboard analitik terpadu untuk **Core Team & Supervisor (Tech & Non-Tech)** berdasarkan survei registrasi member baru."
)

total_m = len(df)
offline_ready = (
    df["Online or offline?"].isin(["Offline (on campus)", "Either is fine"]).sum()
)
high_commit = (
    df["If we open a class for your main path, how often would you join?"]
    .isin(["Every session", "Most sessions"])
    .sum()
)
low_spec_cnt = (df["Can you bring a laptop to on-campus sessions?"] != "Yes").sum()
mentor_ready = (
    df["Would you like to help teach or mentor in your main path?"] == "Yes"
).sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Member Valid", f"{total_m} Orang")
col2.metric(
    "Siap Hadir Offline",
    f"{offline_ready} ({offline_ready/max(total_m,1)*100:.1f}%)",
)
col3.metric(
    "Komitmen Tinggi",
    f"{high_commit} ({high_commit/max(total_m,1)*100:.1f}%)",
)
col4.metric(
    "Kandidat Peer Mentor",
    f"{mentor_ready} ({mentor_ready/max(total_m,1)*100:.1f}%)",
)
col5.metric(
    "Laptop Low-Spec / Tanpa Laptop",
    f"{low_spec_cnt} ({low_spec_cnt/max(total_m,1)*100:.1f}%)",
)

st.markdown("---")

# ==========================================
# 6. TABS ARSITEKTUR DASHBOARD
# ==========================================
tab_ops, tab_curr, tab_mentor, tab_voice = st.tabs(
    [
        " 1. Event & Operasional (Non-Tech)",
        " 2. Kurikulum & Skill Matrix (Tech)",
        " 3. Kandidat Peer Mentor & Direktori",
        " 4. Aspirasi & Saran Member",
    ]
)

# ------------------------------------------
# TAB 1: EVENT & OPERASIONAL (NON-TECH)
# ------------------------------------------
with tab_ops:
    st.subheader("Profil Demografi & Kesiapan Operasional Event")
    c1, c2 = st.columns(2)

    with c1:
        path_cnt = (
            df["Which one is your main path?"]
            .value_counts()
            .reset_index()
        )
        path_cnt.columns = ["Main Path", "Jumlah"]
        fig_path = px.bar(
            path_cnt,
            x="Jumlah",
            y="Main Path",
            orientation="h",
            text="Jumlah",
            color="Main Path",
            color_discrete_sequence=GDG_COLORS,
            title="Distribusi Pilihan Main Path",
        )
        fig_path.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_path, use_container_width=True)

    with c2:
        year_prodi = (
            df.groupby(["Tingkat", "Program_Studi"])
            .size()
            .reset_index(name="Jumlah")
        )
        fig_demo = px.bar(
            year_prodi,
            x="Tingkat",
            y="Jumlah",
            color="Program_Studi",
            title="Sebaran Tingkat Angkatan & Program Studi",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(fig_demo, use_container_width=True)

    c3, c4, c5 = st.columns(3)
    with c3:
        mode_cnt = df["Online or offline?"].value_counts().reset_index()
        mode_cnt.columns = ["Mode", "Jumlah"]
        fig_mode = px.pie(
            mode_cnt,
            names="Mode",
            values="Jumlah",
            hole=0.45,
            color_discrete_sequence=GDG_COLORS,
            title="Preferensi Mode Pelaksanaan Kelas",
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    with c4:
        lap_cnt = (
            df["Can you bring a laptop to on-campus sessions?"]
            .value_counts()
            .reset_index()
        )
        lap_cnt.columns = ["Status Laptop", "Jumlah"]
        fig_lap = px.pie(
            lap_cnt,
            names="Status Laptop",
            values="Jumlah",
            hole=0.45,
            color_discrete_sequence=["#34A853", "#FBBC04", "#EA4335"],
            title="Kesiapan Membawa Laptop ke Kampus",
        )
        st.plotly_chart(fig_lap, use_container_width=True)

    with c5:
        comm_cnt = (
            df["If we open a class for your main path, how often would you join?"]
            .value_counts()
            .reset_index()
        )
        comm_cnt.columns = ["Komitmen", "Jumlah"]
        fig_comm = px.bar(
            comm_cnt,
            x="Komitmen",
            y="Jumlah",
            text="Jumlah",
            color="Komitmen",
            color_discrete_sequence=GDG_COLORS,
            title="Komitmen Kehadiran Kelas Rutin",
        )
        fig_comm.update_layout(showlegend=False)
        st.plotly_chart(fig_comm, use_container_width=True)

    # Metode Belajar Favorit
    style_df = count_canonical_options(
        df["How do you like to learn?"], LEARNING_STYLE_OPTIONS
    )
    fig_style = px.bar(
        style_df,
        x="Jumlah",
        y="Opsi",
        orientation="h",
        text=style_df.apply(lambda r: f"{r['Jumlah']} ({r['Persentase']}%)", axis=1),
        title="Format Kegiatan Belajar yang Paling Diminati Member",
        color_discrete_sequence=["#34A853"],
    )
    st.plotly_chart(fig_style, use_container_width=True)

    # --- INSIGHT CARDS TAB 1 ---
    st.markdown("#### Catatan Operasional")
    _t1 = len(df)
    _offline  = df["Online or offline?"].isin(["Offline (on campus)", "Either is fine"]).sum()
    _commit   = df["If we open a class for your main path, how often would you join?"].isin(["Every session", "Most sessions"]).sum()
    _low_spec = (df["Can you bring a laptop to on-campus sessions?"] != "Yes").sum()
    _path_top = df["Which one is your main path?"].value_counts()
    _maba_n   = (df["Tingkat"] == "Tingkat 1 (Maba)").sum()
    _maba_pct = round(_maba_n / max(_t1, 1) * 100, 1)
    _low_pct  = round(_low_spec / max(_t1, 1) * 100, 1)
    _commit_pct = round(_commit / max(_t1, 1) * 100, 1)

    _top_path_name = _path_top.idxmax()
    _top_path_n    = _path_top.iloc[0]
    _bot_path_name = _path_top.idxmin()
    _bot_path_n    = _path_top.iloc[-1]

    _t1_points = []

    # Dominasi Maba
    if _maba_pct >= 50:
        _t1_points.append(("warn",
            f"Sebanyak {_maba_n} dari {_t1} member ({_maba_pct}%) adalah Tingkat 1. "
            "Mayoritas belum punya pengalaman organisasi teknis, jadi onboarding dan orientasi awal perlu diprioritaskan sebelum kelas dimulai."
        ))

    # Path tidak merata
    _ratio = _top_path_n / max(_bot_path_n, 1)
    if _ratio >= 5:
        _t1_points.append(("warn",
            f"{_top_path_name} mendominasi dengan {_top_path_n} member, "
            f"sementara {_bot_path_name} hanya {_bot_path_n} orang. "
            "Alokasi fasilitator dan ruangan perlu menyesuaikan ketimpangan ini agar kelas tidak overload di satu divisi."
        ))

    # Komitmen tinggi
    if _commit_pct >= 75:
        _t1_points.append(("info",
            f"{_commit} member ({_commit_pct}%) menyatakan komitmen tinggi untuk hadir setiap sesi atau hampir setiap sesi. "
            "Angka ini cukup untuk menjustifikasi jadwal kelas rutin, bukan hanya event sporadis."
        ))

    # Laptop issue
    if _low_pct >= 25:
        _t1_points.append(("danger",
            f"{_low_spec} member ({_low_pct}%) tidak bisa membawa laptop ke kampus atau hanya membawa perangkat spesifikasi rendah. "
            "Kelas yang butuh coding langsung perlu menyediakan alternatif seperti cloud IDE atau sesi pair-programming."
        ))
    elif _low_pct >= 15:
        _t1_points.append(("warn",
            f"{_low_spec} member ({_low_pct}%) memiliki keterbatasan laptop. "
            "Pertimbangkan sesi praktik berbasis cloud untuk memastikan semua bisa ikut."
        ))

    if not _t1_points:
        _t1_points.append(("info", "Tidak ada anomali operasional signifikan pada segmen yang dipilih."))

    insight_card(_t1_points)

# ------------------------------------------
# TAB 2: KURIKULUM & SKILL MATRIX (TECH)
# ------------------------------------------
with tab_curr:
    st.subheader("Pemetaan Kemampuan Awal & Kebutuhan Kurikulum per Path")

    lvl_path = (
        df.groupby(["Which one is your main path?", "Level_Kategori"])
        .size()
        .reset_index(name="Jumlah")
    )
    fig_lvl = px.bar(
        lvl_path,
        x="Which one is your main path?",
        y="Jumlah",
        color="Level_Kategori",
        barmode="group",
        text="Jumlah",
        color_discrete_sequence=["#EA4335", "#FBBC04", "#34A853"],
        title="Perbandingan Level Penguasaan Awal di Setiap Path",
    )
    st.plotly_chart(fig_lvl, use_container_width=True)

    # --- CROSS-TAB PATH x TINGKAT x LEVEL ---
    st.markdown("#### Sebaran Level per Path dan Angkatan")
    st.caption(
        "Tabel ini menunjukkan berapa member di setiap kombinasi Path, Tingkat Angkatan, dan Level Skill. "
        "Gunakan ini untuk merancang kelas yang levelnya sesuai dengan segmen yang ada."
    )
    cross_tab = (
        df.groupby(["Which one is your main path?", "Tingkat", "Level_Kategori"])
        .size()
        .reset_index(name="Jumlah")
        .rename(columns={
            "Which one is your main path?": "Main Path",
            "Level_Kategori": "Level Skill",
        })
    )
    # Pivot supaya mudah dibaca
    cross_pivot = cross_tab.pivot_table(
        index=["Main Path", "Tingkat"],
        columns="Level Skill",
        values="Jumlah",
        fill_value=0,
    ).reset_index()
    st.dataframe(cross_pivot, use_container_width=True, hide_index=True)

    # --- MISMATCH ALERT ---
    st.markdown("#### Potensi Mismatch: Skill Sudah Ada tapi Masih Ingin Dipelajari Lagi")
    st.caption(
        "Member di bawah ini menuliskan topik yang sebenarnya sudah mereka centang di bagian skill yang dikuasai. "
        "Ini bisa jadi sinyal bahwa mereka ingin pendalaman lebih lanjut, atau ada kebingungan saat mengisi survei. "
        "Pertimbangkan placement check sebelum kelas dimulai."
    )
    mismatch_df = detect_mismatch(df)
    if mismatch_df.empty:
        st.success("Tidak ada mismatch terdeteksi pada segmen yang dipilih.")
    else:
        st.dataframe(mismatch_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Telusuri Detail per Path")
    chosen_path = st.selectbox(
        "Pilih Path untuk melihat detail kompetensi dan request materi:",
        options=PATH_OPTIONS,
    )
    df_path = df[df["Which one is your main path?"] == chosen_path]

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        skill_df = count_canonical_options(
            df_path["Current_Skills"], SKILL_OPTIONS_BY_PATH[chosen_path]
        )
        fig_sk = px.bar(
            skill_df,
            x="Jumlah",
            y="Opsi",
            orientation="h",
            text=skill_df.apply(
                lambda r: f"{r['Jumlah']} ({r['Persentase']}%)", axis=1
            ),
            title=f"Skill yang Sudah Dikuasai Member {chosen_path} (n={len(df_path)})",
            color_discrete_sequence=["#4285F4"],
        )
        st.plotly_chart(fig_sk, use_container_width=True)

    with col_b:
        st.markdown(f"**Topik yang Ingin Dipelajari di {chosen_path}:**")
        topics = (
            df_path[["Full name", "Tingkat", "Level_Kategori", "Want_To_Learn"]]
            .dropna(subset=["Want_To_Learn"])
            .rename(
                columns={
                    "Full name": "Nama",
                    "Level_Kategori": "Level Saat Ini",
                    "Want_To_Learn": "Topik yang Diharapkan",
                }
            )
        )
        st.dataframe(topics, use_container_width=True, height=350, hide_index=True)

    # --- INSIGHT CARDS TAB 2 ---
    st.markdown("#### Catatan Kurikulum")
    _pemula = df[df["Level_Kategori"] == "Pemula Murni (0 Skill)"]
    _pemula_per_path = _pemula.groupby("Which one is your main path?").size()
    _total_pemula = len(_pemula)
    _pemula_pct = round(_total_pemula / max(len(df), 1) * 100, 1)

    _inter = df[df["Level_Kategori"] == "Intermediate/Advanced (3-5 Skill)"]
    _inter_pct = round(len(_inter) / max(len(df), 1) * 100, 1)

    _t2_points = []

    # Pemula dominan
    if _pemula_pct >= 25:
        _worst_path = _pemula_per_path.idxmax() if not _pemula_per_path.empty else "-"
        _worst_n    = int(_pemula_per_path.max()) if not _pemula_per_path.empty else 0
        _t2_points.append(("warn",
            f"Total {_total_pemula} member ({_pemula_pct}%) belum menguasai satu skill pun di path mereka. "
            f"Terbanyak ada di {_worst_path} dengan {_worst_n} orang. "
            "Kelas untuk segmen ini perlu dimulai dari fondasi, bukan langsung ke topik lanjutan."
        ))

    # Path tertentu 100% pemula
    for path_name, pemula_n in _pemula_per_path.items():
        total_path = len(df[df["Which one is your main path?"] == path_name])
        if total_path > 0 and pemula_n / total_path >= 0.50:
            _t2_points.append(("danger",
                f"Di {path_name}, {pemula_n} dari {total_path} member ({round(pemula_n/total_path*100)}%) "
                "adalah Pemula Murni. Kurikulum untuk path ini perlu dirancang ulang dari nol jika belum ada modul dasar."
            ))

    # Ada intermediate yang cukup banyak
    if _inter_pct >= 20:
        _t2_points.append(("info",
            f"Sebanyak {len(_inter)} member ({_inter_pct}%) sudah di level Intermediate atau Advanced. "
            "Segmen ini bisa dipertimbangkan untuk jalur studi jam atau proyek kolaboratif, bukan hanya kelas reguler."
        ))

    # Mismatch ada
    if not mismatch_df.empty:
        _t2_points.append(("warn",
            f"Terdeteksi {len(mismatch_df)} member yang menuliskan topik yang sudah mereka kuasai sebagai hal yang ingin dipelajari. "
            "Disarankan ada sesi placement singkat di awal untuk memvalidasi level sebelum kelas dimulai."
        ))

    if not _t2_points:
        _t2_points.append(("info", "Tidak ada anomali kurikulum signifikan pada segmen yang dipilih."))

    insight_card(_t2_points)

# ------------------------------------------
# TAB 3: KANDIDAT PEER MENTOR & DIREKTORI
# ------------------------------------------
with tab_mentor:
    st.subheader("Daftar Kandidat Peer Mentor / Asisten Study Jam")
    st.markdown(
        "Menampilkan member yang menjawab **'Yes'** pada kesediaan membantu mengajar/mentoring, diurutkan berdasarkan jumlah *skill* yang sudah dikuasai."
    )

    mentors_df = df[
        df["Would you like to help teach or mentor in your main path?"] == "Yes"
    ].sort_values(
        by=["Skill_Count", "Which one is your main path?"],
        ascending=[False, True],
    )

    base_cols = [
        "Full name",
        "Class_Clean",
        "Program_Studi",
        "Which one is your main path?",
        "Skill_Count",
        "Current_Skills",
        "Online or offline?",
    ]

    st.dataframe(
        mentors_df[base_cols].rename(
            columns={
                "Full name": "Nama Lengkap",
                "Class_Clean": "Kelas",
                "Program_Studi": "Prodi",
                "Which one is your main path?": "Main Path",
                "Skill_Count": "Jml Skill",
                "Current_Skills": "Detail Skill Dikuasai",
                "Online or offline?": "Preferensi Mode",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- INSIGHT CARDS TAB 3 ---
    st.markdown("#### Catatan Mentor")
    _total_members = len(df)
    _total_mentors = len(mentors_df)
    _mentor_pct = round(_total_mentors / max(_total_members, 1) * 100, 1)
    _mentor_per_path = mentors_df.groupby("Which one is your main path?").size()
    _member_per_path = df.groupby("Which one is your main path?").size()

    _t3_points = []

    # Rasio mentor keseluruhan
    if _total_mentors == 0:
        _t3_points.append(("danger",
            "Tidak ada kandidat peer mentor pada segmen yang dipilih. "
            "Pertimbangkan pendekatan berbeda untuk pengadaan fasilitator kelas."
        ))
    elif _mentor_pct < 20:
        _t3_points.append(("warn",
            f"Hanya {_total_mentors} dari {_total_members} member ({_mentor_pct}%) yang bersedia membantu mengajar. "
            "Jumlah ini perlu dipantau, terutama jika kelas dijadwalkan paralel di beberapa divisi sekaligus."
        ))
    else:
        _t3_points.append(("info",
            f"Ada {_total_mentors} kandidat peer mentor ({_mentor_pct}% dari total member). "
            "Jumlah ini cukup untuk mendukung kelas rutin jika distribusi per path merata."
        ))

    # Cek path yang kekurangan mentor
    for path_name in df["Which one is your main path?"].unique():
        m_count = int(_mentor_per_path.get(path_name, 0))
        total_path = int(_member_per_path.get(path_name, 0))
        if total_path >= 5 and m_count == 0:
            _t3_points.append(("danger",
                f"{path_name} memiliki {total_path} member tetapi belum ada kandidat peer mentor. "
                "Divisi ini perlu fasilitator eksternal atau rekrutmen mentor tambahan."
            ))
        elif total_path >= 10 and m_count <= 2:
            _t3_points.append(("warn",
                f"{path_name} hanya punya {m_count} kandidat mentor untuk {total_path} member. "
                "Rasio ini terlalu tipis jika kelas dibagi per angkatan atau level."
            ))

    if len(_t3_points) == 0:
        _t3_points.append(("info", "Distribusi kandidat mentor terlihat cukup merata pada segmen yang dipilih."))

    insight_card(_t3_points)

# ------------------------------------------
# TAB 4: ASPIRASI & SARAN MEMBER
# ------------------------------------------
with tab_voice:
    st.subheader("Ekspektasi Member & Kotak Saran Terbuka")

    exp_df = count_canonical_options(
        df[
            "What do you want from Google Developer Group on Campus Universitas Gunadarma?"
        ],
        EXPECTATION_OPTIONS,
    )
    fig_exp = px.bar(
        exp_df,
        x="Jumlah",
        y="Opsi",
        orientation="h",
        text=exp_df.apply(lambda r: f"{r['Jumlah']} ({r['Persentase']}%)", axis=1),
        title="Ekspektasi Utama Member Bergabung ke GDGoC Universitas Gunadarma",
        color_discrete_sequence=["#FBBC04"],
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown("#### 💡 Eksplorasi Saran & Masukan Member")
    search_kw = st.text_input(
        "Cari kata kunci dalam saran (misal: lomba, workshop, rekaman, jadwal, discord, solution challenge):",
        "",
    )

    sugg_col = "Any suggestions to make Google Developer Group on Campus Universitas Gunadarma better?"
    valid_sugg_df = df.dropna(subset=[sugg_col]).copy()
    valid_sugg_df = valid_sugg_df[
        valid_sugg_df[sugg_col].astype(str).str.strip().str.len() > 3
    ]
    valid_sugg_df = valid_sugg_df[
        ~valid_sugg_df[sugg_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "-",
                "none",
                "nothing",
                "no",
                "tidak ada",
                "belum ada",
                "nothing yet",
                "nope",
                "aman",
            ]
        )
    ]

    if search_kw:
        valid_sugg_df = valid_sugg_df[
            valid_sugg_df[sugg_col].str.contains(search_kw, case=False, na=False)
        ]

    st.caption(f"Menampilkan **{len(valid_sugg_df)}** saran konstruktif dari member:")
    st.dataframe(
        valid_sugg_df[
            [
                "Full name",
                "Tingkat",
                "Which one is your main path?",
                sugg_col,
            ]
        ].rename(
            columns={
                "Full name": "Nama",
                "Which one is your main path?": "Main Path",
                sugg_col: "Saran & Masukan",
            }
        ),
        use_container_width=True,
        height=400,
        hide_index=True,
    )

    # --- INSIGHT CARDS TAB 4 ---
    st.markdown("#### Catatan Aspirasi")
    _total_resp = len(df)
    _sugg_col_all = df[sugg_col].dropna()
    _valid_all = _sugg_col_all[
        (_sugg_col_all.astype(str).str.strip().str.len() > 3) &
        (~_sugg_col_all.astype(str).str.strip().str.lower().isin(
            ["-","none","nothing","no","tidak ada","belum ada","nothing yet","nope","aman"]
        ))
    ]
    _sugg_rate = round(len(_valid_all) / max(_total_resp, 1) * 100, 1)

    # Keyword detection untuk tema saran yang sering muncul
    _sugg_texts = " ".join(_valid_all.astype(str).str.lower().tolist())
    _theme_map = {
        "rekaman atau recording sesi": ["rekam", "record", "recording", "ulang"],
        "lomba atau kompetisi": ["lomba", "kompetisi", "competition", "contest", "hackathon"],
        "jadwal yang fleksibel": ["jadwal", "schedule", "waktu", "padat", "bentrok"],
        "Discord atau platform komunitas": ["discord", "grup", "komunitas", "platform"],
        "proyek nyata atau portofolio": ["proyek", "project", "portofolio", "portfolio", "real"],
        "informasi dan promosi lebih luas": ["informasi", "promosi", "publikasi", "sosmed", "instagram"],
    }

    _t4_points = []

    # Partisipasi saran
    if _sugg_rate >= 60:
        _t4_points.append(("info",
            f"Sebanyak {len(_valid_all)} member ({_sugg_rate}%) memberikan saran yang konstruktif. "
            "Tingkat partisipasi ini cukup tinggi, artinya member peduli dan punya ekspektasi yang perlu ditindaklanjuti."
        ))
    elif _sugg_rate < 40:
        _t4_points.append(("warn",
            f"Hanya {len(_valid_all)} member ({_sugg_rate}%) yang mengisi kolom saran. "
            "Ini bisa berarti member belum cukup tahu ekspektasi mereka, atau pertanyaan survei perlu diperjelas di periode berikutnya."
        ))

    # Tema yang muncul di saran
    for theme_label, keywords in _theme_map.items():
        count_kw = sum(_sugg_texts.count(kw) for kw in keywords)
        if count_kw >= 5:
            _t4_points.append(("warn",
                f"Tema '{theme_label}' muncul berulang kali di kolom saran member. "
                "Topik ini layak masuk agenda perencanaan program, bukan hanya dicatat."
            ))

    # Ekspektasi dominan
    _exp_col = "What do you want from Google Developer Group on Campus Universitas Gunadarma?"
    _exp_top = None
    _exp_top_n = 0
    for opt in EXPECTATION_OPTIONS:
        cnt = df[_exp_col].dropna().apply(lambda x: opt in str(x)).sum()
        if cnt > _exp_top_n:
            _exp_top_n = cnt
            _exp_top = opt
    if _exp_top:
        _exp_pct = round(_exp_top_n / max(_total_resp, 1) * 100, 1)
        _t4_points.append(("info",
            f"Ekspektasi terbanyak dari member adalah '{_exp_top}' ({_exp_top_n} orang, {_exp_pct}%). "
            "Pastikan program yang dirancang menjawab harapan ini secara langsung, bukan hanya sebagai pelengkap."
        ))

    if not _t4_points:
        _t4_points.append(("info", "Tidak ada pola dominan yang terdeteksi pada segmen yang dipilih."))

    insight_card(_t4_points)