import io
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "Monitoraggio operativo logistica CAAT - Dashboard flussi in ingresso"
USEFUL_COLUMNS = [
    "data/ora",
    "dispositivo",
    "id",
    "profilo prodotto",
    "categoria utente",
    "evento",
    "esito validazione",
    "targa",
]
BIN_MINUTES = 15
DIRECTION_OPTIONS = ["Ingresso", "Uscita", "Altro"]
TIPO_OPTIONS = ["Occasionale", "Abbonato / non occasionale"]
PLATE_STATUS_OPTIONS = ["Letta", "NOT READ", "Mancante"]


# =========================================================
# CONFIG UI
# =========================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":material/bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <div style="
        padding: 1.1rem 1.4rem;
        border: 1px solid #d9dee7;
        border-radius: 16px;
        background: linear-gradient(90deg, #fafbfc 0%, #f4f6f9 100%);
        margin-bottom: 1rem;">
        <div style="
            font-size: 1.8rem;
            font-weight: 700;
            color: #1f2937;
            line-height: 1.2;">
            {APP_TITLE}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# NORMALIZZAZIONE E PARSING
# =========================================================
def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_text(col) for col in out.columns]
    valid_cols = [col for col in out.columns if col and not col.startswith("unnamed")]
    return out.loc[:, valid_cols]


def minutes_to_label(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def time_options(step: int = BIN_MINUTES) -> List[int]:
    return list(range(0, 24 * 60, step))


def normalize_plate(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return None if text in {"", "NAN"} else text


def plate_status(value) -> str:
    plate = normalize_plate(value)
    if plate is None:
        return "Mancante"
    if plate == "NOT READ":
        return "NOT READ"
    return "Letta"


def direction_from_device(device) -> str:
    text = str(device).strip().upper()
    if text.startswith("LE"):
        return "Ingresso"
    if text.startswith("LX"):
        return "Uscita"
    return "Altro"


def profile_code(profile) -> str:
    text = normalize_text(profile)

    for prefix in ["ct", "lpmu", "a"]:
        match = re.match(rf"^({prefix}\d*)", text)
        if match:
            return match.group(1)

    if text.startswith("m2"):
        if "ct" in text:
            return "m2 ct"
        if "lpmu" in text:
            return "m2 lpmu"
        if "a" in text:
            return "m2 a"
        return "m2"

    special_tokens = {
        "passpartout": "passpartout",
        "privati consumatori": "privati consumatori",
        "tessera anonima": "tessera anonima",
        "biglietto": "biglietto",
    }
    for token, label in special_tokens.items():
        if token in text:
            return label

    head = text.split(" - ")[0].strip()
    return head if head else "altro"


def profile_family(code: str) -> str:
    code = normalize_text(code)
    if code.startswith("ct") or code == "m2 ct":
        return "ct"
    if code.startswith("lpmu") or code == "m2 lpmu":
        return "lpmu"
    if code.startswith("a") or code == "m2 a":
        return "a"
    if code.startswith("m2"):
        return "m2"
    if code in {"passpartout", "privati consumatori", "tessera anonima", "biglietto"}:
        return code
    return "altro"


def tipo_cliente(categoria_utente) -> str:
    return "Occasionale" if "utenti occasionali" in normalize_text(categoria_utente) else "Abbonato / non occasionale"


def event_status(evento, esito) -> str:
    evento_text = normalize_text(evento)
    esito_text = normalize_text(esito)

    if evento_text in {"ingresso riuscito", "uscita effettuata", "uscita riuscita", "ingresso pedone riuscito"}:
        return "Riuscito"
    if evento_text == "validazione titolo" and esito_text == "valido":
        return "Validazione valida"
    if "fallit" in evento_text:
        return "Fallito"
    if "negat" in esito_text or "accesso negato" in esito_text:
        return "Negato"
    if any(token in evento_text for token in ["violazione", "forzat", "barriera"]):
        return "Anomalia/forzato"
    return "Altro"


# =========================================================
# CARICAMENTO E PREPARAZIONE DATI
# =========================================================
@st.cache_data

def load_excel_data(file_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    excel = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = {normalize_text(name): name for name in excel.sheet_names}

    sheet_key = next((key for key in ["dati ripuliti", "raw_data", "raw data"] if key in sheets), None)
    if sheet_key is None:
        raise ValueError("Il file deve contenere il foglio 'dati ripuliti' oppure 'raw_data'.")

    sheet_name = sheets[sheet_key]
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    df = clean_columns(df)

    available_cols = [col for col in USEFUL_COLUMNS if col in df.columns]
    if not available_cols:
        raise ValueError("Nel file non sono presenti le colonne utili attese.")
    if "data/ora" not in df.columns:
        raise ValueError("Manca la colonna 'Data/Ora'.")

    return df[available_cols].copy(), sheet_name


@st.cache_data

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["data/ora"] = pd.to_datetime(out["data/ora"], errors="coerce")
    out = out.dropna(subset=["data/ora"]).copy()

    out["data"] = out["data/ora"].dt.date
    out["ora"] = out["data/ora"].dt.hour
    out["minuto"] = out["data/ora"].dt.minute
    out["minuti_giorno"] = out["ora"] * 60 + out["minuto"]

    out["direzione"] = out["dispositivo"].apply(direction_from_device)
    out["codice_profilo"] = out["profilo prodotto"].apply(profile_code)
    out["famiglia_profilo"] = out["codice_profilo"].apply(profile_family)
    out["tipo_cliente"] = out["categoria utente"].apply(tipo_cliente)
    out["stato_evento"] = out.apply(lambda row: event_status(row.get("evento"), row.get("esito validazione")), axis=1)
    out["success_candidate"] = out["stato_evento"].isin(["Riuscito", "Validazione valida"])
    out["targa_normalizzata"] = out["targa"].apply(normalize_plate)
    out["stato_targa"] = out["targa"].apply(plate_status)
    return out


@st.cache_data

def deduplicate_movements(df: pd.DataFrame, dedupe_seconds: int = 20) -> pd.DataFrame:
    out = df[df["success_candidate"]].copy()
    out = out[out["direzione"].isin(["Ingresso", "Uscita"])].copy()

    if out.empty:
        return out

    out["priority_status"] = np.where(out["stato_evento"] == "Riuscito", 2, 1)
    out["priority_plate"] = np.select(
        [
            out["stato_targa"] == "Letta",
            out["stato_targa"] == "NOT READ",
            out["stato_targa"] == "Mancante",
        ],
        [2, 1, 0],
        default=0,
    )

    group_cols = ["id", "direzione", "data", "codice_profilo"]
    out = out.sort_values(group_cols + ["data/ora"]).copy()
    out["diff_sec"] = out.groupby(group_cols)["data/ora"].diff().dt.total_seconds()
    out["cluster"] = out.groupby(group_cols)["diff_sec"].transform(lambda s: (s.isna() | (s > dedupe_seconds)).cumsum())

    sort_cols = group_cols + ["cluster", "priority_status", "priority_plate", "data/ora"]
    sort_ascending = [True, True, True, True, True, False, False, True]
    out = out.sort_values(by=sort_cols, ascending=sort_ascending)

    return (
        out.groupby(group_cols + ["cluster"], as_index=False)
        .head(1)
        .sort_values("data/ora")
        .reset_index(drop=True)
    )


# =========================================================
# FILTRI E CONTESTO
# =========================================================
def add_operational_bands(df: pd.DataFrame, end_1: int, end_2: int) -> pd.DataFrame:
    out = df.copy()
    labels = [
        f"00:00 - {minutes_to_label(end_1)}",
        f"{minutes_to_label(end_1)} - {minutes_to_label(end_2)}",
        f"{minutes_to_label(end_2)} - 24:00",
    ]
    out["fascia_operativa"] = pd.Categorical(
        np.select(
            [
                (out["minuti_giorno"] >= 0) & (out["minuti_giorno"] < end_1),
                (out["minuti_giorno"] >= end_1) & (out["minuti_giorno"] < end_2),
                (out["minuti_giorno"] >= end_2) & (out["minuti_giorno"] < 24 * 60),
            ],
            labels,
            default="Non classificato",
        ),
        categories=labels + ["Non classificato"],
        ordered=True,
    )
    return out


def base_scope(df: pd.DataFrame, start_date, end_date, start_min: int, end_min: int, directions: Sequence[str], tipi: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out = out[(out["data"] >= start_date) & (out["data"] <= end_date)]
    out = out[(out["minuti_giorno"] >= start_min) & (out["minuti_giorno"] <= end_min)]

    if not directions or not tipi:
        return out.iloc[0:0]

    out = out[out["direzione"].isin(directions)]
    out = out[out["tipo_cliente"].isin(tipi)]
    return out


def profile_scope(df: pd.DataFrame, families: Sequence[str], codes: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if not families or not codes:
        return out.iloc[0:0]
    out = out[out["famiglia_profilo"].isin(families)]
    out = out[out["codice_profilo"].isin(codes)]
    return out


def direction_context(selected_direction: Sequence[str], ct: bool = False) -> Dict[str, str]:
    chosen = set(selected_direction)
    suffix = " CT" if ct else ""
    entity_suffix = " ct" if ct else ""

    if chosen == {"Ingresso"}:
        return {
            "entity": f"ingressi{entity_suffix}",
            "entity_cap": f"Ingressi{suffix}",
            "device_title": f"Riepilogo ingressi{suffix} per codice LE",
            "curve_title": f"Curva degli ingressi{suffix}",
            "daily_title": f"Andamento giornaliero degli ingressi{suffix}",
            "heat_title": f"Mappa di concentrazione degli ingressi{suffix}",
        }
    if chosen == {"Uscita"}:
        return {
            "entity": f"uscite{entity_suffix}",
            "entity_cap": f"Uscite{suffix}",
            "device_title": f"Riepilogo uscite{suffix} per codice LX",
            "curve_title": f"Curva delle uscite{suffix}",
            "daily_title": f"Andamento giornaliero delle uscite{suffix}",
            "heat_title": f"Mappa di concentrazione delle uscite{suffix}",
        }
    if chosen == {"Altro"}:
        return {
            "entity": f"transiti altri{entity_suffix}",
            "entity_cap": f"Transiti altri{suffix}",
            "device_title": f"Riepilogo transiti{suffix} per dispositivo",
            "curve_title": f"Curva dei transiti altri{suffix}",
            "daily_title": f"Andamento giornaliero dei transiti altri{suffix}",
            "heat_title": f"Mappa di concentrazione dei transiti altri{suffix}",
        }
    return {
        "entity": f"movimenti{entity_suffix}",
        "entity_cap": f"Movimenti{suffix}",
        "device_title": f"Riepilogo{suffix} per codice dispositivo",
        "curve_title": f"Curva dei flussi{suffix}",
        "daily_title": f"Andamento giornaliero dei movimenti{suffix}",
        "heat_title": f"Mappa di concentrazione dei movimenti{suffix}",
    }


# =========================================================
# AGGREGAZIONI
# =========================================================
def build_curve(df: pd.DataFrame, metric_mode: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bin_minute", "valore"])

    out = df.copy()
    out["bin_minute"] = (out["minuti_giorno"] // BIN_MINUTES) * BIN_MINUTES

    if metric_mode == "Media per giorno":
        daily = out.groupby(["data", "bin_minute"]).size().reset_index(name="n")
        out = daily.groupby("bin_minute")["n"].mean().reset_index(name="valore")
    else:
        out = out.groupby("bin_minute").size().reset_index(name="valore")

    return out.sort_values("bin_minute")


def build_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["bin_minute"] = (out["minuti_giorno"] // BIN_MINUTES) * BIN_MINUTES
    return (
        out.groupby(["data", "bin_minute"])
        .size()
        .reset_index(name="movimenti")
        .pivot(index="data", columns="bin_minute", values="movimenti")
        .fillna(0)
        .sort_index()
    )


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["data", "movimenti"])
    return df.groupby("data").size().reset_index(name="movimenti").sort_values("data")


def build_profile_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["famiglia_profilo", "codice_profilo", "Movimenti", "% sul totale"])

    out = (
        df.groupby(["famiglia_profilo", "codice_profilo"])
        .size()
        .reset_index(name="Movimenti")
        .sort_values(["Movimenti", "codice_profilo"], ascending=[False, True])
    )
    total = out["Movimenti"].sum()
    out["% sul totale"] = np.where(total > 0, (out["Movimenti"] / total * 100).round(1), 0)
    return out


def build_zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["fascia_operativa", "Movimenti", "% sul totale"])

    out = df.groupby("fascia_operativa", observed=False).size().reset_index(name="Movimenti")
    total = out["Movimenti"].sum()
    out["% sul totale"] = np.where(total > 0, (out["Movimenti"] / total * 100).round(1), 0)
    return out


def build_device_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["direzione", "dispositivo", "Movimenti"])
    return (
        df.groupby(["direzione", "dispositivo"])
        .size()
        .reset_index(name="Movimenti")
        .sort_values(["direzione", "Movimenti", "dispositivo"], ascending=[True, False, True])
    )


def build_plate_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["targa_visualizzata", "Occorrenze", "Primo accesso", "Ultimo accesso", "Codici profilo"])

    out = df.copy()
    out["targa_visualizzata"] = np.where(
        out["stato_targa"] == "Letta",
        out["targa_normalizzata"],
        "[" + out["stato_targa"].astype(str) + "]",
    )

    out = (
        out.groupby("targa_visualizzata")
        .agg(
            Occorrenze=("targa_visualizzata", "size"),
            Primo_accesso=("data/ora", "min"),
            Ultimo_accesso=("data/ora", "max"),
            Codici_profilo=("codice_profilo", lambda x: ", ".join(sorted(set(map(str, x.dropna()))))[:120]),
        )
        .reset_index()
        .sort_values("Occorrenze", ascending=False)
    )

    return out.rename(
        columns={
            "Primo_accesso": "Primo accesso",
            "Ultimo_accesso": "Ultimo accesso",
            "Codici_profilo": "Codici profilo",
        }
    )


def build_plate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["data/ora", "stato_targa", "id", "codice_profilo", "dispositivo", "evento", "esito validazione"])

    cols = ["data/ora", "stato_targa", "id", "codice_profilo", "dispositivo", "evento", "esito validazione"]
    return df[df["stato_targa"].isin(["NOT READ", "Mancante"])][cols].sort_values("data/ora", ascending=False)


# =========================================================
# FIGURE
# =========================================================
def create_curve_figure(curve_df: pd.DataFrame, start_min: int, end_min: int, end_1: int, end_2: int, metric_mode: str, context: Dict[str, str]):
    fig = go.Figure()

    if not curve_df.empty:
        fig.add_trace(
            go.Bar(
                x=curve_df["bin_minute"],
                y=curve_df["valore"],
                marker=dict(color=curve_df["valore"], colorscale="YlOrRd"),
                opacity=0.28,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=curve_df["bin_minute"],
                y=curve_df["valore"],
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.10)",
                line=dict(width=4, shape="spline", smoothing=0.7, color="#2563eb"),
                marker=dict(
                    size=8,
                    color=curve_df["valore"],
                    colorscale="Turbo",
                    line=dict(width=1, color="white"),
                    showscale=False,
                ),
                name=context["entity_cap"],
                hovertemplate="Ora: %{x}<br>Valore: %{y:.1f}<extra></extra>",
            )
        )

    zones = [
        (0, end_1, "rgba(59,130,246,0.08)", "Fascia 1"),
        (end_1, end_2, "rgba(245,158,11,0.08)", "Fascia 2"),
        (end_2, 24 * 60, "rgba(16,185,129,0.08)", "Fascia 3"),
    ]
    for x0, x1, color, label in zones:
        a = max(x0, start_min)
        b = min(x1, end_min)
        if a < b:
            fig.add_vrect(x0=a, x1=b, fillcolor=color, line_width=0, annotation_text=label, annotation_position="top left")

    ticks = list(range(start_min - (start_min % 60), end_min + 1, 60))
    fig.update_layout(
        title=f"{context['curve_title']} - {metric_mode.lower()}",
        xaxis_title="Ora operativa",
        yaxis_title=context["entity_cap"],
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(
            tickmode="array",
            tickvals=ticks,
            ticktext=[minutes_to_label(x) for x in ticks],
            range=[start_min, end_min],
        ),
        bargap=0,
    )
    return fig


def create_heatmap_figure(heatmap_df: pd.DataFrame, context: Dict[str, str]):
    fig = go.Figure()
    if not heatmap_df.empty:
        x_vals = list(heatmap_df.columns)
        fig.add_trace(
            go.Heatmap(
                z=heatmap_df.values,
                x=[minutes_to_label(int(v)) for v in x_vals],
                y=[str(v) for v in heatmap_df.index],
                colorscale="YlOrRd",
                colorbar_title=context["entity_cap"],
                hovertemplate="Data: %{y}<br>Ora: %{x}<br>Movimenti: %{z}<extra></extra>",
            )
        )
    fig.update_layout(title=context["heat_title"], height=460, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def create_daily_figure(daily_df: pd.DataFrame, context: Dict[str, str]):
    fig = go.Figure()
    if not daily_df.empty:
        fig.add_trace(
            go.Scatter(
                x=daily_df["data"],
                y=daily_df["movimenti"],
                mode="lines+markers",
                line=dict(width=3, color="#2563eb"),
                marker=dict(size=7),
                name=context["entity_cap"],
                hovertemplate="Data: %{x}<br>Movimenti: %{y}<extra></extra>",
            )
        )
    fig.update_layout(title=context["daily_title"], height=340, margin=dict(l=20, r=20, t=55, b=20))
    return fig


# =========================================================
# EXPORT
# =========================================================
def build_excel_bytes(df_operativo, zone_summary_df, profile_summary_df, plate_summary_df, plate_anomalies_df, device_summary_df) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_operativo.to_excel(writer, sheet_name="movimenti_operativi", index=False)
        zone_summary_df.to_excel(writer, sheet_name="riepilogo_fasce", index=False)
        profile_summary_df.to_excel(writer, sheet_name="riepilogo_profili", index=False)
        device_summary_df.to_excel(writer, sheet_name="riepilogo_dispositivi", index=False)
        plate_summary_df.to_excel(writer, sheet_name="targhe_ingresso", index=False)
        plate_anomalies_df.to_excel(writer, sheet_name="targhe_problematiche", index=False)
    output.seek(0)
    return output.getvalue()


# =========================================================
# RENDER VISTA
# =========================================================
def render_kpis(oper_view: pd.DataFrame, neg_view: pd.DataFrame, device_summary_df: pd.DataFrame, selected_direction: Sequence[str], extra_metrics=None):
    chosen = set(selected_direction)
    total_mov = int(len(oper_view))
    total_in = int((oper_view["direzione"] == "Ingresso").sum()) if not oper_view.empty else 0
    total_out = int((oper_view["direzione"] == "Uscita").sum()) if not oper_view.empty else 0
    total_denied = int(len(neg_view))

    if chosen == {"Ingresso"}:
        cols = st.columns(4 if extra_metrics else 3)
        metrics = [
            ("Ingressi operativi", total_in),
            ("Negati/Falliti", total_denied),
            ("Codici LE presenti", int(device_summary_df[device_summary_df["direzione"] == "Ingresso"]["dispositivo"].nunique())),
        ]
    elif chosen == {"Uscita"}:
        cols = st.columns(4 if extra_metrics else 3)
        metrics = [
            ("Uscite operative", total_out),
            ("Negati/Falliti", total_denied),
            ("Codici LX presenti", int(device_summary_df[device_summary_df["direzione"] == "Uscita"]["dispositivo"].nunique())),
        ]
    else:
        extra_count = len(extra_metrics) if extra_metrics else 0
        cols = st.columns(4 + extra_count)
        metrics = [
            ("Movimenti operativi", total_mov),
            ("Ingressi", total_in),
            ("Uscite", total_out),
            ("Negati/Falliti", total_denied),
        ]

    if extra_metrics:
        metrics.extend(extra_metrics)

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)


def render_device_summary(device_summary_df: pd.DataFrame, selected_direction: Sequence[str], title: str):
    chosen = set(selected_direction)
    st.markdown(f"**{title}**")

    def show_direction(direction, heading):
        filtered = device_summary_df[device_summary_df["direzione"] == direction]
        st.markdown(f"**{heading}**")
        st.dataframe(filtered[["dispositivo", "Movimenti"]], use_container_width=True, hide_index=True)

    if chosen == {"Ingresso"}:
        st.dataframe(
            device_summary_df[device_summary_df["direzione"] == "Ingresso"][["dispositivo", "Movimenti"]],
            use_container_width=True,
            hide_index=True,
        )
    elif chosen == {"Uscita"}:
        st.dataframe(
            device_summary_df[device_summary_df["direzione"] == "Uscita"][["dispositivo", "Movimenti"]],
            use_container_width=True,
            hide_index=True,
        )
    elif chosen == {"Altro"}:
        st.dataframe(
            device_summary_df[device_summary_df["direzione"] == "Altro"][["dispositivo", "Movimenti"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            show_direction("Ingresso", "Codici LE")
        with col2:
            show_direction("Uscita", "Codici LX")


def render_plate_section(oper_view: pd.DataFrame, selected_plate_status: Sequence[str]):
    incoming = oper_view[oper_view["direzione"] == "Ingresso"].copy()
    incoming = incoming[incoming["stato_targa"].isin(selected_plate_status)]

    plate_summary_df = build_plate_summary(incoming)
    plate_anomalies_df = build_plate_anomalies(incoming)

    st.subheader("Monitoraggio targhe in ingresso")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.metric("Ingressi con targa letta", int((incoming["stato_targa"] == "Letta").sum()))
    with t2:
        st.metric("Ingressi NOT READ", int((incoming["stato_targa"] == "NOT READ").sum()))
    with t3:
        st.metric("Ingressi senza targa", int((incoming["stato_targa"] == "Mancante").sum()))
    with t4:
        lette = incoming[incoming["stato_targa"] == "Letta"]
        st.metric("Targhe uniche lette", int(lette["targa_normalizzata"].nunique()))

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Targhe in ingresso - riepilogo**")
        st.dataframe(plate_summary_df, use_container_width=True, hide_index=True)
    with p2:
        st.markdown("**Targhe problematiche**")
        st.dataframe(plate_anomalies_df, use_container_width=True, hide_index=True)

    return plate_summary_df, plate_anomalies_df


def render_dashboard(base_view: pd.DataFrame, oper_view: pd.DataFrame, selected_direction: Sequence[str], selected_plate_status: Sequence[str], context: Dict[str, str], key_prefix: str, view_title: str, extra_metrics=None):
    neg_view = base_view[base_view["stato_evento"].isin(["Negato", "Fallito"])].copy()
    device_summary_df = build_device_summary(oper_view)
    zone_summary_df = build_zone_summary(oper_view)
    profile_summary_df = build_profile_summary(oper_view)

    st.subheader(view_title)
    render_kpis(oper_view, neg_view, device_summary_df, selected_direction, extra_metrics=extra_metrics)
    render_device_summary(device_summary_df, selected_direction, context["device_title"])

    st.subheader(context["curve_title"])
    ctrl1, ctrl2 = st.columns([1.2, 4])
    with ctrl1:
        metric_mode = st.radio("Valore curva", ["Totale periodo", "Media per giorno"], index=0, key=f"{key_prefix}_curve_mode")
    with ctrl2:
        st.caption("Curva e heatmap calcolate su slot fissi da 15 minuti. Le zone colorate rappresentano le fasce operative impostate nella sidebar.")

    curve_df = build_curve(oper_view, metric_mode)
    heatmap_df = build_heatmap(oper_view)
    daily_df = build_daily(oper_view)

    st.plotly_chart(
        create_curve_figure(curve_df, window_start, window_end, zone_1_end, zone_2_end, metric_mode, context),
        use_container_width=True,
        key=f"{key_prefix}_curve",
    )

    g1, g2 = st.columns([1.4, 1])
    with g1:
        st.plotly_chart(create_heatmap_figure(heatmap_df, context), use_container_width=True, key=f"{key_prefix}_heat")
    with g2:
        st.plotly_chart(create_daily_figure(daily_df, context), use_container_width=True, key=f"{key_prefix}_daily")

    st.subheader("Riepiloghi operativi")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Riepilogo per fascia operativa**")
        st.dataframe(zone_summary_df, use_container_width=True, hide_index=True)
    with r2:
        st.markdown("**Riepilogo per codice profilo**")
        st.dataframe(profile_summary_df, use_container_width=True, hide_index=True)

    plate_summary_df, plate_anomalies_df = render_plate_section(oper_view, selected_plate_status)

    with st.expander("Dettaglio movimenti operativi", expanded=False):
        cols = [
            "data/ora",
            "direzione",
            "dispositivo",
            "id",
            "tipo_cliente",
            "famiglia_profilo",
            "codice_profilo",
            "profilo prodotto",
            "evento",
            "esito validazione",
            "targa_normalizzata",
            "stato_targa",
            "fascia_operativa",
        ]
        st.dataframe(oper_view[cols], use_container_width=True)

    st.subheader("Download")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Scarica movimenti operativi CSV",
            data=oper_view.to_csv(index=False).encode("utf-8"),
            file_name=f"{key_prefix}_movimenti_operativi.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )
    with d2:
        st.download_button(
            "Scarica report Excel",
            data=build_excel_bytes(oper_view, zone_summary_df, profile_summary_df, plate_summary_df, plate_anomalies_df, device_summary_df),
            file_name=f"{key_prefix}_report_flussi_caat.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_xlsx",
        )


# =========================================================
# APP
# =========================================================
st.sidebar.header("Importa file Excel")
uploaded_excel = st.sidebar.file_uploader("Carica file Excel", type=["xlsx", "xls"])

if uploaded_excel is None:
    st.info("Carica un file Excel con la struttura dei transiti totali per iniziare.")
    st.stop()

try:
    raw_df, source_sheet = load_excel_data(uploaded_excel.getvalue())
except Exception as error:
    st.error(f"Errore nella lettura del file Excel: {error}")
    st.stop()

base_df = prepare_data(raw_df)
oper_df = deduplicate_movements(base_df, dedupe_seconds=20)

st.caption(
    f"Origine dati utilizzata: foglio **{source_sheet}**. "
    f"Il filtro direzione resta su Ingresso / Uscita / Altro. "
    f"La colonna Categoria utente viene corretta in 'Occasionale' oppure 'Abbonato / non occasionale'."
)

st.sidebar.header("Impostazioni orarie")
all_time_options = time_options()
window_start, window_end = st.sidebar.select_slider(
    "Intervallo orario da analizzare",
    options=all_time_options,
    value=(0, 12 * 60),
    format_func=minutes_to_label,
)
zone_1_end = st.sidebar.select_slider(
    "Fine fascia operativa 1",
    options=all_time_options[1:-2],
    value=210,
    format_func=minutes_to_label,
)
zone_2_candidates = [value for value in all_time_options[2:-1] if value > zone_1_end]
zone_2_end = st.sidebar.select_slider(
    "Fine fascia operativa 2",
    options=zone_2_candidates,
    value=360 if 360 in zone_2_candidates else zone_2_candidates[0],
    format_func=minutes_to_label,
)

base_df = add_operational_bands(base_df, zone_1_end, zone_2_end)
oper_df = add_operational_bands(oper_df, zone_1_end, zone_2_end)

st.sidebar.header("Filtri")
date_min = oper_df["data"].min()
date_max = oper_df["data"].max()
date_range = st.sidebar.date_input(
    "Intervallo date",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)
start_date, end_date = date_range if isinstance(date_range, (list, tuple)) and len(date_range) == 2 else (date_min, date_max)

selected_direction = st.sidebar.multiselect("Direzione", DIRECTION_OPTIONS, default=["Ingresso"])
selected_tipo = st.sidebar.multiselect("Tipo cliente", TIPO_OPTIONS, default=TIPO_OPTIONS)

family_options = sorted(oper_df["famiglia_profilo"].dropna().astype(str).unique().tolist())
default_family = ["ct"] if "ct" in family_options else family_options
selected_family = st.sidebar.multiselect("Famiglia profilo", family_options, default=default_family)

code_source = oper_df[oper_df["famiglia_profilo"].isin(selected_family)] if selected_family else oper_df.iloc[0:0]
code_options = sorted(code_source["codice_profilo"].dropna().astype(str).unique().tolist())
selected_code = st.sidebar.multiselect("Codice profilo", code_options, default=code_options)

selected_plate_status = st.sidebar.multiselect("Stato targa (report targhe)", PLATE_STATUS_OPTIONS, default=PLATE_STATUS_OPTIONS)

base_scoped = base_scope(base_df, start_date, end_date, window_start, window_end, selected_direction, selected_tipo)
oper_scoped = base_scope(oper_df, start_date, end_date, window_start, window_end, selected_direction, selected_tipo)

general_base_view = profile_scope(base_scoped, selected_family, selected_code)
general_oper_view = profile_scope(oper_scoped, selected_family, selected_code)

ct_code_options = sorted(
    oper_scoped[oper_scoped["famiglia_profilo"] == "ct"]["codice_profilo"].dropna().astype(str).unique().tolist()
)


# =========================================================
# TABS
# =========================================================
general_tab, ct_tab = st.tabs(["Vista generale", "Vista CT"])

with general_tab:
    render_dashboard(
        base_view=general_base_view,
        oper_view=general_oper_view,
        selected_direction=selected_direction,
        selected_plate_status=selected_plate_status,
        context=direction_context(selected_direction, ct=False),
        key_prefix="generale",
        view_title="Vista generale",
    )

with ct_tab:
    st.subheader("Impostazioni vista CT")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        selected_ct_codes = st.multiselect("Codici CT", ct_code_options, default=ct_code_options, key="ct_codes")
    with c2:
        ct_share = round((len(oper_scoped[oper_scoped["famiglia_profilo"] == "ct"]) / len(oper_scoped) * 100), 1) if len(oper_scoped) > 0 else 0
        st.metric("Quota CT sul totale vista", f"{ct_share}%")

    ct_base_view = profile_scope(base_scoped[base_scoped["famiglia_profilo"] == "ct"], ["ct"], selected_ct_codes)
    ct_oper_view = profile_scope(oper_scoped[oper_scoped["famiglia_profilo"] == "ct"], ["ct"], selected_ct_codes)

    unique_ct_plates = int(
        ct_oper_view.loc[
            (ct_oper_view["direzione"] == "Ingresso") & (ct_oper_view["stato_targa"] == "Letta"),
            "targa_normalizzata",
        ].nunique()
    ) if not ct_oper_view.empty else 0

    render_dashboard(
        base_view=ct_base_view,
        oper_view=ct_oper_view,
        selected_direction=selected_direction,
        selected_plate_status=selected_plate_status,
        context=direction_context(selected_direction, ct=True),
        key_prefix="ct",
        view_title="Vista CT",
        extra_metrics=[("Targhe CT uniche", unique_ct_plates)],
    )
