#!/usr/bin/env python3
"""
Visualisation interactive des résultats d'optimisation K16 avec Streamlit.
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="K16 Optimization Results", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    """Charge les résultats d'optimisation."""
    try:
        with open('optimization_results.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error("Fichier optimization_results.json non trouvé!")
        return None

def prepare_dataframe(data):
    """Convertit les résultats en DataFrame pour faciliter l'analyse."""
    rows = []
    for result in data:
        row = {
            'search_type': result['search_type'],
            'max_depth': result['params']['max_depth'],
            'max_leaf_size': result['params']['max_leaf_size'],
            'max_data': result['params']['max_data'],
            'beam_width': result['params'].get('beam_width', None),
            'build_time': result['build_time'],
            'avg_tree_time_ms': result['results']['avg_tree_time'] * 1000,
            'avg_filter_time_ms': result['results']['avg_filter_time'] * 1000,
            'avg_total_time_ms': result['results']['avg_total_time'] * 1000,
            'avg_naive_time_ms': result['results']['avg_naive_time'] * 1000,
            'speedup': result['results']['speedup'],
            'recall': result['results']['avg_recall'],
            'avg_candidates': result['results']['avg_candidates'],
            'score': result['results']['avg_recall'] / result['results']['avg_total_time']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Créer une colonne pour l'identifiant unique de l'arbre
    df['tree_config'] = df.apply(lambda x: f"depth={x['max_depth']}_leaf={x['max_leaf_size']}_data={x['max_data']}", axis=1)
    
    return df

def main():
    st.title("🔬 K16 Optimization Results Analyzer")
    st.markdown("Analyse interactive des résultats d'optimisation K16")
    
    # Charger les données
    data = load_data()
    if data is None:
        return
        
    df = prepare_dataframe(data)
    
    # Sidebar pour les filtres
    st.sidebar.header("🎛️ Filtres")
    
    # Filtre par type de recherche
    search_types = st.sidebar.multiselect(
        "Type de recherche",
        options=['single', 'beam'],
        default=['single', 'beam']
    )
    
    # Filtre par max_data
    max_data_values = sorted(df['max_data'].unique())
    selected_max_data = st.sidebar.multiselect(
        "max_data",
        options=max_data_values,
        default=max_data_values
    )
    
    # Filtre par max_leaf_size
    max_leaf_sizes = sorted(df['max_leaf_size'].unique())
    selected_leaf_sizes = st.sidebar.multiselect(
        "max_leaf_size",
        options=max_leaf_sizes,
        default=max_leaf_sizes
    )
    
    # Filtre par beam_width
    beam_widths = sorted(df[df['beam_width'].notna()]['beam_width'].unique())
    selected_beam_widths = st.sidebar.multiselect(
        "beam_width (pour beam search)",
        options=beam_widths,
        default=beam_widths
    )
    
    # Appliquer les filtres
    filtered_df = df[
        (df['search_type'].isin(search_types)) &
        (df['max_data'].isin(selected_max_data)) &
        (df['max_leaf_size'].isin(selected_leaf_sizes))
    ]
    
    if 'beam' in search_types:
        beam_df = filtered_df[filtered_df['search_type'] == 'beam']
        beam_df = beam_df[beam_df['beam_width'].isin(selected_beam_widths)]
        single_df = filtered_df[filtered_df['search_type'] == 'single']
        filtered_df = pd.concat([single_df, beam_df])
    
    # Métriques principales
    st.header("📊 Métriques Principales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_speed = filtered_df.loc[filtered_df['avg_total_time_ms'].idxmin()]
        st.metric("⚡ Temps minimal", f"{best_speed['avg_total_time_ms']:.2f} ms")
        st.caption(f"Config: {best_speed['tree_config']}")
        st.caption(f"Type: {best_speed['search_type']}")
        if best_speed['beam_width']:
            st.caption(f"Beam width: {best_speed['beam_width']}")
    
    with col2:
        best_recall = filtered_df.loc[filtered_df['recall'].idxmax()]
        st.metric("🎯 Recall maximal", f"{best_recall['recall']:.3f}")
        st.caption(f"Config: {best_recall['tree_config']}")
        st.caption(f"Type: {best_recall['search_type']}")
        if best_recall['beam_width']:
            st.caption(f"Beam width: {best_recall['beam_width']}")
    
    with col3:
        best_speedup = filtered_df.loc[filtered_df['speedup'].idxmax()]
        st.metric("🚀 Speedup maximal", f"{best_speedup['speedup']:.1f}x")
        st.caption(f"Config: {best_speedup['tree_config']}")
        st.caption(f"Type: {best_speedup['search_type']}")
    
    with col4:
        best_score = filtered_df.loc[filtered_df['score'].idxmax()]
        st.metric("⭐ Meilleur compromis", f"Score: {best_score['score']:.1f}")
        st.caption(f"Config: {best_score['tree_config']}")
        st.caption(f"Type: {best_score['search_type']}")
        if best_score['beam_width']:
            st.caption(f"Beam width: {best_score['beam_width']}")
    
    # Graphiques
    st.header("📈 Visualisations")
    
    # Graphique principal : Temps vs Recall
    fig1 = px.scatter(
        filtered_df,
        x='avg_total_time_ms',
        y='recall',
        color='max_data',
        symbol='search_type',
        size='speedup',
        hover_data=['tree_config', 'beam_width', 'speedup'],
        title='Compromis Vitesse vs Recall',
        labels={
            'avg_total_time_ms': 'Temps moyen (ms)',
            'recall': 'Recall',
            'max_data': 'max_data'
        }
    )
    
    # Marquer les meilleurs points
    fig1.add_trace(go.Scatter(
        x=[best_speed['avg_total_time_ms']],
        y=[best_speed['recall']],
        mode='markers',
        marker=dict(size=20, symbol='star', color='red'),
        name='Meilleure vitesse',
        showlegend=True
    ))
    
    fig1.add_trace(go.Scatter(
        x=[best_recall['avg_total_time_ms']],
        y=[best_recall['recall']],
        mode='markers',
        marker=dict(size=20, symbol='star', color='green'),
        name='Meilleur recall',
        showlegend=True
    ))
    
    fig1.add_trace(go.Scatter(
        x=[best_score['avg_total_time_ms']],
        y=[best_score['recall']],
        mode='markers',
        marker=dict(size=20, symbol='star', color='gold'),
        name='Meilleur compromis',
        showlegend=True
    ))
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Impact du beam width
    beam_data = filtered_df[filtered_df['search_type'] == 'beam']
    if not beam_data.empty:
        fig2 = px.line(
            beam_data,
            x='beam_width',
            y='recall',
            color='tree_config',
            markers=True,
            title='Impact du Beam Width sur le Recall'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Impact du beam width sur le temps
        fig3 = px.line(
            beam_data,
            x='beam_width',
            y='avg_total_time_ms',
            color='tree_config',
            markers=True,
            title='Impact du Beam Width sur le Temps'
        )
        
        with col2:
            st.plotly_chart(fig3, use_container_width=True)
    
    # Heatmap des performances
    st.subheader("🔥 Heatmap des Performances")
    
    # Pour single
    single_data = filtered_df[filtered_df['search_type'] == 'single'].pivot_table(
        values='recall',
        index='max_leaf_size',
        columns='max_data'
    )
    
    fig4 = go.Figure(data=go.Heatmap(
        z=single_data.values,
        x=single_data.columns,
        y=single_data.index,
        colorscale='Viridis',
        text=single_data.values.round(3),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig4.update_layout(
        title='Recall par Configuration (Single Search)',
        xaxis_title='max_data',
        yaxis_title='max_leaf_size'
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Distribution des temps
    st.subheader("📊 Distribution des Temps")
    
    fig5 = px.box(
        filtered_df,
        x='max_data',
        y='avg_total_time_ms',
        color='search_type',
        title='Distribution des Temps par max_data',
        labels={'avg_total_time_ms': 'Temps moyen (ms)'}
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    # Tableau détaillé
    st.header("📋 Tableau Détaillé")
    
    # Top 10 configurations
    top_configs = filtered_df.nlargest(10, 'score')[
        ['tree_config', 'search_type', 'beam_width', 'avg_total_time_ms', 
         'recall', 'speedup', 'score']
    ].round(3)
    
    st.dataframe(top_configs, use_container_width=True)
    
    # Analyse par configuration d'arbre
    st.header("🌳 Analyse par Configuration d'Arbre")
    
    tree_analysis = filtered_df.groupby('tree_config').agg({
        'recall': ['min', 'max', 'mean'],
        'avg_total_time_ms': ['min', 'max', 'mean'],
        'speedup': ['min', 'max', 'mean'],
        'build_time': 'first'
    }).round(3)
    
    st.dataframe(tree_analysis, use_container_width=True)
    
    # Export des données filtrées
    st.header("💾 Export des Données")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Télécharger les données filtrées (CSV)",
        data=csv,
        file_name="k16_optimization_filtered.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()