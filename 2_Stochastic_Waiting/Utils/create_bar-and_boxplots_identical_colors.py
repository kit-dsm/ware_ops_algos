import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from brokenaxes import brokenaxes
import numpy as np
from collections import defaultdict
from matplotlib.lines import Line2D


def cm_to_inches(cm):
    """Konvertiert Zentimeter zu Zoll für matplotlib figsize"""
    return cm / 2.54


def get_figure_size(width_cm, height_cm):
    """Gibt figsize in Zoll basierend auf cm-Eingabe zurück"""
    return (cm_to_inches(width_cm), cm_to_inches(height_cm))


def get_legend_position(position_str):
    """
    Konvertiert String-Position in bbox_to_anchor und loc Parameter
    """
    positions = {
        'auto': 'auto',
        'upper_right': ((0.98, 0.98), 'upper right'),
        'upper_left': ((0.02, 0.98), 'upper left'),
        'lower_right': ((0.98, 0.02), 'lower right'),
        'lower_left': ((0.02, 0.02), 'lower left'),
        'upper_center': ((0.5, 0.98), 'upper center'),
        'outside': ((1.02, 1), 'upper left')  # Außerhalb rechts
    }
    return positions.get(position_str, 'auto')


def find_best_legend_position(ax, df, kpi):
    """
    Findet die beste Position für die Legende basierend auf der Datenverteilung
    Analysiert die Datendichte in verschiedenen Bereichen des Plots
    """
    # Mögliche Positionen (bbox_to_anchor, loc, score_weight)
    positions = [
        ((0.98, 0.98), 'upper right', 'upper_right'),  # Oben rechts
        ((0.02, 0.98), 'upper left', 'upper_left'),  # Oben links
        ((0.98, 0.02), 'lower right', 'lower_right'),  # Unten rechts
        ((0.02, 0.02), 'lower left', 'lower_left'),  # Unten links
        ((0.5, 0.98), 'upper center', 'upper_center'),  # Oben mittig
    ]

    # Datenwerte analysieren
    data_values = df[kpi].dropna()
    if len(data_values) == 0:
        return positions[0][:2]  # Fallback

    y_min, y_max = data_values.min(), data_values.max()
    y_range = y_max - y_min

    # Wenn sehr kleine Range, verwende einfach obere rechte Position
    if y_range == 0:
        return positions[0][:2]

    # Berechne Scores für jede Position basierend auf Datendichte
    position_scores = {}

    # Definiere Bereiche für die Analyse (in Prozent der Datenrange)
    regions = {
        'upper_right': {'y_min': 0.7, 'y_max': 1.0, 'x_side': 'right'},
        'upper_left': {'y_min': 0.7, 'y_max': 1.0, 'x_side': 'left'},
        'lower_right': {'y_min': 0.0, 'y_max': 0.3, 'x_side': 'right'},
        'lower_left': {'y_min': 0.0, 'y_max': 0.3, 'x_side': 'left'},
        'upper_center': {'y_min': 0.8, 'y_max': 1.0, 'x_side': 'center'}
    }

    # Für Boxplots: Analysiere Datendichte in verschiedenen Kategorien
    waiting_strategies = sorted(df['waiting_strategy'].unique())
    n_categories = len(waiting_strategies)

    for pos_name, region in regions.items():
        score = 0

        # Y-Bereich für diese Region
        region_y_min = y_min + region['y_min'] * y_range
        region_y_max = y_min + region['y_max'] * y_range

        # Bestimme welche x-Kategorien zu prüfen sind
        if region['x_side'] == 'right':
            x_categories = waiting_strategies[-2:] if n_categories >= 2 else waiting_strategies
        elif region['x_side'] == 'left':
            x_categories = waiting_strategies[:2]
        else:  # center
            mid = n_categories // 2
            start_idx = max(0, mid - 1)
            end_idx = min(n_categories, mid + 2)
            x_categories = waiting_strategies[start_idx:end_idx]

        # Zähle Datenpunkte in diesem Bereich
        data_in_region = 0
        total_points = 0

        for cat in x_categories:
            if cat in waiting_strategies:
                cat_data = df[df['waiting_strategy'] == cat][kpi].dropna()
                total_points += len(cat_data)

                # Zähle Punkte in der Y-Region
                points_in_y_range = cat_data[
                    (cat_data >= region_y_min) & (cat_data <= region_y_max)
                    ]
                data_in_region += len(points_in_y_range)

        # Score: Niedriger ist besser (weniger Datenpunkte = mehr Platz)
        if total_points > 0:
            density = data_in_region / total_points
        else:
            density = 0

        # Berücksichtige auch Quantile für bessere Bewertung
        if region['y_min'] >= 0.7:  # Obere Bereiche
            q75_norm = (data_values.quantile(0.75) - y_min) / y_range
            if q75_norm < 0.6:  # Daten sind eher unten
                score = density * 0.5  # Bonus für obere Positionen
            else:
                score = density * 1.5  # Malus für obere Positionen
        else:  # Untere Bereiche
            q25_norm = (data_values.quantile(0.25) - y_min) / y_range
            if q25_norm > 0.4:  # Daten sind eher oben
                score = density * 0.5  # Bonus für untere Positionen
            else:
                score = density * 1.5  # Malus für untere Positionen

        position_scores[pos_name] = score

    # Finde Position mit niedrigstem Score (weniger Überlappung)
    best_position_name = min(position_scores.keys(), key=lambda k: position_scores[k])

    # Finde entsprechende Position
    for pos in positions:
        if pos[2] == best_position_name:
            return pos[:2]

    # Fallback
    return positions[0][:2]


def flatten_multiindex_columns(df):
    """Flacht MultiIndex-Spalten ab, falls vorhanden."""
    if isinstance(df.columns, pd.MultiIndex):
        # Echter MultiIndex
        df.columns = ['_'.join(str(col).strip() for col in multi_col if str(col).strip() != '')
                      for multi_col in df.columns.values]
    else:
        # Prüfe, ob Spalten String-Repräsentationen von Tupeln sind
        new_columns = []
        for col in df.columns:
            col_str = str(col)
            if col_str.startswith("('") and col_str.endswith("')"):
                # Parse Tupel-String: "('mean_order_completion_time', 'mean')" -> mean_order_completion_time_mean
                try:
                    # Entferne äußere Klammern und Anführungszeichen
                    inner = col_str[2:-2]  # Entferne "(' und ')"
                    parts = [part.strip().strip("'\"") for part in inner.split("', '")]
                    # Kombiniere nicht-leere Teile
                    combined = '_'.join(part for part in parts if part)
                    new_columns.append(combined)
                except:
                    # Falls Parsing fehlschlägt, verwende Original
                    new_columns.append(col_str)
            else:
                new_columns.append(col_str)
        df.columns = new_columns
    return df


def list_available_kpis(df):
    exclude_patterns = ['batch_capacity', 'waiting_strategy', 'routing_strategy', 'Layout',
                        'storage_assignment', 'instance', 'exp_num_orders', 'arrival_config',
                        'num_articles_in_order_config', 'due_date_config', 'article_config', 'storage_policy']

    # Erkenne Datenformat: Rohdaten oder aggregierte Daten
    has_mean_columns = any('_mean' in str(col) for col in df.columns)

    if has_mean_columns:
        # Aggregierte Daten: Finde KPI-Basis-Namen (ohne _mean, _std, etc.)
        kpi_bases = set()
        for col in df.columns:
            col_str = str(col)

            # Überspringe ausgeschlossene Spalten
            if any(excl in col_str for excl in exclude_patterns):
                continue

            # Entferne Suffixe wie _mean, _std, _ci_lower, _ci_upper, _n
            base_name = re.sub(r'_(mean|std|ci_lower|ci_upper|n)$', '', col_str)
            if base_name != col_str:  # Nur wenn ein Suffix entfernt wurde
                kpi_bases.add(base_name)

        return sorted(list(kpi_bases))
    else:
        # Rohdaten: Alle numerischen Spalten außer den ausgeschlossenen
        kpis = []
        for col in df.columns:
            col_str = str(col)

            # Überspringe ausgeschlossene Spalten
            if any(excl in col_str for excl in exclude_patterns):
                continue

            # Prüfe ob numerisch
            if pd.api.types.is_numeric_dtype(df[col]):
                kpis.append(col_str)

        return sorted(kpis)


def get_custom_waiting_order(unique_waiting):
    preferred_order = [
        'StartImmediatelyPolicy',
        'WaitingOrderCountPolicy_1',
        'WaitingOrderCountPolicy_2',
        'WaitingOrderCountPolicy_3',
        'WaitingOrderCountPolicy_4',
        'WaitingOrderCountPolicy_5',
        'WaitingOrderCountPolicy_6',
        'WaitingOrderCountPolicy_7',
        'Waiting'
    ]
    ordered = [s for s in preferred_order if s in unique_waiting]
    ordered += [s for s in unique_waiting if s not in ordered]
    return ordered


def get_global_routing_palette():
    """Definiert eine globale, konsistente Farbpalette für routing strategies."""
    global_routing_strategies = [
        'SShapeRouting', 'ReturnRouting', 'LargestGapRouting',
        'NearestNeighbourhoodRouting', 'ExactTSPRouting'
    ]
    colors = sns.color_palette("husl", len(global_routing_strategies))
    return dict(zip(global_routing_strategies, colors))


def get_routing_palette(routing_strategies):
    """Gibt konsistente Farben für routing strategies zurück."""
    global_palette = get_global_routing_palette()

    # Verwende globale Farben wenn verfügbar, sonst generiere neue
    palette = {}
    used_colors = []

    for strategy in sorted(routing_strategies):
        if strategy in global_palette:
            palette[strategy] = global_palette[strategy]
            used_colors.append(global_palette[strategy])

    # Für unbekannte Strategien: verwende verfügbare Farben
    remaining_colors = [c for c in sns.color_palette("husl", 20) if c not in used_colors]
    unknown_strategies = [s for s in sorted(routing_strategies) if s not in global_palette]

    for i, strategy in enumerate(unknown_strategies):
        if i < len(remaining_colors):
            palette[strategy] = remaining_colors[i]
        else:
            # Fallback: generiere neue Farbe
            palette[strategy] = sns.color_palette("husl", len(routing_strategies))[len(palette)]

    return palette


def get_color_palette_for_column(df, column):
    """Erstellt eine Farbpalette für beliebige Spalten."""
    unique_values = sorted(df[column].unique())
    if column == 'routing_strategy':
        return get_routing_palette(unique_values)
    else:
        colors = sns.color_palette("husl", len(unique_values))
        return dict(zip(unique_values, colors))


def prepare_strategy_label(df: pd.DataFrame) -> pd.DataFrame:
    waiting = [c for c in df.columns if 'waiting' in str(c).lower() and 'strategy' in str(c).lower()]
    routing = [c for c in df.columns if 'routing' in str(c).lower() and 'strategy' in str(c).lower()]

    print(f"Gefundene waiting columns: {waiting}")
    print(f"Gefundene routing columns: {routing}")

    if not waiting or not routing:
        print(f"Alle Spalten: {list(df.columns)}")
        raise KeyError("Waiting- or routing_strategy columns not found")
    df['waiting_strategy'] = df[waiting[0]].astype(str)
    df['routing_strategy'] = df[routing[0]].astype(str)
    return df


def apply_all_filters(df, args):
    """
    Wendet alle Filter basierend auf den Argumenten an
    """
    original_count = len(df)
    print(f"Originalanzahl Zeilen: {original_count}")

    # Filter-Mapping: argument_name -> column_name
    filter_mapping = {
        'batch_capacity': 'batch_capacity',
        'waiting': 'waiting_strategy',
        'routing': 'routing_strategy',
        'layout': 'Layout',
        'storage_assignment': 'storage_assignment',
        'exp_num_orders': 'exp_num_orders',
        'arrival_config': 'arrival_config',
        'num_articles_in_order_config': 'num_articles_in_order_config',
        'due_date_config': 'due_date_config',
        'article_config': 'article_config',
        'storage_policy': 'storage_policy'
    }

    # Numerische Filter (Gleichheit)
    numeric_filters = ['batch_capacity', 'exp_num_orders']

    # Listen-Filter (in-Operator)
    list_filters = ['waiting', 'routing', 'layout', 'storage_assignment',
                    'arrival_config', 'num_articles_in_order_config',
                    'due_date_config', 'article_config', 'storage_policy']

    for arg_name, col_name in filter_mapping.items():
        filter_value = getattr(args, arg_name, None)

        if filter_value is None:
            continue

        # Prüfe ob Spalte existiert
        if col_name not in df.columns:
            print(f"WARNUNG: Spalte '{col_name}' nicht im DataFrame gefunden. Filter übersprungen.")
            continue

        try:
            if arg_name in numeric_filters:
                # Numerische Filter (Gleichheit)
                df = df[df[col_name] == filter_value]
                print(f"Filter {col_name} == {filter_value}: {len(df)} Zeilen verbleiben")

            elif arg_name in list_filters:
                # Listen-Filter (isin)
                df = df[df[col_name].isin(filter_value)]
                print(f"Filter {col_name} in {filter_value}: {len(df)} Zeilen verbleiben")

        except Exception as e:
            print(f"FEHLER beim Anwenden des Filters {col_name}: {e}")
            continue

    # Generische Filter anwenden (falls implementiert)
    if hasattr(args, 'filter') and args.filter:
        df = apply_generic_filters(df, args.filter)

    filtered_count = len(df)
    if filtered_count == 0:
        print("WARNUNG: Nach Anwenden der Filter sind keine Daten mehr vorhanden!")
        print("Verfügbare eindeutige Werte pro Spalte:")
        original_df = pd.read_excel(args.input_box if hasattr(args, 'input_box') and args.input_box else args.input_bar)
        original_df = flatten_multiindex_columns(original_df)
        original_df = prepare_strategy_label(original_df)
        for col in filter_mapping.values():
            if col in original_df.columns:
                unique_vals = original_df[col].unique()
                print(
                    f"  {col}: {sorted(unique_vals) if len(unique_vals) <= 10 else f'{len(unique_vals)} unique values'}")
    else:
        print(f"Gesamt: {original_count} -> {filtered_count} Zeilen ({filtered_count / original_count * 100:.1f}%)")

    return df


def apply_generic_filters(df, filters):
    """
    Wendet generische Filter an
    Format: --filter column_name operator value
    """
    for filter_spec in filters:
        column, operator, value = filter_spec

        if column not in df.columns:
            print(f"WARNUNG: Spalte '{column}' nicht gefunden.")
            continue

        # Wert-Parsing
        if operator in ['in', 'not_in']:
            values = [v.strip() for v in value.split(',')]
            try:
                values = [float(v) if '.' in v else int(v) for v in values]
            except ValueError:
                pass
        else:
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                pass

        # Filter anwenden
        try:
            if operator == '==':
                df = df[df[column] == value]
            elif operator == '!=':
                df = df[df[column] != value]
            elif operator == '>':
                df = df[df[column] > value]
            elif operator == '<':
                df = df[df[column] < value]
            elif operator == '>=':
                df = df[df[column] >= value]
            elif operator == '<=':
                df = df[df[column] <= value]
            elif operator == 'in':
                df = df[df[column].isin(values)]
            elif operator == 'not_in':
                df = df[~df[column].isin(values)]

            print(f"Generischer Filter {column} {operator} {value}: {len(df)} Zeilen verbleiben")

        except Exception as e:
            print(f"FEHLER beim generischen Filter {column} {operator} {value}: {e}")

    return df


def list_filter_options(df_box, df_bar):
    """
    Zeigt verfügbare Filterwerte für alle relevanten Spalten an
    """
    filter_columns = ['batch_capacity', 'waiting_strategy', 'routing_strategy', 'Layout',
                      'storage_assignment', 'exp_num_orders', 'arrival_config',
                      'num_articles_in_order_config', 'due_date_config',
                      'article_config', 'storage_policy']

    print("Verfügbare Filterwerte:")
    print("=" * 60)

    # Verwende Box-Daten für Filter-Optionen (mehr Variabilität)
    df = df_box if df_box is not None else df_bar

    for col in filter_columns:
        if col in df.columns:
            unique_vals = sorted(df[col].unique())
            print(f"\n{col}:")
            if len(unique_vals) <= 15:
                for val in unique_vals:
                    print(f"  {val}")
            else:
                print(f"  {len(unique_vals)} unique values:")
                for i, val in enumerate(unique_vals[:10]):
                    print(f"  {val}")
                print(f"  ... und {len(unique_vals) - 10} weitere")
        else:
            print(f"\n{col}: SPALTE NICHT GEFUNDEN")


def apply_axis_break(ax, data, axis='y', threshold=0.05):
    """
    Wendet Achsenbruch an, wenn der minimale Wert deutlich größer als 0 ist.
    - ax: Matplotlib Achse
    - data: Array oder Series mit Werten
    - axis: 'x' oder 'y'
    - threshold: Prozentsatz für Bedingung (Default 5%)
    """
    vals = np.array(data.dropna() if hasattr(data, "dropna") else data)
    if len(vals) == 0:
        return  # keine Daten

    vmin, vmax = vals.min(), vals.max()

    if vmin > threshold * vmax:
        if axis == 'y':
            ax.set_ylim(vmin * 0.98, vmax * 1.1)
            # Marker für Achsenbruch (waagrechte Linie, oben/unten abgeschnitten)
            d = 0.5
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)

        elif axis == 'x':
            ax.set_xlim(vmin * 0.98, vmax * 1.1)
            # Marker für Achsenbruch (senkrechte Linie, links/rechts abgeschnitten)
            d = 0.5
            kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
                          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax.plot([0, 0], [0, 1], transform=ax.transAxes, **kwargs)


def create_boxplots(df, output_folder, kpis, legend_position='auto', figure_size_cm=(27, 13.5)):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    waiting_order = get_custom_waiting_order(df['waiting_strategy'].unique())
    routing_order = sorted(df['routing_strategy'].unique())
    routing_palette = get_routing_palette(routing_order)

    # Konvertiere cm zu Zoll für matplotlib
    fig_width, fig_height = get_figure_size(figure_size_cm[0], figure_size_cm[1])

    print("Erstelle Boxplots mit echtem Achsenbruch")
    for kpi in kpis:
        if kpi not in df.columns:
            print(f"Warnung: KPI '{kpi}' nicht gefunden. Übersprungen.")
            continue

        vals = df[kpi].dropna().values
        vmin, vmax = vals.min(), vals.max()

        # Achsenbruch wenn vmin deutlich > 0
        if vmin > 0.05 * vmax:
            fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex=True,
                                                     gridspec_kw={'height_ratios': [4, 0.2]},
                                                     figsize=(fig_width, fig_height))

            # Oberes Panel: Daten von vmin bis vmax mit etwas mehr Platz für Ausreißer
            sns.boxplot(
                data=df,
                x='waiting_strategy', y=kpi,
                hue='routing_strategy',
                order=waiting_order, hue_order=routing_order,
                palette=routing_palette,
                ax=ax_upper
            )
            # Vergrößerter Bereich für Ausreißer - 10% statt 5% zusätzlicher Platz
            ax_upper.set_ylim(vmin * 0.98, vmax * 1.1)
            ax_upper.spines['bottom'].set_visible(False)
            ax_upper.tick_params(bottom=False)
            ax_upper.set_ylabel(kpi)
            ax_upper.set_title(f"Boxplot: {kpi}")

            # Intelligente Legendenpositionierung
            if legend_position == 'auto':
                bbox_anchor, loc = find_best_legend_position(ax_upper, df, kpi)
            else:
                bbox_anchor, loc = legend_position

            ax_upper.legend(title='Routing Strategy', bbox_to_anchor=bbox_anchor, loc=loc,
                            framealpha=0.9, fancybox=True, shadow=True)

            # Unteres Panel
            ax_lower.set_ylim(-vmin * 0.02, vmin * 0.1)
            ax_lower.spines['top'].set_visible(False)

            # Keine Y-Achse-Skala unten außer Null
            ax_lower.yaxis.set_major_locator(plt.NullLocator())
            ax_lower.set_yticks([0])  # nur 0 anzeigen
            ax_lower.spines['left'].set_visible(True)

            ax_lower.set_ylabel('')
            ax_lower.tick_params(labeltop=False)
            ax_lower.tick_params(axis='x', rotation=45)
            ax_lower.set_xlabel('Waiting Strategy')

            # Achsenbruchlinien KORRIGIERT: parallele Linien mit 45°
            d = 0.5  # Proportion der diagonalen Linie
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none",
                          color='k', mec='k', mew=1, clip_on=False)
            ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
            ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)

            fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.92, hspace=0.05)

        else:
            # Normaler Boxplot ohne Achsenbruch
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.boxplot(
                data=df,
                x='waiting_strategy', y=kpi,
                hue='routing_strategy',
                order=waiting_order, hue_order=routing_order,
                palette=routing_palette,
                ax=ax
            )
            ax.set_ylabel(kpi)
            ax.set_title(f"Boxplot: {kpi}")

            # Intelligente Legendenpositionierung
            if legend_position == 'auto':
                bbox_anchor, loc = find_best_legend_position(ax, df, kpi)
            else:
                bbox_anchor, loc = legend_position

            ax.legend(title='Routing Strategy', bbox_to_anchor=bbox_anchor, loc=loc,
                      framealpha=0.9, fancybox=True, shadow=True)

            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('Waiting Strategy')

            fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.92)

        # Speichern
        output_path = output_folder / f'boxplot_{kpi}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Boxplot für {kpi} gespeichert: {output_path}")


def plot_bar_with_errorbars(df: pd.DataFrame, kpi: str, groupby: str, out: Path, legend_position='auto',
                            figure_size_cm=(27, 13.5)):
    """
    Erstellt Barplots aus aggregierten Daten mit Fehlerbalken
    """
    # Suche nach den entsprechenden Spalten für aggregierte Daten
    mean_cols = [c for c in df.columns if str(c).startswith(kpi) and str(c).endswith('_mean')]
    lower_cols = [c for c in df.columns if str(c).startswith(kpi) and 'ci_lower' in str(c)]
    upper_cols = [c for c in df.columns if str(c).startswith(kpi) and 'ci_upper' in str(c)]

    if not mean_cols or not lower_cols or not upper_cols:
        print(f"[WARN] KPI '{kpi}'-Spalten (mean/CI) nicht gefunden. Überspringe Barplot.")
        print(f"       Verfügbare Spalten: {[c for c in df.columns if str(c).startswith(kpi)]}")
        return

    mean_col = mean_cols[0]
    lower_col = lower_cols[0]
    upper_col = upper_cols[0]

    df = df.copy()
    df['_yerr_lower'] = df[mean_col] - df[lower_col]
    df['_yerr_upper'] = df[upper_col] - df[mean_col]

    # Bestimme Ordnung für groupby-Spalte
    if groupby == 'waiting_strategy':
        group_order = get_custom_waiting_order(df[groupby].unique())
    else:
        group_order = sorted(df[groupby].unique())

    routing_order = sorted(df['routing_strategy'].unique())
    routing_palette = get_routing_palette(routing_order)

    # Konvertiere cm zu Zoll für matplotlib
    fig_width, fig_height = get_figure_size(figure_size_cm[0], figure_size_cm[1])

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(
        data=df,
        x=groupby,
        y=mean_col,
        hue='routing_strategy',
        order=group_order,
        hue_order=routing_order,
        palette=routing_palette,
        errorbar=None
    )

    # Fehlerbalken hinzufügen
    n = len(routing_order)
    width = 0.8 / n
    for _, row in df.iterrows():
        if row[groupby] not in group_order:
            continue
        xi = group_order.index(row[groupby])
        hi = routing_order.index(row['routing_strategy'])
        offset = (hi - (n - 1) / 2) * width
        ax.errorbar(
            x=xi + offset,
            y=row[mean_col],
            yerr=[[row['_yerr_lower']], [row['_yerr_upper']]],
            fmt='none', c='black', capsize=3
        )

    # Intelligente Legendenpositionierung
    if legend_position == 'auto':
        bbox_anchor, loc = find_best_legend_position(ax, df, mean_col)
    else:
        bbox_anchor, loc = legend_position

    ax.legend(title='Routing Strategy', bbox_to_anchor=bbox_anchor, loc=loc,
              framealpha=0.9, fancybox=True, shadow=True)

    ax.set_title(f"Barplot: {kpi.replace('_', ' ').capitalize()} by {groupby.replace('_', ' ').capitalize()}")
    ax.set_xlabel(groupby.replace('_', ' ').capitalize())
    ax.set_ylabel(kpi.replace('_', ' ').capitalize())
    ax.tick_params(axis='x', rotation=45)

    # Manuelle Anpassung der Ränder statt tight_layout für exakte Größe
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)

    out.mkdir(parents=True, exist_ok=True)
    file = out / f"barplot_{kpi}_by_{groupby}.png"
    plt.savefig(file, dpi=150)  # bbox_inches='tight' entfernt für exakte Größe
    plt.close()
    print(f"Barplot für KPI '{kpi}' gespeichert: {file.name}")


def create_tradeoff_plot(df, output_folder, figure_size_cm=(30, 12.5), legend_position='auto'):
    """
    Trade-off: x = mean_order_completion_time_mean, y = mean_length_per_item_mean
               Bubble-Größe = mean_tardiness_mean
               Farbe        = routing_strategy
               Label        = waiting_strategy
    Zeichnet Achsenbrüche auf X und Y, beide Achsen beginnen bei 0.
    Verbesserungen:
     - Bubble-size legend oben rechts mit größeren Abständen zwischen Bubbles
     - Labels (waiting_strategy) werden mit adjustText optimiert (falls installiert), sonst heuristisch platziert
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # ---- Spaltennamen
    x_col = "mean_order_completion_time_mean"
    y_col = "mean_length_per_item_mean"
    s_col = "mean_tardiness_mean"
    w_col = "waiting_strategy"
    r_col = "routing_strategy"

    # Guard: fehlende Spalten melden
    need_cols = [x_col, y_col, s_col, w_col, r_col]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Für den Trade-off-Plot fehlen Spalten: {missing}. Abbruch.")
        return

    # ---- Farbpalette (Routing)
    routing_order = sorted(df[r_col].dropna().unique())
    routing_palette = get_routing_palette(routing_order)  # erwartet dict: strat -> Farbe
    colors = df[r_col].map(routing_palette)

    # ---- Bubble-Größe (Tardiness) – dynamisch & deutlich größer (Flächen-Angaben für scatter)
    t = df[s_col].fillna(0).astype(float)
    if t.max() > t.min():
        t_norm = (t - t.min()) / (t.max() - t.min())
        sizes = 300 + t_norm * (2400 - 300)  # 300..2400 Punkte (Fläche)
    else:
        sizes = pd.Series(800, index=df.index)  # Konstante Größe, falls keine Varianz

    # ---- Achsen-Bereiche & Break-Definition
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    use_x_break = (x_min > 0) and (x_min > 0.05 * x_max)
    use_y_break = (y_min > 0) and (y_min > 0.05 * y_max)

    if use_x_break:
        x_low_max = max(0.15 * x_min, 1e-9)
        x_high_min = 0.92 * x_min
        x_segments = [(0.0, x_low_max), (x_high_min, 1.25 * x_max)]
    else:
        x_segments = [(0.0, 1.25 * x_max)]

    if use_y_break:
        y_low_max = max(0.15 * y_min, 1e-9)
        y_high_min = 0.92 * y_min
        y_segments = [(0.0, y_low_max), (y_high_min, 1.25 * y_max)]
    else:
        y_segments = [(0.0, 1.25 * y_max)]

    # ---- Figure size (cm -> Zoll)
    fig_w, fig_h = get_figure_size(figure_size_cm[0], figure_size_cm[1])

    # ---- Broken axes erstellen
    bax = brokenaxes(
        xlims=tuple(x_segments),
        ylims=tuple(y_segments),
        wspace=0.05,
        hspace=0.05
    )
    fig = bax.fig
    fig.set_size_inches(fig_w, fig_h)

    # ---- Scatter
    sc = bax.scatter(
        df[x_col].values,
        df[y_col].values,
        s=sizes.values,
        c=colors.values,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )

    # ---- Labels (waiting_strategy) – heuristische Platzierung + adjustText-Fallback
    axes_list = []
    for row_axes in bax.axs:
        if isinstance(row_axes, (list, tuple, np.ndarray)):
            for a in row_axes:
                axes_list.append(a)
        else:
            axes_list.append(row_axes)

    def _find_axis_for_point(x, y):
        for ax in axes_list:
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            if (x >= min(xl) - 1e-9) and (x <= max(xl) + 1e-9) and (y >= min(yl) - 1e-9) and (y <= max(yl) + 1e-9):
                return ax
        return axes_list[-1]

    texts_by_axis = defaultdict(list)
    used_positions = []

    xs = df[x_col].astype(float).values
    ys = df[y_col].astype(float).values
    x_range = (xs.max() - xs.min()) if len(xs) > 0 else 1.0
    y_range = (ys.max() - ys.min()) if len(ys) > 0 else 1.0
    base_offset_x = x_range * 0.03
    base_offset_y = y_range * 0.03

    offset_directions = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (2, 1), (1, 2), (-2, -1), (-1, -2), (2, 0), (-2, 0)
    ]

    for i, (_, row) in enumerate(df.iterrows()):
        x = float(row[x_col])
        y = float(row[y_col])
        label = str(row[w_col]) if pd.notna(row[w_col]) else ""

        ax = _find_axis_for_point(x, y)

        chosen = None
        for d in offset_directions:
            cand_x = x + d[0] * base_offset_x
            cand_y = y + d[1] * base_offset_y
            too_close = any(((cand_x - px)**2 / (x_range**2) + (cand_y - py)**2 / (y_range**2)) < (0.02**2) for (px, py) in used_positions)
            if not too_close:
                chosen = (cand_x, cand_y)
                break

        if chosen is None:
            jitter_x = (np.random.rand() - 0.5) * base_offset_x * 0.6
            jitter_y = (np.random.rand() - 0.5) * base_offset_y * 0.6
            chosen = (x + jitter_x, y + jitter_y)

        used_positions.append(chosen)

        txt = ax.text(
            chosen[0], chosen[1],
            label,
            ha='center', va='center',
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9,
                      edgecolor='gray', linewidth=0.5),
            zorder=10
        )
        texts_by_axis[ax].append(txt)

    # ----- IMPORTANT: arrowprops definieren und an adjust_text übergeben -----
    # Erhöhe shrinkA von 10 auf 15 um die brokenaxes Transform-Warnung zu beheben
    arrowprops = dict(
        arrowstyle='-',            # '-' = Linie ohne Pfeilspitze, '->' = mit Spitze
        color='gray',
        lw=0.6,
        shrinkA=20,                # Erhöht von 10 auf 15 um Warnung zu beheben
        shrinkB=5,                 # Abstand vom Datenpunkt
        connectionstyle="arc3,rad=0.0"  # leichte Gerade; rad=0.1 macht es gebogen
    )

    # Versuche, adjustText zu verwenden (pro Achse). Wenn die Arrow-Implementierung intern
    # auf annotate zurückfällt, helfen shrinkA/shrinkB in vielen Fällen.
    try:
        from adjustText import adjust_text
        for ax, text_list in texts_by_axis.items():
            if not text_list:
                continue
            # adjust_text benötigt Text-Objekte; wir erlauben Verschiebung in x+y für Texte,
            # und wir geben arrowprops mit shrinkA/shrinkB.
            adjust_text(
                text_list,
                ax=ax,
                expand_text=(1.2, 1.2),
                expand_points=(1.2, 1.2),
                force_text=0.5,
                force_points=0.2,
                arrowprops=arrowprops,
                only_move={'points': 'y', 'texts': 'xy'}
            )
    except Exception:
        # Falls adjustText nicht verfügbar ist oder ein Fehler auftritt: Fallback bereits oben angewendet
        print("[INFO] adjustText nicht verfügbar oder Fehler beim Anwenden. Verwende heuristisches Offsetting. "
              "Für bessere Ergebnisse: pip install adjustText")

    # ---- Achsenlabels & Titel
    bax.set_xlabel("Mean order completion time")
    bax.set_ylabel("Mean length per item")
    fig.suptitle("Trade-off: completion time vs. length/item (Bubble = tardiness)")

    # ---- Legende: Routing-Strategien (Farben)
    handles = [
        Line2D([0], [0], marker='o', color='w',
               label=strat, markerfacecolor=routing_palette[strat],
               markeredgecolor='k', markeredgewidth=0.5, markersize=8)
        for strat in routing_order
    ]

    bbox_anchor = (0.98, 0.02)
    loc = 'lower right'

    if hasattr(bax, 'axs') and len(bax.axs) > 0:
        bax.axs[-1].legend(handles=handles, title="Routing Strategy",
                           bbox_to_anchor=bbox_anchor, loc=loc,
                           framealpha=0.9, fancybox=True, shadow=True)
    else:
        fig.legend(handles=handles, title="Routing Strategy",
                   bbox_to_anchor=bbox_anchor, loc=loc,
                   framealpha=0.9, fancybox=True, shadow=True)

    # ---- Größenreferenz für Bubbles (Tardiness) - verbessert mit mehr Abstand und z-order
    t_vals = df[s_col].fillna(0).astype(float)
    if t_vals.max() > t_vals.min():
        t_min, t_max = t_vals.min(), t_vals.max()
        t_mid = (t_min + t_max) / 2
        ref_values = [round(t_min, 3), round(t_mid, 3), round(t_max, 3)]

        ref_sizes = []
        for val in ref_values:
            if t_vals.max() > t_vals.min():
                val_norm = (val - t_vals.min()) / (t_vals.max() - t_vals.min())
                size = 300 + val_norm * (2400 - 300)
            else:
                size = 800
            ref_sizes.append(size)

        # Bubble Size Legende - deutlich größeres Feld und optimierte Bubble-Verteilung
        legend_ax = fig.add_axes([0.75, 0.50, 0.24, 0.45])  # Größer und tiefer positioniert
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis('off')
        legend_ax.set_zorder(1000)  # Sehr hoher z-order für Vordergrund

        legend_ax.text(0.05, 0.95, 'Bubble Size:',
                       fontsize=10, fontweight='bold', va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.9),
                       zorder=1001)  # Noch höherer z-order für Text

        # Optimierte Y-Positionen mit deutlich mehr Abstand - keine Überschneidungen mehr
        y_positions = [0.78, 0.50, 0.22]  # Noch weiter auseinander für große Bubbles

        for val, size, y_pos in zip(ref_values, ref_sizes, y_positions):
            legend_ax.scatter(0.20, y_pos, s=size, c='gray', alpha=0.6,
                              edgecolors='black', linewidths=0.5, zorder=1002)  # Hoher z-order
            legend_ax.text(0.45, y_pos, f'{val:.3f}',
                           fontsize=8, va='center', ha='left',
                           bbox=dict(boxstyle="round,pad=0.15", facecolor='white', alpha=0.9),
                           zorder=1003)  # Höchster z-order für Labels

    # ---- speichern
    out_path = output_folder / "tradeoff_plot.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Trade-off-Plot gespeichert: {out_path}")


def load_and_process_data(file_path, args, data_type):
    """
    Lädt und verarbeitet Daten für einen bestimmten Plot-Typ
    """
    if file_path is None:
        return None

    print(f"\n=== Lade {data_type}-Daten ===")
    df = pd.read_excel(file_path)
    print(f"Original {data_type} DataFrame shape: {df.shape}")

    df = flatten_multiindex_columns(df)
    df = prepare_strategy_label(df)

    # Filter anwenden
    df = apply_all_filters(df, args)
    print(f"Nach Filtern {data_type} DataFrame shape: {df.shape}")

    return df


def main():
    # ANPASSBARE DIAGRAMM-GRÖSSEN (in cm) - Hier können Sie die Werte ändern:
    BOXPLOT_SIZE_CM = (30, 12.5)  # Breite x Höhe für Boxplots
    BARPLOT_SIZE_CM = (30, 12.5)  # Breite x Höhe für Barplots

    parser = argparse.ArgumentParser(description="Generator für Box- & Barplots mit separaten Inputs")

    # SEPARATE INPUTS FÜR BOX- UND BARPLOTS
    parser.add_argument('--input-box', help='Pfad zur Excel-Datei mit Rohdaten (für Boxplots)')
    parser.add_argument('--input-bar', help='Pfad zur Excel-Datei mit aggregierten Daten (für Barplots)')
    parser.add_argument('-o', '--output', required=True, help='Ordner für Plots')

    # Informations-Optionen
    parser.add_argument('--list-kpis', action='store_true', help='Nur KPIs listen')
    parser.add_argument('--list-filters', action='store_true', help='Verfügbare Filterwerte anzeigen')

    # KPI-Auswahl
    parser.add_argument('--kpis', nargs='+', help='KPIs auswählen')

    # ALLE SPEZIFISCHEN FILTER
    parser.add_argument('--batch_capacity', type=int, help='Filter batch_capacity (Zahl)')
    parser.add_argument('--waiting', nargs='+', help='Filter waiting_strategy (Liste)')
    parser.add_argument('--routing', nargs='+', help='Filter routing_strategy (Liste)')
    parser.add_argument('--layout', nargs='+', help='Filter Layout (Liste)')
    parser.add_argument('--storage_assignment', nargs='+', help='Filter storage_assignment (Liste)')
    parser.add_argument('--exp_num_orders', type=int, help='Filter exp_num_orders (Zahl)')
    parser.add_argument('--arrival_config', nargs='+', help='Filter arrival_config (Liste)')
    parser.add_argument('--num_articles_in_order_config', nargs='+', help='Filter num_articles_in_order_config (Liste)')
    parser.add_argument('--due_date_config', nargs='+', help='Filter due_date_config (Liste)')
    parser.add_argument('--article_config', nargs='+', help='Filter article_config (Liste)')
    parser.add_argument('--storage_policy', nargs='+', help='Filter storage_policy (Liste)')

    # Generische Filter
    parser.add_argument('--filter', nargs=3, action='append', metavar=('COLUMN', 'OPERATOR', 'VALUE'),
                        help='Generischer Filter: --filter column_name == value')

    # Plot-Optionen
    parser.add_argument('--plot-type', choices=['box', 'bar', 'both'], default='both',
                        help='Art des Plots: box, bar oder both')
    parser.add_argument('--groupby', default='waiting_strategy', help='Spalte zum Gruppieren für Barplots')
    parser.add_argument('--legend-position', default='auto',
                        choices=['auto', 'upper_right', 'upper_left', 'lower_right', 'lower_left', 'upper_center',
                                 'outside'],
                        help='Position der Legende: auto (automatisch), upper_right, upper_left, lower_right, lower_left, upper_center, outside')

    args = parser.parse_args()

    # Validierung der Inputs
    if not args.input_box and not args.input_bar:
        print("FEHLER: Mindestens --input-box oder --input-bar muss angegeben werden!")
        return

    if args.plot_type == 'box' and not args.input_box:
        print("FEHLER: Für Boxplots wird --input-box benötigt!")
        return

    if args.plot_type == 'bar' and not args.input_bar:
        print("FEHLER: Für Barplots wird --input-bar benötigt!")
        return

    if args.plot_type == 'both' and (not args.input_box or not args.input_bar):
        print("FEHLER: Für beide Plot-Typen werden sowohl --input-box als auch --input-bar benötigt!")
        return

    # Daten laden und verarbeiten
    df_box = load_and_process_data(args.input_box, args, "Box") if args.input_box else None
    df_bar = load_and_process_data(args.input_bar, args, "Bar") if args.input_bar else None

    # Informations-Modi
    if args.list_filters:
        list_filter_options(df_box, df_bar)
        return

    if args.list_kpis:
        print("Verfügbare KPIs:")
        if df_box is not None:
            print("\n--- Aus Boxplot-Daten (Rohdaten) ---")
            box_kpis = list_available_kpis(df_box)
            for k in box_kpis:
                print(f"  {k}")
        if df_bar is not None:
            print("\n--- Aus Barplot-Daten (aggregiert) ---")
            bar_kpis = list_available_kpis(df_bar)
            for k in bar_kpis:
                print(f"  {k}")
        return

    # Prüfe ob nach Filterung noch Daten vorhanden
    if df_box is not None and len(df_box) == 0:
        print("WARNUNG: Keine Boxplot-Daten nach Filterung vorhanden.")
        df_box = None

    if df_bar is not None and len(df_bar) == 0:
        print("WARNUNG: Keine Barplot-Daten nach Filterung vorhanden.")
        df_bar = None

    if df_box is None and df_bar is None:
        print("FEHLER: Keine Daten nach Filterung vorhanden. Abbruch.")
        return

    # KPIs bestimmen - verwende verfügbare Daten
    if args.kpis:
        kpis = args.kpis
    else:
        # Automatische KPI-Bestimmung aus verfügbaren Daten
        if df_box is not None:
            kpis = list_available_kpis(df_box)
        else:
            kpis = list_available_kpis(df_bar)

    print(f"\nZu plottende KPIs: {kpis}")

    # Plots erstellen
    legend_pos = get_legend_position(args.legend_position)

    if args.plot_type in ['box', 'both'] and df_box is not None:
        print(f"\n=== Erstelle Boxplots ===")
        create_boxplots(df_box, Path(args.output) / 'boxplots', kpis, legend_pos, BOXPLOT_SIZE_CM)

    if args.plot_type in ['bar', 'both'] and df_bar is not None:
        print(f"\n=== Erstelle Barplots ===")
        bar_out = Path(args.output) / 'barplots'
        for k in kpis:
            plot_bar_with_errorbars(df_bar, k, args.groupby, bar_out, legend_pos, BARPLOT_SIZE_CM)

    if df_bar is not None:
        print(f"\n=== Erstelle Trade-off-Plot ===")
        tradeoff_out = Path(args.output) / 'tradeoff'
        create_tradeoff_plot(df_bar, tradeoff_out, BARPLOT_SIZE_CM, legend_pos)

    print(f"\n=== Fertig! Plots gespeichert in: {args.output} ===")


if __name__ == '__main__':
    main()