"""
Skript zum Neuberechnen der Summary-Dateien aus vorhandenen batch_results.parquet
Nutzt die bestehenden vollständigen Daten - keine Neuberechnung der Simulation nötig!
"""

import pandas as pd
from pathlib import Path
from scipy import stats
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_stats(series: pd.Series, alpha: float = 0.05):
    """Returns mean statistics and 95%-CI for a series."""
    data = series.dropna()
    n = data.size
    if n < 2:
        return pd.Series({
            'mean': data.mean() if n == 1 else np.nan,
            'std': np.nan,
            'sum': data.sum() if n == 1 else np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n': n
        })
    mean = data.mean()
    std = data.std(ddof=1)
    sem = std / np.sqrt(n)
    ci_low, ci_high = stats.t.interval(1 - alpha, df=n-1, loc=mean, scale=sem)
    return pd.Series({
        'mean': mean,
        'std': std,
        'sum': data.sum(),
        'ci_lower': ci_low,
        'ci_upper': ci_high,
        'n': n
    })

def analysis_and_report_new(agg_df: pd.DataFrame, output_folder: Path, timestamp: str):
    """Enhanced analysis with instance meta information and 95%-CIs via compute_stats."""

    meta_cols = [
        'layout',
        'storage_assignment',
        'exp_num_orders',
        'arrival_config',
        'num_articles_in_order_config',
        'due_date_config',
        'article_config',
        'storage_policy'
    ]

    agg_merged = agg_df

    # define grouping columns
    stats_group = ['batch_capacity', 'nr_vehicles', 'waiting_strategy', 'routing_strategy'] + meta_cols

    kpi_cols = [
        'mean_order_completion_time', 'mean_tardiness',
        'mean_length', 'mean_duration',
        'mean_length_per_item', 'mean_duration_per_item',
        'mean_dur_walk', 'mean_dur_wait',
        'mean_walk_per_item', 'mean_wait_per_item', 'utilisation',
        'count_orders', 'count_batches'
    ]

    existing_kpi_cols = [col for col in kpi_cols if col in agg_merged.columns]

    # CI- and statistics calculations per strategy
    stats_df = (
        agg_merged
        .groupby(stats_group)[existing_kpi_cols]
        .apply(lambda grp: grp.apply(compute_stats))
        .unstack(level=-1)  # flatten MultiIndex-Spalten
        .reset_index()
    )

    # Export mit new_ Präfix
    summary_file = output_folder / f'new_summary_kpi_full_{timestamp}.parquet'
    stats_df.to_parquet(summary_file, index=False, engine="pyarrow", compression="snappy")
    logger.info(f"Strategy comparison (incl. CI, std, mean) saved: {summary_file}")

    return stats_df


def main():
    # ANPASSEN: Pfad zu Ihrem Results-Ordner (der all_paramsets enthält)
    results_base = Path(r'U:/Diss/Routing_uncertainty/Results')

    # Navigiere zu all_paramsets Ordner
    all_paramsets_folder = results_base / 'all_paramsets_20251017_130637_results'

    if not all_paramsets_folder.exists():
        logger.error(f"Ordner nicht gefunden: {all_paramsets_folder}")
        logger.info("Bitte passen Sie den Pfad 'results_base' im Skript an!")
        return

    # Finde alle Unterordner in all_paramsets
    subfolders = sorted([f for f in all_paramsets_folder.iterdir() if f.is_dir()])

    if not subfolders:
        logger.error(f"Keine Unterordner gefunden in {all_paramsets_folder}")
        return

    logger.info(f"Gefunden: {len(subfolders)} Unterordner in {all_paramsets_folder.name}")
    logger.info("="*80)

    success_count = 0
    error_count = 0

    for subfolder in subfolders:
        logger.info(f"\nVerarbeite: {subfolder.name}")

        # Finde die batch_results.parquet Datei
        parquet_files = list(subfolder.glob("batch_*_results.parquet"))

        if not parquet_files:
            logger.warning(f"  Keine batch_*_results.parquet Datei gefunden in {subfolder.name}")
            error_count += 1
            continue

        if len(parquet_files) > 1:
            logger.warning(f"  Mehrere parquet-Dateien gefunden, nutze erste: {parquet_files[0].name}")

        parquet_file = parquet_files[0]
        logger.info(f"  Lade: {parquet_file.name}")

        # Lade die vollständigen Daten
        try:
            batch_df = pd.read_parquet(parquet_file)
            logger.info(f"  Geladen: {len(batch_df)} Zeilen")

            # Finde vorhandene summary_kpi_full Dateien
            existing_summaries = list(subfolder.glob("summary_kpi_full_*.parquet"))

            if existing_summaries:
                # Extrahiere Timestamp aus vorhandener Summary-Datei
                original_summary = existing_summaries[0]
                original_name = original_summary.stem  # ohne .parquet
                # Entferne "summary_kpi_full_" Präfix
                timestamp = original_name.replace("summary_kpi_full_", "")
                logger.info(f"  Verwende Timestamp aus vorhandener Summary: {timestamp}")
            else:
                # Fallback: nutze Timestamp aus batch_results Dateinamen
                timestamp = parquet_file.stem.replace("batch_", "").replace("_results", "")
                logger.info(f"  Verwende Timestamp aus batch_results: {timestamp}")

            # Berechne Summary neu mit neuem Namen
            stats_df = analysis_and_report_new(batch_df, subfolder, timestamp)

            # Zeige Beispiel-Statistik
            try:
                # Versuche mean_order_completion_time n-Spalte zu finden
                moc_n_col = None
                for col in stats_df.columns:
                    if 'mean_order_completion_time' in str(col) and 'n' in str(col):
                        moc_n_col = col
                        break

                if moc_n_col is not None:
                    sample = stats_df[['waiting_strategy', 'routing_strategy', moc_n_col]].head(3)
                    logger.info(f"\n  Beispiel n-Werte für mean_order_completion_time:\n{sample}")
                else:
                    logger.info(f"  Anzahl Zeilen in Summary: {len(stats_df)}")
            except Exception as e:
                logger.info(f"  Anzahl Zeilen in Summary: {len(stats_df)}")

            logger.info(f"  ✓ Summary erfolgreich neu erstellt!")
            success_count += 1

        except Exception as e:
            logger.error(f"  Fehler beim Verarbeiten von {subfolder.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            error_count += 1
            continue

    logger.info("\n" + "="*80)
    logger.info("FERTIG!")
    logger.info(f"Erfolgreich verarbeitet: {success_count}")
    logger.info(f"Fehler: {error_count}")
    logger.info("Die neuen Dateien heißen: new_summary_kpi_full_*.parquet")
    logger.info("="*80)


if __name__ == "__main__":
    main()