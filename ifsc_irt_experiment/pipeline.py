from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


REQUIRED_COLUMNS = {'athlete', 'event', 'year', 'round', 'problem', 'topped'}
KAGGLE_DATASET = 'mxmlnv/ifsc-competition-climbing'


def download_ifsc_dataset() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().lower() for c in df.columns})


def _find_candidate_csv_files(data_path: Path) -> Iterable[Path]:
    if data_path.is_file() and data_path.suffix.lower() == '.csv':
        return [data_path]
    if data_path.is_dir():
        return sorted(data_path.glob('*.csv'))
    raise FileNotFoundError(f'Data path not found: {data_path}')


def load_ifsc_dataframe(data_path: Path) -> pd.DataFrame:
    candidates = _find_candidate_csv_files(data_path)
    for csv_file in candidates:
        df = pd.read_csv(csv_file)
        normalized = _normalize_columns(df)
        if REQUIRED_COLUMNS.issubset(set(normalized.columns)):
            return normalized
    raise ValueError(
        f'No CSV with required columns {sorted(REQUIRED_COLUMNS)} found under: {data_path}'
    )


def prepare_matrix(df: pd.DataFrame, round_filter: str, missing_mode: str, min_attempts: int) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    if 'discipline' in df.columns:
        boulder = df[df['discipline'].astype(str).str.lower() == 'boulder'].copy()
    else:
        boulder = df.copy()

    if round_filter != 'all':
        boulder = boulder[boulder['round'].astype(str).str.lower() == round_filter]

    boulder['problem_id'] = (
        boulder['event'].astype(str) + '_' +
        boulder['year'].astype(str) + '_' +
        boulder['round'].astype(str) + '_' +
        boulder['problem'].astype(str)
    )

    matrix = boulder.pivot_table(index='athlete', columns='problem_id', values='topped', aggfunc='max')

    if missing_mode == 'zero':
        matrix = matrix.fillna(0)

    problem_sum = matrix.sum(axis=0, skipna=True)
    matrix = matrix.loc[:, (problem_sum > 0) & (problem_sum < len(matrix))]

    attempt_count = matrix.notna().sum(axis=1) if missing_mode == 'nan' else matrix.shape[1]
    return matrix.loc[attempt_count >= min_attempts]


def fit_2pl(matrix: pd.DataFrame, epochs: int, learning_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = torch.tensor(matrix.values, dtype=torch.float32)
    mask = ~torch.isnan(r)
    filled = torch.nan_to_num(r, nan=0.0)

    n_climbers, n_problems = filled.shape
    theta = torch.nn.Parameter(torch.zeros(n_climbers))
    alpha_raw = torch.nn.Parameter(torch.zeros(n_problems))
    beta = torch.nn.Parameter(torch.zeros(n_problems))

    optim = torch.optim.Adam([theta, alpha_raw, beta], lr=learning_rate)

    for _ in range(epochs):
        optim.zero_grad()
        alpha = torch.nn.functional.softplus(alpha_raw) + 1e-4
        logits = alpha.unsqueeze(0) * (theta.unsqueeze(1) - beta.unsqueeze(0))
        log_p = torch.nn.functional.logsigmoid(logits)
        log_1mp = torch.nn.functional.logsigmoid(-logits)
        ll = torch.where(filled == 1, log_p, log_1mp)
        ll = (ll * mask).sum()

        reg = 0.5 * (theta.pow(2).mean() + beta.pow(2).mean())
        loss = -ll / mask.sum().clamp(min=1) + 0.01 * reg
        loss.backward()
        optim.step()

    alpha = (torch.nn.functional.softplus(alpha_raw) + 1e-4).detach().cpu().numpy()
    return theta.detach().cpu().numpy(), alpha, beta.detach().cpu().numpy()


def run_experiment(
    data_path: Path,
    output_dir: Path,
    missing_mode: str,
    round_filter: str,
    epochs: int,
    learning_rate: float,
    min_attempts: int,
    use_kagglehub: bool,
) -> None:
    if use_kagglehub:
        data_path = download_ifsc_dataset()

    df = load_ifsc_dataframe(data_path)
    matrix = prepare_matrix(df, round_filter=round_filter, missing_mode=missing_mode, min_attempts=min_attempts)

    if matrix.empty:
        raise ValueError('No data left after filtering.')

    theta, alpha, beta = fit_2pl(matrix, epochs=epochs, learning_rate=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / 'matrix_used.csv')
    pd.DataFrame({'athlete': matrix.index, 'theta': theta}).sort_values('theta', ascending=False).to_csv(output_dir / 'theta.csv', index=False)
    problem_df = pd.DataFrame({'problem_id': matrix.columns, 'alpha': alpha, 'beta': beta})
    problem_df.to_csv(output_dir / 'problem_params.csv', index=False)

    alpha_std = float(problem_df['alpha'].std())
    with open(output_dir / 'summary.md', 'w', encoding='utf-8') as f:
        f.write('# Experiment summary\n\n')
        f.write(f'- missing_mode: {missing_mode}\n')
        f.write(f'- round_filter: {round_filter}\n')
        f.write(f'- matrix_shape: {matrix.shape}\n')
        f.write(f'- alpha_std: {alpha_std:.6f}\n')
        f.write('- spearman_theta_vs_official: TODO (provide official ranking source)\n')
