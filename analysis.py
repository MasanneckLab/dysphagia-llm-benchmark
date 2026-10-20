"""Offline, reproducible revision analyses; never queries a model API."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from prepare_data import DOMAINS, HERE, MODELS, build_data

SEED = 20261006
REPLICATES = 20000


def interval(values):
    return tuple(float(x) for x in np.nanquantile(values, [.025, .975]))


def kappa(table, weight='unweighted'):
    table = np.asarray(table)
    n = table.sum(axis=(-1, -2))
    dist = abs(np.arange(3)[:, None] - np.arange(3)[None, :])
    if weight == 'unweighted':
        dist = (dist != 0).astype(float)
    elif weight == 'quadratic':
        dist = dist ** 2
    elif weight != 'linear':
        raise ValueError(weight)
    expected = table.sum(axis=-1)[..., :, None] * table.sum(axis=-2)[..., None, :] / n[..., None, None]
    return 1 - (table * dist).sum(axis=(-1, -2)) / (expected * dist).sum(axis=(-1, -2))


def bootstrap_indices(replicates=REPLICATES):
    rng = np.random.default_rng(SEED)
    # Preserve the deliberately fixed allocation of 15 topic pairs per domain.
    return np.concatenate([rng.integers(15 * d, 15 * (d + 1), size=(replicates, 15)) for d in range(3)], axis=1)


def save_figure(fig, stem):
    for ext in ('pdf', 'png', 'tiff'):
        fig.savefig(HERE / f'figures/{stem}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def figures():
    sns.set_theme(font_scale=1, style='whitegrid', font='DejaVu Sans')
    # Read the actual exported tables, rather than maintaining a second plotting dataset.
    support = pd.read_csv(HERE / 'results/source_support.csv').dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    labels = ['GPT-5.2', 'Open Evidence', 'Sonar-RP', 'Sonar']
    colors = sns.cubehelix_palette(6, rot=-.25, light=.7)
    for j, (ax, endpoint) in enumerate(zip(axes, ['guideline', 'high_evidence'])):
        ds = support[support.endpoint == endpoint].set_index('system').loc[MODELS]
        ax.bar(np.arange(4), ds.conditional_proportion * 100, width=.64, color=colors[4])
        ax.errorbar(np.arange(4), ds.conditional_proportion * 100,
                    yerr=np.array([ds.conditional_proportion - ds.conditional_ci_low,
                                   ds.conditional_ci_high - ds.conditional_proportion]) * 100,
                    fmt='none', color='dimgrey', capsize=4, lw=1)
        for x, row in enumerate(ds.itertuples()):
            ax.text(x, max(row.conditional_ci_high, row.conditional_proportion) * 100 + 2,
                    f'{row.n_supported}/{row.n_correct}\n{100 * row.conditional_proportion:.1f}%',
                    ha='center', va='bottom', fontsize=9, color='dimgrey')
        ax.set_xticks(np.arange(4), labels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(['(a) Complete guideline support', '(b) Inclusive high-level evidence'][j], fontsize=11, loc='left')
        ax.grid(False)
        ax.tick_params(length=0, labelcolor='dimgrey')
        ax.patch.set_edgecolor('lightgrey')
        ax.patch.set_linewidth(.8)
    axes[0].set_ylabel('Source support among\nCorrect responses (%)', fontsize=10)
    fig.text(.5, -.025, 'Descriptive endpoints; 95% topic-pair bootstrap intervals; unequal conditional denominators.',
             ha='center', fontsize=9, color='dimgrey')
    sns.despine(left=True, bottom=True)
    save_figure(fig, 'Figure4_source_support_no_significance')

    correctness = pd.read_csv(HERE / 'results/correctness.csv').dropna()
    totals = correctness[correctness.group == 'Overall'].set_index('system').loc[MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, field, low, high, title in [
        (axes[0], 'proportion', 'cluster_ci_low', 'cluster_ci_high', '(a) Guideline-concordant responses'),
        (axes[1], 'joint_proportion', 'joint_ci_low', 'joint_ci_high', '(b) Correct AND guideline-supported')]:
        ds = totals if field == 'proportion' else support[support.endpoint == 'guideline'].set_index('system').loc[MODELS]
        yy = np.arange(4)[::-1]
        ax.errorbar(ds[field] * 100, yy, xerr=np.array([ds[field] - ds[low], ds[high] - ds[field]]) * 100,
                    fmt='o', color=colors[5], capsize=4, lw=1.5)
        ax.set_yticks(yy, MODELS, fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Proportion of all 90 responses per system (%)')
        ax.set_title(title, fontsize=11, loc='left')
        ax.grid(False)
        ax.tick_params(length=0, labelcolor='dimgrey')
    fig.tight_layout()
    sns.despine(left=True, bottom=True)
    save_figure(fig, 'Correctness_and_joint_endpoint')

    # Preserve the manuscript's established category semantics: blue, yellow, red.
    rating_colors = ['#4575b4', '#fee090', '#d73027']
    cat_names = ['Correct', 'Inaccurate/Incomplete', 'Wrong']

    def stacked(ax, table, denominator, title):
        bottom = np.zeros(4)
        for category, color, name in zip(['correct', 'inaccurate', 'wrong'], rating_colors, cat_names):
            values = table[category].to_numpy() / denominator * 100
            ax.bar(np.arange(4), values, bottom=bottom, color=color, label=name, width=.65)
            for x, value in enumerate(values):
                if value >= 8:
                    ax.text(x, bottom[x] + value / 2, f'{value:.1f}%', ha='center', va='center',
                            color='white' if category != 'inaccurate' else 'dimgrey', fontsize=8)
            bottom += values
        ax.set_xticks(np.arange(4), ['GPT-5.2', 'Open\nEvidence', 'Sonar-RP', 'Sonar'], fontsize=8)
        ax.set_ylim(0, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, loc='left', fontsize=10)
        ax.tick_params(length=0, labelcolor='dimgrey')
        ax.grid(False)

    fig, ax = plt.subplots(figsize=(7, 4))
    stacked(ax, totals, 90, 'Final consensus rating distribution')
    ax.set_ylabel('Responses per system (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=3, fontsize=9,
              frameon=True, facecolor='white', framealpha=.8, edgecolor='lightgrey', labelcolor='dimgrey')
    sns.despine(left=True, bottom=True)
    save_figure(fig, 'Figure1_rating_distribution')

    paired = pd.read_csv(HERE / 'results/paired_comparisons_original_framework.csv').dropna()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    yy = np.arange(6)[::-1]
    ax.errorbar(paired.mean_ordinal_difference, yy,
                xerr=np.array([paired.mean_ordinal_difference - paired.cluster_ci_low,
                               paired.cluster_ci_high - paired.mean_ordinal_difference]),
                fmt='o', capsize=4, color=colors[5])
    ax.set_yticks(yy, [f'{r.system_a} vs {r.system_b}' for r in paired.itertuples()], fontsize=9)
    ax.axvline(0, ls='--', lw=1, color='dimgrey')
    ax.set_xlabel('Descriptive mean difference on 0-2 rating coding')
    ax.set_title('Paired descriptive contrasts with topic-pair bootstrap intervals', loc='left', fontsize=11)
    ax.tick_params(length=0, labelcolor='dimgrey')
    ax.grid(False)
    sns.despine(left=True, bottom=True)
    save_figure(fig, 'Figure2_descriptive_paired_differences')

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for i, (ax, domain) in enumerate(zip(axes, DOMAINS)):
        ds = correctness[correctness.group == domain].set_index('system').loc[MODELS]
        stacked(ax, ds, 30, ['(a) Clinical assessment', '(b) Instrumental assessment', '(c) Therapy and rehabilitation'][i])
    axes[0].set_ylabel('Responses within domain (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.14), ncol=3,
               fontsize=9, frameon=True, facecolor='white', framealpha=.8, edgecolor='lightgrey', labelcolor='dimgrey')
    sns.despine(left=True, bottom=True)
    save_figure(fig, 'Figure3_domain_rating_distributions')

    formats = pd.read_csv(HERE / 'results/question_format.csv').dropna().set_index('system').loc[MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    ax = axes[0]
    for off, field, color, label in [(-.18, 'knowledge', colors[2], 'Knowledge'), (.18, 'case', colors[5], 'Case')]:
        values = formats[field + '_correct'] / 45
        ax.bar(np.arange(4) + off, values * 100, width=.33, color=color, label=label)
        ax.errorbar(np.arange(4) + off, values * 100,
                    yerr=np.array([values - formats[field + '_ci_low'], formats[field + '_ci_high'] - values]) * 100,
                    fmt='none', capsize=3, lw=1, color='dimgrey')
    ax.set_xticks(np.arange(4), ['GPT-5.2', 'Open\nEvidence', 'Sonar-RP', 'Sonar'], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Correct responses per format (%)')
    ax.set_title('(a) Guideline-concordant responses', loc='left', fontsize=11)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2, fontsize=9,
              frameon=True, facecolor='white', framealpha=.8, edgecolor='lightgrey', labelcolor='dimgrey')
    ax = axes[1]
    vals = -formats.case_minus_knowledge_mean_ordinal
    low, high = -formats.cluster_ci_high, -formats.cluster_ci_low
    yy = np.arange(4)[::-1]
    ax.errorbar(vals, yy, xerr=np.array([vals - low, high - vals]), fmt='o', capsize=4, color=colors[5])
    ax.set_yticks(yy, ['GPT-5.2', 'Open Evidence', 'Sonar-RP', 'Sonar'], fontsize=8)
    ax.axvline(0, ls='--', lw=1, color='dimgrey')
    ax.set_xlabel('Knowledge minus case mean rating (0-2 coding)')
    ax.set_title('(b) Paired descriptive contrasts', loc='left', fontsize=11)
    for ax in axes:
        ax.tick_params(length=0, labelcolor='dimgrey')
        ax.grid(False)
    fig.tight_layout()
    sns.despine(left=True, bottom=True)
    save_figure(fig, 'Figure5_question_format')
    legends = (
        '# Revised figure legends\n\n'
        'Figure 1. Final consensus rating distributions. Blue, yellow and red represent Correct, Inaccurate/Incomplete and Wrong, respectively. Each system contributes 90 responses.\n\n'
        'Figure 2. Descriptive paired differences in 0-2 ratings. Positive differences indicate a higher mean coded rating for the first system; the scale does not imply equally spaced clinical benefit. Error bars are percentile 95% intervals from 20,000 domain-stratified topic-pair bootstrap resamples, retaining both formats and all systems. No multiplicity-adjusted inference is attached to these descriptive intervals.\n\n'
        'Figure 3. Final consensus ratings by clinical domain. Each bar summarizes 30 responses per system per domain; colors match Figure 1. The pooled domain pattern is descriptive and does not identify a causal domain-difficulty mechanism.\n\n'
        'Figure 4. Source support among final consensus-Correct responses. The endpoint-specific Yes counts and model-specific Correct denominators are printed above bars. Panel A shows guideline support; panel B shows guideline OR high-level support. CH recorded the final source-support consensus with BL. Error bars are domain-stratified topic-pair bootstrap 95% intervals. There are no between-system significance annotations. Sonar-RP denotes Sonar-Reasoning-Pro.\n\n'
        'Figure 5. Knowledge versus case formats. Panel A shows Correct proportions among 45 responses per format per system, with domain-stratified topic-pair bootstrap 95% intervals. Panel B shows descriptive knowledge-minus-case differences on the 0-2 scale, with paired topic-pair bootstrap intervals. Lack of detected differences does not demonstrate equivalence or broad clinical reasoning.\n\n'
        'Additional figure. Guideline concordance and the joint Correct AND guideline-supported endpoint use a common denominator of 90 per system. Error bars are topic-pair bootstrap intervals. The joint endpoint is not a measurement of citation quality in noncorrect answers. In all figures, Sonar-RP denotes Sonar-Reasoning-Pro; CH denotes Cinja Huber and BL denotes Bendix Labeit.\n'
    )
    (HERE / 'figures/figure_legends.md').write_text(legends)


def run_descriptives(data):
    wide = data.pivot(index='item_id', columns='system', values='rating')[MODELS].sort_index()
    ratings = wide.to_numpy().reshape(45, 2, 4)
    correct = ratings == 2
    draws = bootstrap_indices()
    boot_correct = correct[draws].mean(axis=(1, 2))
    rows = []
    for j, m in enumerate(MODELS):
        for group, ids in [('Overall', np.arange(90))] + [(d, np.arange(i * 30, (i + 1) * 30)) for i, d in enumerate(DOMAINS)]:
            s = wide.loc[ids, m]
            n, k = len(s), int(s.eq(2).sum())
            blo, bhi = interval(boot_correct[:, j] if group == 'Overall' else correct[draws[:, DOMAINS.index(group) * 15:(DOMAINS.index(group) + 1) * 15], :, j].mean(axis=(1, 2)))
            wlo, whi = proportion_confint(k, n, method='wilson')
            rows.append({'system': m, 'group': group, 'n': n, 'correct': k, 'inaccurate': int(s.eq(1).sum()),
                         'wrong': int(s.eq(0).sum()), 'proportion': k / n, 'cluster_ci_low': blo,
                         'cluster_ci_high': bhi, 'wilson_ci_low': wlo, 'wilson_ci_high': whi})
    correctness = pd.DataFrame(rows)
    correctness.to_csv(HERE / 'results/correctness.csv', index=False)
    domains = []
    for i, d in enumerate(DOMAINS):
        prop = correct[i * 15:(i + 1) * 15].mean()
        lo, hi = interval(correct[draws[:, i * 15:(i + 1) * 15]].mean(axis=(1, 2, 3)))
        domains.append({'domain': d, 'n': 120, 'correct': int(prop * 120 + .1),
                        'proportion': prop, 'cluster_ci_low': lo, 'cluster_ci_high': hi})
    pd.DataFrame(domains).to_csv(HERE / 'results/pooled_domains.csv', index=False)

    pairwise = []
    for a, b in combinations(range(4), 2):
        delta = ratings[:, :, a] - ratings[:, :, b]
        test = stats.wilcoxon(wide.iloc[:, a], wide.iloc[:, b], zero_method='wilcox', method='approx', correction=False)
        lo, hi = interval(delta[draws].mean(axis=(1, 2)))
        clo, chi = interval(boot_correct[:, a] - boot_correct[:, b])
        pairwise.append({'system_a': MODELS[a], 'system_b': MODELS[b], 'mean_ordinal_difference': delta.mean(),
                         'cluster_ci_low': lo, 'cluster_ci_high': hi,
                         'correct_risk_difference': correct[:, :, a].mean() - correct[:, :, b].mean(),
                         'risk_difference_ci_low': clo, 'risk_difference_ci_high': chi,
                         'item_wins': int((delta > 0).sum()), 'item_losses': int((delta < 0).sum()),
                         'item_ties': int((delta == 0).sum()), 'wilcoxon_stat': test.statistic, 'p_unadjusted': test.pvalue})
    pairwise = pd.DataFrame(pairwise)
    pairwise['p_holm'] = multipletests(pairwise.p_unadjusted, method='holm')[1]
    pairwise.to_csv(HERE / 'results/paired_comparisons_original_framework.csv', index=False)

    tables = np.zeros((45, 3, 3), dtype=int)
    for r in data.itertuples():
        tables[r.statement_id, r.rater_CH, r.rater_BL] += 1
    tab = tables.sum(axis=0)
    bt = tables[draws].sum(axis=1)
    kappas = []
    for weight in ('unweighted', 'linear', 'quadratic'):
        lo, hi = interval(kappa(bt, weight))
        kappas.append({'weighting': weight, 'kappa': float(kappa(tab, weight)), 'ci_low': lo, 'ci_high': hi})
    pd.DataFrame(kappas).to_csv(HERE / 'results/rater_kappa.csv', index=False)
    pd.DataFrame(tab, index=['CH Wrong', 'CH Inaccurate', 'CH Correct'],
                 columns=['BL Wrong', 'BL Inaccurate', 'BL Correct']).to_csv(HERE / 'results/rater_cross_tabulation.csv')

    support_rows, joint_rows = [], []
    for endpoint, col in [('guideline', 'guideline_support'), ('high_evidence', 'high_evidence_inclusive')]:
        sw = data.pivot(index='item_id', columns='system', values=col)[MODELS].sort_index().to_numpy().reshape(45, 2, 4)
        # Do NOT interpret unassessed source support on noncorrect answers as "no".
        joint = correct & (sw == 1)
        bnum = joint[draws].sum(axis=(1, 2))
        bden = correct[draws].sum(axis=(1, 2))
        bconditional = np.divide(bnum, bden, out=np.full_like(bnum, np.nan, dtype=float), where=bden > 0)
        for j, m in enumerate(MODELS):
            k, n = int(joint[:, :, j].sum()), int(correct[:, :, j].sum())
            cl, ch = interval(bconditional[:, j])
            jl, jh = interval(bnum[:, j] / 90)
            support_rows.append({'system': m, 'endpoint': endpoint, 'n_supported': k, 'n_correct': n,
                                 'conditional_proportion': k / n, 'conditional_ci_low': cl, 'conditional_ci_high': ch,
                                 'joint_proportion': k / 90, 'joint_ci_low': jl, 'joint_ci_high': jh})
        for a, b in combinations(range(4), 2):
            jl, jh = interval((bnum[:, a] - bnum[:, b]) / 90)
            joint_rows.append({'endpoint': endpoint, 'system_a': MODELS[a], 'system_b': MODELS[b],
                               'risk_difference': joint[:, :, a].mean() - joint[:, :, b].mean(),
                               'cluster_ci_low': jl, 'cluster_ci_high': jh})
    pd.DataFrame(support_rows).to_csv(HERE / 'results/source_support.csv', index=False)
    pd.DataFrame(joint_rows).to_csv(HERE / 'results/joint_source_support_risk_differences.csv', index=False)

    items = data[data.system == MODELS[0]][['item_id', 'statement_id', 'domain', 'question_no', 'question_type',
                                                'guideline_id', 'question', 'reference_answer']].set_index('item_id').join(wide)
    items['all_correct'] = wide.eq(2).all(axis=1)
    items['all_wrong'] = wide.eq(0).all(axis=1)
    items['none_correct'] = wide.ne(2).all(axis=1)
    items['mixed_rating_categories'] = wide.nunique(axis=1) > 1
    items.to_csv(HERE / 'results/item_level_results.csv')
    guideline_contribution = items.groupby(['guideline_id', 'domain']).agg(n_items=('statement_id', 'size'), n_topic_pairs=('statement_id', 'nunique')).reset_index()
    guideline_contribution.to_csv(HERE / 'results/guideline_contribution.csv', index=False)
    guideline_perf = data.groupby(['guideline_id', 'system']).rating.agg(n='size', mean_ordinal='mean', correct=lambda x: x.eq(2).sum()).reset_index()
    guideline_perf['correct_proportion'] = guideline_perf.correct / guideline_perf.n
    guideline_perf.to_csv(HERE / 'results/performance_by_guideline.csv', index=False)
    robustness = []
    for unit, excluded in [('statement_id', s) for s in range(45)] + [('guideline_id', g) for g in sorted(data.guideline_id.unique())]:
        keep_ids = data.loc[data[unit] != excluded, 'item_id'].unique()
        sub = wide.loc[keep_ids]
        for other in MODELS[1:]:
            robustness.append({'leave_out_unit': unit, 'excluded_id': excluded, 'comparator': other, 'remaining_items': len(sub),
                               'GPT_mean_ordinal_difference': float((sub[MODELS[0]] - sub[other]).mean()),
                               'GPT_correct_risk_difference': float((sub[MODELS[0]].eq(2) - sub[other].eq(2).astype(int)).mean())})
    pd.DataFrame(robustness).to_csv(HERE / 'results/leave_one_cluster_or_guideline_out.csv', index=False)
    formats = []
    for j, m in enumerate(MODELS):
        test = stats.wilcoxon(ratings[:, 0, j], ratings[:, 1, j], zero_method='wilcox', method='approx')
        lo, hi = interval((ratings[:, 1, j] - ratings[:, 0, j])[draws].mean(axis=1))
        formats.append({'system': m, 'knowledge_correct': int(correct[:, 0, j].sum()), 'case_correct': int(correct[:, 1, j].sum()),
                        'knowledge_ci_low': interval(correct[draws, 0, j].mean(axis=1))[0],
                        'knowledge_ci_high': interval(correct[draws, 0, j].mean(axis=1))[1],
                        'case_ci_low': interval(correct[draws, 1, j].mean(axis=1))[0],
                        'case_ci_high': interval(correct[draws, 1, j].mean(axis=1))[1],
                        'n_per_format': 45, 'case_minus_knowledge_mean_ordinal': float((ratings[:, 1, j] - ratings[:, 0, j]).mean()),
                        'cluster_ci_low': lo, 'cluster_ci_high': hi, 'p_unadjusted': test.pvalue})
    fm = pd.DataFrame(formats)
    fm['p_holm'] = multipletests(fm.p_unadjusted, method='holm')[1]
    fm.to_csv(HERE / 'results/question_format.csv', index=False)
    friedman = stats.friedmanchisquare(*(wide[m] for m in MODELS))
    summary = {'seed': SEED, 'bootstrap_replicates': REPLICATES,
               'CI_method': 'Percentile, stratified topic-pair cluster bootstrap; retain 2 items x 4 systems per pair, 15 pairs per domain',
               'inference_scope': 'Purposive selected-content benchmark. These intervals do not establish population representativeness or model-generation variability.',
               'friedman_chi2': float(friedman.statistic), 'friedman_p': float(friedman.pvalue),
               'agreement_n': int(np.trace(tab)), 'agreement_total': int(tab.sum()),
               'all_correct_items': int(items.all_correct.sum()), 'all_wrong_items': int(items.all_wrong.sum()),
               'none_correct_items': int(items.none_correct.sum()), 'mixed_category_items': int(items.mixed_rating_categories.sum()),
               'identical_category_items': int((~items.mixed_rating_categories).sum()),
               'noncorrect_total': int((data.rating != 2).sum()), 'guideline_ids_contributing': sorted(map(int, data.guideline_id.unique())),
               'environment': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'statsmodels', 'openpyxl', 'matplotlib', 'seaborn', 'lxml', 'python-docx']},
               'python': platform.python_version()}
    (HERE / 'results/summary.json').write_text(json.dumps(summary, indent=2))
    (HERE / 'requirements-lock.txt').write_text('\n'.join(f'{p}=={v}' for p, v in summary['environment'].items()) + '\n')
    return summary


def ordinal_cluster_sensitivity(data):
    from ordinal_mixed import contrast_tables, design_matrix, omnibus_system_test
    separated = [int(g) for g, sub in data.groupby('guideline_id') if sub.rating.eq(2).all()]
    assert separated == [5], 'Reassess separation handling if the source data change.'
    sub = data[~data.guideline_id.isin(separated)]
    diagnostics = []
    for frame, adjust_guideline, stem in [
        (sub, True, 'ordinal_cluster_robust'),
        (data, False, 'ordinal_cluster_robust_without_guideline'),
    ]:
        X, y, names = design_matrix(frame, include_guideline=adjust_guideline)
        groups = np.repeat(np.arange(frame.statement_id.nunique()), 8)
        model = OrderedModel(y.ravel(), X.reshape(len(frame), -1), distr='logit')
        fit = model.fit(method='bfgs', maxiter=1500, gtol=1e-8, disp=False,
                        cov_type='cluster', cov_kwds={'groups': groups, 'use_correction': True})
        information = -model.hessian(fit.params)
        information = (information + information.T) / 2
        eigenvalues = np.linalg.eigvalsh(information)
        score_max = float(max(abs(model.score(fit.params))))
        gate = bool(fit.mle_retvals['converged'] and score_max < 1e-3
                    and eigenvalues.min() > 1e-6 and np.linalg.cond(information) < 1e6)
        diagnostics.append({'analysis': stem, 'n_responses': len(frame),
                            'n_topic_pairs': frame.statement_id.nunique(),
                            'guideline_adjusted': adjust_guideline,
                            'excluded_separated_guideline_ids': separated if adjust_guideline else [],
                            'optimizer_converged': bool(fit.mle_retvals['converged']),
                            'score_max': score_max, 'information_eigenvalues': eigenvalues.tolist(),
                            'information_condition': float(np.linalg.cond(information)),
                            'release_gate_passed': gate})
        (HERE / 'results/ordinal_cluster_robust_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
        if not gate:
            raise RuntimeError(f'{stem} failed finite-fit diagnostics; do not release inference.')
        p = len(names)
        order = np.r_[p, p + 1, np.arange(p)]
        params = np.asarray(fit.params)[order]
        covariance = np.asarray(fit.cov_params())[np.ix_(order, order)]
        contrast_tables(params, covariance, names).to_csv(HERE / 'results' / f'{stem}_sensitivity.csv', index=False)
        (HERE / 'results' / f'{stem}_omnibus.json').write_text(json.dumps(omnibus_system_test(params, covariance, names), indent=2))
    # Threshold-specific system-only GEE checks avoid sparse guideline separation.
    # They are exploratory diagnostics, not a formal mixed-model proportional-odds test.
    import statsmodels.api as sm
    checks = []
    X, y, _ = design_matrix(data)
    groups = np.repeat(np.arange(data.statement_id.nunique()), 8)
    Xsys = sm.add_constant(X.reshape(360, -1)[:, :3])
    for endpoint, yy in [('Correct', (y.ravel() == 2).astype(int)), ('At least Inaccurate', (y.ravel() >= 1).astype(int))]:
        bfit = sm.GLM(yy, Xsys, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': groups})
        for j, m in enumerate(MODELS[1:], 1):
            ci = np.asarray(bfit.conf_int())[j]
            checks.append({'threshold': endpoint, 'system_vs_GPT': m, 'OR': np.exp(bfit.params[j]),
                           'ci_low': np.exp(ci[0]), 'ci_high': np.exp(ci[1]), 'log_OR': bfit.params[j]})
    pd.DataFrame(checks).to_csv(HERE / 'results/proportional_odds_exploratory_check.csv', index=False)
    # A clinically simpler sensitivity targets Correct, avoiding proportional-odds assumptions.
    # The all-Correct guideline-5 stratum again has no finite adjusted logit effect.
    Xb, yb, nb = design_matrix(sub)
    binary = sm.GEE((yb.ravel() == 2).astype(int), sm.add_constant(Xb.reshape(len(sub), -1)),
                    groups=np.repeat(np.arange(len(sub) // 8), 8), family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Independence()).fit(cov_type='bias_reduced')
    assert binary.converged
    contrasts = contrast_tables(np.asarray(binary.params), np.asarray(binary.cov_params()), nb, beta_offset=1)
    contrasts = contrasts.rename(columns={'OR_higher_category': 'OR_Correct'})
    contrasts.to_csv(HERE / 'results/binary_correct_GEE_sensitivity.csv', index=False)
    (HERE / 'results/binary_correct_GEE_omnibus.json').write_text(json.dumps(omnibus_system_test(binary.params, binary.cov_params(), nb, beta_offset=1), indent=2))


def final_report_tables(data):
    """Regenerate secondary tables too, including during portable CLI reproduction."""
    catalog = pd.read_csv(HERE / 'data/guideline_titles.csv')
    count = pd.read_csv(HERE / 'results/guideline_contribution.csv').set_index(['guideline_id', 'domain'])
    grid = pd.MultiIndex.from_product([catalog.guideline_id.astype(int), DOMAINS], names=['guideline_id', 'domain'])
    count.reindex(grid, fill_value=0).reset_index().merge(catalog, on='guideline_id').to_csv(
        HERE / 'results/guideline_contribution_complete_catalogue.csv', index=False)
    omnibus = []
    for name in ['ordinal_mixed_omnibus.json', 'ordinal_cluster_robust_omnibus.json',
                 'ordinal_cluster_robust_without_guideline_omnibus.json', 'binary_correct_GEE_omnibus.json']:
        path = HERE / 'results' / name
        if path.exists():
            row = json.loads(path.read_text())
            row['analysis'] = name.replace('_omnibus.json', '')
            omnibus.append(row)
    pd.DataFrame(omnibus).to_csv(HERE / 'results/overall_system_tests.csv', index=False)
    coverage = []
    for system, part in data.groupby('system'):
        noncorrect = part[part.rating != 2]
        coverage.append({'system': system, 'all_responses': len(part),
                         'safety_explicit_yes': int(part.safety_risk.eq(1).sum()),
                         'safety_explicit_no': int(part.safety_risk.eq(0).sum()),
                         'safety_blank': int(part.safety_risk.isna().sum()),
                         'noncorrect_responses': len(noncorrect),
                         'noncorrect_safety_recorded': int(noncorrect.safety_risk.notna().sum()),
                         'noncorrect_safety_missing': int(noncorrect.safety_risk.isna().sum())})
    pd.DataFrame(coverage).to_csv(HERE / 'results/safety_annotation_coverage.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed', action='store_true', help='Also fit the validated nested cumulative-link mixed model.')
    parser.add_argument('--from-clean-data', action='store_true', help='Use bundled audited CSV; portable without original collection files or revision DOCX.')
    args = parser.parse_args()
    data = pd.read_csv(HERE / 'data/response_level.csv') if args.from_clean_data else build_data()
    summary = run_descriptives(data)
    ordinal_cluster_sensitivity(data)
    figures()
    if args.mixed:
        from ordinal_mixed import run_mixed
        run_mixed(data)
    final_report_tables(data)
    print(json.dumps(summary, indent=2))
    print(pd.read_csv(HERE / 'results/source_support.csv').to_string(index=False))


if __name__ == '__main__':
    main()
