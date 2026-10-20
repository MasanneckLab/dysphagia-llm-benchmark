"""Nested logistic cumulative-link model, fitted by marginal maximum likelihood.

The two nested Gaussian intercepts permit factorized Gauss-Hermite integration:
integrate each item's intercept, multiply the two item likelihoods, then integrate
the topic-pair intercept. This is non-adaptive quadrature, NOT R's Laplace CLMM.
Positive location effects increase the odds of a higher rating category.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit, log_expit, logsumexp
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests

from prepare_data import DOMAINS, HERE, MODELS


def design_matrix(data, include_guideline=True):
    d = data.sort_values(['statement_id', 'format_index', 'system'], key=lambda s: s.map({m: i for i, m in enumerate(MODELS)}) if s.name == 'system' else s)
    columns = {f'system:{m}': (d.system == m).to_numpy(float) for m in MODELS[1:]}
    columns.update({f'domain:{c}': (d.domain == c).to_numpy(float) for c in DOMAINS[1:]})
    columns['format:Case'] = (d.question_type == 'Case').to_numpy(float)
    if include_guideline:
        columns.update({f'guideline:{c}': (d.guideline_id == c).to_numpy(float) for c in sorted(d.guideline_id.unique())[1:]})
    X = np.column_stack(list(columns.values()))
    assert np.linalg.matrix_rank(X) == X.shape[1]
    assert d.groupby('statement_id').size().eq(8).all()
    n_pairs = d.statement_id.nunique()
    return X.reshape(n_pairs, 2, 4, -1), d.rating.to_numpy().reshape(n_pairs, 2, 4), list(columns)


class NestedOrdinal:
    def __init__(self, X, y, nodes=21, item_effect=True):
        self.X, self.y = np.asarray(X), np.asarray(y)
        self.p = X.shape[-1]
        z, w = hermgauss(nodes)
        self.z = z * np.sqrt(2)
        self.logw = np.log(w / np.sqrt(np.pi))
        self.nodes = nodes
        self.item_effect = item_effect
        self.z_inner = self.z if item_effect else np.array([0.0])
        self.logw_inner = self.logw if item_effect else np.array([0.0])

    def objective(self, params):
        theta, loggap = params[:2]
        gap = np.exp(loggap)
        beta = params[2:2 + self.p]
        sd_topic = np.exp(params[-2] if self.item_effect else params[-1])
        sd_item = np.exp(params[-1]) if self.item_effect else 0.0
        eta = np.einsum('ijkp,p->ijk', self.X, beta)[..., None, None]
        eta = eta + sd_topic * self.z[None, None, None, :, None] + sd_item * self.z_inner[None, None, None, None, :]
        a, b = theta - eta, theta + gap - eta
        sa, sb = expit(a), expit(b)
        y = self.y[..., None, None]
        lp0, lp2 = log_expit(a), log_expit(-b)
        lp1 = log_expit(b) + log_expit(-a) + np.log(-np.expm1(-gap))
        lp = np.where(y == 0, lp0, np.where(y == 1, lp1, lp2))
        # Posterior quadrature weights factor into outer topic and inner item weights.
        item_log = lp.sum(axis=2) + self.logw_inner[None, None, None, :]
        item_integral = logsumexp(item_log, axis=-1)
        topic_log = item_integral.sum(axis=1) + self.logw[None, :]
        topic_integral = logsumexp(topic_log, axis=-1)
        outer = np.exp(topic_log - topic_integral[:, None])
        inner = np.exp(item_log - item_integral[..., None])
        posterior = outer[:, None, None, :, None] * inner[:, :, None, :, :]
        deta = np.where(y == 0, sa - 1, np.where(y == 1, sa + sb - 1, sb))
        dtheta = -deta
        dloggap = np.where(y == 0, 0, np.where(y == 1, 1 - sb + 1 / np.expm1(gap), -sb)) * gap
        weighted = posterior * deta
        grad_beta = np.einsum('ijk,ijkp->p', weighted.sum(axis=(-1, -2)), self.X)
        tail = [np.sum(weighted * sd_topic * self.z[None, None, None, :, None])]
        if self.item_effect:
            tail.append(np.sum(weighted * sd_item * self.z_inner[None, None, None, None, :]))
        grad = np.r_[np.sum(posterior * dtheta), np.sum(posterior * dloggap), grad_beta, tail]
        return -float(topic_integral.sum()), -grad

    def fit(self, start=None):
        if start is None:
            start = np.r_[-2.5, 1.2, np.zeros(self.p), [0.0, 0.0] if self.item_effect else [0.0]]
        bounds = [(-15, 15), (-4, 3)] + [(-15, 15)] * self.p + [(-8, 3)] * (2 if self.item_effect else 1)
        return minimize(self.objective, start, method='L-BFGS-B', jac=True, bounds=bounds,
                        options={'maxiter': 1200, 'ftol': 1e-13, 'gtol': 1e-7, 'maxls': 40})


def contrast_tables(params, cov, names, beta_offset=2):
    indices = {m: (None if i == 0 else beta_offset + names.index(f'system:{m}')) for i, m in enumerate(MODELS)}
    rows = []
    for a, b in combinations(MODELS, 2):
        c = np.zeros(len(params))
        if indices[a] is not None:
            c[indices[a]] += 1
        if indices[b] is not None:
            c[indices[b]] -= 1
        effect, se = float(c @ params), np.sqrt(c @ cov @ c)
        rows.append({'system_a': a, 'system_b': b, 'log_OR': effect, 'SE': se,
                     'OR_higher_category': np.exp(effect), 'ci_low': np.exp(effect - 1.96 * se),
                     'ci_high': np.exp(effect + 1.96 * se), 'p_unadjusted': 2 * norm.sf(abs(effect / se))})
    result = pd.DataFrame(rows)
    result['p_holm'] = multipletests(result.p_unadjusted, method='holm')[1]
    return result


def omnibus_system_test(params, cov, names, beta_offset=2):
    ix = [beta_offset + names.index(f'system:{m}') for m in MODELS[1:]]
    beta = np.asarray(params)[ix]
    covariance = np.asarray(cov)[np.ix_(ix, ix)]
    statistic = float(beta @ np.linalg.solve(covariance, beta))
    return {'Wald_chi2': statistic, 'df': 3, 'p_value': float(chi2.sf(statistic, 3))}


def run_mixed(data):
    X, y, names = design_matrix(data)
    model = NestedOrdinal(X, y, 21)
    starts = [np.r_[-2.5, 1.2, np.zeros(model.p), s] for s in ([0, 0], [0.7, -0.5], [-0.5, 0.7])]
    fits = []
    for start in starts:
        f = model.fit(start)
        print(f'21-node start: nll={f.fun:.8f}; gradient={max(abs(f.jac)):.3g}; success={f.success}', flush=True)
        fits.append(f)
    fit = min(fits, key=lambda r: r.fun)
    diagnostics = {'formula': 'rating ~ system + domain + guideline_id + question_type + (1|statement_id) + (1|item_id)',
                   'estimator': 'Marginal MLE, factorized non-adaptive Gauss-Hermite quadrature, analytic gradient',
                   'n_responses': 360, 'n_items': 90, 'n_topic_pairs': 45,
                   'starts': [{'nll': float(f.fun), 'success': bool(f.success), 'gradient_max': float(max(abs(f.jac)))} for f in fits],
                   'quadrature': []}
    previous = fit.x.copy()
    for n in (31, 41, 61):
        model = NestedOrdinal(X, y, n)
        f = model.fit(previous)
        shift = float(max(abs(f.x[:2 + model.p] - previous[:2 + model.p])))
        diagnostics['quadrature'].append({'nodes': n, 'nll': float(f.fun), 'success': bool(f.success),
                                          'gradient_max': float(max(abs(f.jac))), 'max_fixed_parameter_shift': shift})
        print(f'{n}-node fit: nll={f.fun:.8f}; shift={shift:.3g}; gradient={max(abs(f.jac)):.3g}', flush=True)
        previous, fit = f.x.copy(), f
    hessian = approx_derivative(lambda p: model.objective(p)[1], fit.x, method='3-point', rel_step=2e-5)
    hessian = (hessian + hessian.T) / 2
    eigenvalues = np.linalg.eigvalsh(hessian)
    cov = np.linalg.inv(hessian)
    diagnostics.update({'final_nodes': model.nodes, 'nll': float(fit.fun), 'converged': bool(fit.success),
                        'gradient_max': float(max(abs(fit.jac))), 'hessian_eigenvalues': eigenvalues.tolist(),
                        'hessian_condition': float(np.linalg.cond(hessian)),
                        'sd_topic_pair': float(np.exp(fit.x[-2])), 'sd_item': float(np.exp(fit.x[-1])),
                        'thresholds': [float(fit.x[0]), float(fit.x[0] + np.exp(fit.x[1]))],
                        'boundary': bool(any(abs(v) > 7.8 for v in fit.x[-2:]))})
    # A failed diagnostic is a stopping condition, not permission to report convenient p-values.
    diagnostics['release_gate_passed'] = bool(fit.success and max(abs(fit.jac)) < 1e-3 and eigenvalues.min() > 1e-6
                                               and diagnostics['quadrature'][-1]['max_fixed_parameter_shift'] < .02
                                               and not diagnostics['boundary'])
    diagnostics['coefficient_at_bound'] = [names[j] for j, v in enumerate(fit.x[2:2 + model.p]) if abs(v) > 14.9]
    diagnostics['release_gate_passed'] &= not bool(diagnostics['coefficient_at_bound'])
    (HERE / 'results/ordinal_mixed_full_attempt.json').write_text(json.dumps(diagnostics, indent=2))
    if not diagnostics['release_gate_passed']:
        print('Full model failed release gate: boundary item variance / separated guideline. No full-model inferential output released.', flush=True)
        return run_reduced_mixed(data, diagnostics)
    np.savez(HERE / 'results/ordinal_mixed_fit.npz', params=fit.x, covariance=cov, hessian=hessian, names=np.array(names))
    contrasts = contrast_tables(fit.x, cov, names)
    contrasts.to_csv(HERE / 'results/ordinal_mixed_contrasts.csv', index=False)
    (HERE / 'results/ordinal_mixed_omnibus.json').write_text(json.dumps(omnibus_system_test(fit.x, cov, names), indent=2))
    rows = []
    for j, name in enumerate(names):
        effect, se = fit.x[j + 2], np.sqrt(cov[j + 2, j + 2])
        rows.append({'term': name, 'log_OR': effect, 'SE': se, 'OR_higher_category': np.exp(effect),
                     'ci_low': np.exp(effect - 1.96 * se), 'ci_high': np.exp(effect + 1.96 * se),
                     'p_unadjusted': 2 * norm.sf(abs(effect / se))})
    pd.DataFrame(rows).to_csv(HERE / 'results/ordinal_mixed_fixed_effects.csv', index=False)
    # Guideline-adjustment sensitivity, because two guidelines contribute only 1 and 3 topic pairs.
    X2, y2, names2 = design_matrix(data, include_guideline=False)
    reduced = NestedOrdinal(X2, y2, 61)
    redfit = reduced.fit(np.r_[fit.x[:2], [fit.x[2 + names.index(t)] for t in names2], fit.x[-2:]])
    H2 = approx_derivative(lambda p: reduced.objective(p)[1], redfit.x, method='3-point', rel_step=2e-5)
    H2 = (H2 + H2.T) / 2
    redgate = bool(redfit.success and max(abs(redfit.jac)) < 1e-3 and np.linalg.eigvalsh(H2).min() > 1e-6)
    diagnostics['without_guideline'] = {'success': bool(redfit.success), 'gradient_max': float(max(abs(redfit.jac))),
                                        'gate_passed': redgate, 'nll': float(redfit.fun)}
    if redgate:
        contrast_tables(redfit.x, np.linalg.inv(H2), names2).to_csv(HERE / 'results/ordinal_mixed_without_guideline.csv', index=False)
    (HERE / 'results/ordinal_mixed_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    print(contrasts.to_string(index=False), flush=True)
    return diagnostics


def run_reduced_mixed(data, full_attempt=None):
    # Guideline 5 is an entirely Correct stratum: its profile MLE is +infinity.
    # At that limit its likelihood is 1, so it carries no system-contrast information.
    # Remove this stratum only for guideline-adjusted finite-MLE fits.
    separated = [int(g) for g, sub in data.groupby('guideline_id') if sub.rating.eq(2).all()]
    assert separated == [5], 'Reassess the separation handling if the source data change.'
    sub = data[~data.guideline_id.isin(separated)]
    X, y, names = design_matrix(sub)
    diagnostics = {
        'formula': 'rating ~ system + domain + guideline_id + question_type + (1|statement_id)',
        'n_responses': len(sub), 'n_items': sub.item_id.nunique(), 'n_topic_pairs': sub.statement_id.nunique(),
        'excluded_separated_guideline_ids': separated,
        'reason': 'Full-model item variance near zero at the numerical bound; guideline 5 contains eight Correct responses and has no finite fixed-effect MLE. Full attempt preserved separately.',
        'quadrature': [], 'starts': [], 'estimator': 'Marginal MLE; non-adaptive Gauss-Hermite integration; analytic gradient',
    }
    first = NestedOrdinal(X, y, 31, item_effect=False)
    fits = [first.fit(np.r_[-2.5, 1.2, np.zeros(first.p), sd]) for sd in [0.0, .7, -.5]]
    for f in fits:
        diagnostics['starts'].append({'nll': float(f.fun), 'success': bool(f.success), 'gradient_max': float(max(abs(f.jac)))})
    fit = min(fits, key=lambda f: f.fun)
    previous = fit.x.copy()
    for n in (61, 81, 121, 161):
        model = NestedOrdinal(X, y, n, item_effect=False)
        f = model.fit(previous)
        shift = float(max(abs(f.x - previous)))
        diagnostics['quadrature'].append({'nodes': n, 'nll': float(f.fun), 'success': bool(f.success),
                                          'gradient_max': float(max(abs(f.jac))), 'max_parameter_shift': shift})
        print(f'Reduced {n}-node fit: nll={f.fun:.8f}; shift={shift:.3g}; gradient={max(abs(f.jac)):.3g}', flush=True)
        previous, fit = f.x.copy(), f
        if n >= 81 and shift < 1e-4:
            break
    hessian = approx_derivative(lambda p: model.objective(p)[1], fit.x, method='3-point', rel_step=2e-5)
    hessian = (hessian + hessian.T) / 2
    eigenvalues = np.linalg.eigvalsh(hessian)
    diagnostics.update({'final_nodes': model.nodes, 'nll': float(fit.fun), 'converged': bool(fit.success),
                        'gradient_max': float(max(abs(fit.jac))), 'hessian_eigenvalues': eigenvalues.tolist(),
                        'hessian_condition': float(np.linalg.cond(hessian)), 'sd_topic_pair': float(np.exp(fit.x[-1])),
                        'sd_item': 0, 'thresholds': [float(fit.x[0]), float(fit.x[0] + np.exp(fit.x[1]))]})
    diagnostics['release_gate_passed'] = bool(fit.success and max(abs(fit.jac)) < 1e-3 and eigenvalues.min() > 1e-6
                                               and diagnostics['hessian_condition'] < 1e6
                                               and diagnostics['quadrature'][-1]['max_parameter_shift'] < .001)
    (HERE / 'results/ordinal_mixed_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    if not diagnostics['release_gate_passed']:
        raise RuntimeError('Reduced mixed-model release gate failed. Do not report this model as completed.')
    cov = np.linalg.inv(hessian)
    np.savez(HERE / 'results/ordinal_mixed_fit.npz', params=fit.x, covariance=cov, hessian=hessian, names=np.array(names))
    contrasts = contrast_tables(fit.x, cov, names)
    contrasts.to_csv(HERE / 'results/ordinal_mixed_contrasts.csv', index=False)
    (HERE / 'results/ordinal_mixed_omnibus.json').write_text(json.dumps(omnibus_system_test(fit.x, cov, names), indent=2))
    rows = []
    for j, name in enumerate(names):
        effect, se = fit.x[j + 2], np.sqrt(cov[j + 2, j + 2])
        rows.append({'term': name, 'log_OR': effect, 'SE': se, 'OR_higher_category': np.exp(effect),
                     'ci_low': np.exp(effect - 1.96 * se), 'ci_high': np.exp(effect + 1.96 * se),
                     'p_unadjusted': 2 * norm.sf(abs(effect / se))})
    pd.DataFrame(rows).to_csv(HERE / 'results/ordinal_mixed_fixed_effects.csv', index=False)
    # Removing all guideline adjustments restores the separated pair and tests dependence on adjustment.
    X2, y2, names2 = design_matrix(data, include_guideline=False)
    redmodel = NestedOrdinal(X2, y2, model.nodes, item_effect=False)
    redfit = redmodel.fit(np.r_[fit.x[:2], [fit.x[2 + names.index(t)] for t in names2], fit.x[-1]])
    H2 = approx_derivative(lambda p: redmodel.objective(p)[1], redfit.x, method='3-point', rel_step=2e-5)
    H2 = (H2 + H2.T) / 2
    redgate = bool(redfit.success and max(abs(redfit.jac)) < 1e-3 and np.linalg.eigvalsh(H2).min() > 1e-6)
    diagnostics['without_guideline'] = {'gate_passed': redgate, 'n_responses': 360,
                                        'nll': float(redfit.fun), 'gradient_max': float(max(abs(redfit.jac)))}
    if redgate:
        contrast_tables(redfit.x, np.linalg.inv(H2), names2).to_csv(HERE / 'results/ordinal_mixed_without_guideline.csv', index=False)
    (HERE / 'results/ordinal_mixed_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    print(contrasts.to_string(index=False), flush=True)
    return diagnostics


if __name__ == '__main__':
    run_mixed(pd.read_csv(HERE / 'data/response_level.csv'))
