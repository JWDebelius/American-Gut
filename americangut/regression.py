import numpy as np
import pandas as pd
import scipy
import statsmodels.formula.api as smf


def olf_build_model(response, predictors, data, dname=None, model_watch=None,
                    prev_id=None):
    """Builds up a series of Ordinary Least Squares Models

    """

    # Looks for the last model being referenced as the best
    id_, prev_id, prev_eq = _check_watch(model_watch, prev_id)

    if model_watch is None:
        model_watch = pd.DataFrame(data=np.zeros((0, 11)),
                                   columns=['data_set', 'equation', 'var', 'n',
                                            'k', 'aic', 'aicc', 'D_score',
                                            'pearson_r2', 'adj_r2', 'cond no'])
    # Builds up the model
    for var_ in predictors:
        # Builds up the list of equations to be added
        eqs = _equation_builder(var_, prev_eq, response)
        num_var = len(eqs)

        # Fits the equations
        fits = [smf.ols(eq, data=data).fit() for eq in eqs]

        # Summarizes the fit
        fit_check = _populate_fit_check(fits, var_, id_, dname)

        # Updates the fit check with the best previous model
        check_ids = fit_check.index.values
        if prev_id is None:
            prev_id = min(fit_check.index.values)
        else:
            fit_check.loc[prev_id] = model_watch.loc[prev_id]

        # Identifies the best model
        prev_id = identify_best_model(fit_check, prev_id)

        model_watch = pd.concat((model_watch, fit_check.loc[check_ids]))

        # Gets the best fit equation
        prev_eq = model_watch.loc[prev_id, 'equation']

        # Advances the counter
        id_ = id_ + num_var

    return model_watch, prev_eq


def control_cat_order(df, category, old_order=None, new_order=None,
                      counts=None, drop=None):
    """..."""
    # Sets up default variables
    if old_order is None:
        old_order = sorted(df.groupby(category).groups.keys())
    if new_order is None:
        new_order = old_order
    if counts is None:
        counts = np.arange(0, len(new_order))
    if isinstance(drop, str):
        drop = [drop]
    elif drop is None:
        drop = []

    # Sets the dropped categories to nans
    for cat in drop:
        if cat in old_order:
            raise ValueError('%s cannot be dropped and categorized' % cat)
        df.loc[df[category] == cat, category] = np.nan

    # Orders the new category
    for (old_, new_, count) in zip(*[old_order, new_order, counts]):
        df.loc[df[category] == old_, category] = '(%i)%s' % (count, old_)


def _check_watch(model_watch, prev_id):
    """..."""
    if model_watch is None:
        model_watch = {}
        id_ = 1
        prev_id = None
        prev_eq = None
    else:
        id_ = max(model_watch.keys()) + 1
        if prev_id is None:
            models = pd.DataFrame(model_watch).transpose()
            ranked = models.sort(['aicc'], inplace=False).index
            prev_id = ranked[0]
        prev_eq = model_watch[prev_id]['equation']
    return model_watch, id_, prev_id, prev_eq


def _equation_builder(vars_, last_eq=None, response=None):
    """Builds a set of equations trying multiple predictor options"""
    # Checks enough variables are defined
    if last_eq is None and response is None:
        raise ValueError('A response or last equation must be specified.')

    # Checks the class of vars
    if isinstance(vars_, str):
        vars_ = [vars_]

    # Builds the equation
    if last_eq is None:
        return ['%s ~ %s' % (response, pred) for pred in vars_]
    else:
        eqs = []
        for pred in vars_:
            if '&' in pred:
                eqs.append(last_eq.replace(' + %s' % pred[1:], ''))
            else:
                eqs.append('%s + %s' % (last_eq, pred))

    return eqs


def _populate_fit_check(fits, var_, id_=1, dname=None):
    """Updates the fit_check information"""
    fit_check = []
    if isinstance(var_, str):
        var_ = [var_]

    for idy, fit_ in enumerate(fits):
        n = fit_.nobs
        k = fit_.df_model
        aicc = fit_.aic + (2 * k * (k + 1) / (n - k - 1))
        d_score = scipy.stats.kstest(fit_.resid.values, 'norm')[0]

        fit_check.append(pd.Series({'data_set': dname,
                                    'equation':  fit_.model.formula,
                                    'var': var_[idy],
                                    'n': n,
                                    'k': k,
                                    'aic': fit_.aic,
                                    'aicc': aicc,
                                    'D_score': d_score,
                                    'pearson_r2': fit_.rsquared,
                                    'adj_r2': fit_.rsquared_adj,
                                    'cond no': fit_.condition_number},
                                   name=id_ + idy))
    return pd.DataFrame(fit_check)


def _identify_best_model(fit_check, prev_id):
    """..."""
    fit_check['ref'] = prev_id
    prev_r2 = fit_check.loc[prev_id, 'adj_r2']
    prev_aicc = fit_check.loc[prev_id, 'aicc']
    fit_check['score'] = -10000*(((fit_check.adj_r2 - prev_r2) *
                                 (fit_check.aicc - prev_aicc)) /
                                 fit_check['cond no'])
    prev_id = fit_check.loc[fit_check.score == fit_check.score.max()].index[0]
    return prev_id
