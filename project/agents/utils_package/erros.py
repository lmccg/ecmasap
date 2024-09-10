import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score, \
    accuracy_score

global EPSILON
EPSILON = 1e-10


async def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

async def _absolute_error(actual: np.ndarray, predicted: np.ndarray):
    """ Absolute error """
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)
    possible_bool = ['true', 'false']
    if str(predicted[0]).lower() in possible_bool:
        await accuracy_score_(actual=actual, predicted=predicted)
    else:
        return np.abs(np.mean(await _error(actual, predicted)))
    # return np.abs(actual - predicted)

async def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is multiplied by 100
    """
    return (np.abs(actual - predicted) / actual)*100


async def _percentage_error_2(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    predicted_aux = []
    for i in range(0, len(predicted)):
        if predicted[i] < 0:
            predicted_aux.append(0)
        else:
            predicted_aux.append(predicted[i])

    errors = []
    for i in range(0, len(actual)):
        e = 0

        a = actual[i]
        p = predicted_aux[i]

        if a == 0 and p > 0:
            e = 1
        elif a == 0 and p == 0:
            e = 0
        else:
            e = np.abs(a - p) / a

        if e > 1:
            e = 1

        errors.append(e)

    return errors


async def _naive_Errorsing(actual: np.ndarray, seasonality: int = 1):
    """ Naive Errorsing method which just repeats previous samples """
    return actual[:-seasonality]


async def _relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive Errorsing
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
            return await _error(actual[seasonality:], predicted[seasonality:]) / \
                (await _error(actual[seasonality:],
                              await _naive_Errorsing(actual, seasonality)) + EPSILON)

    return await _error(actual, predicted) / (await _error(actual, benchmark) + EPSILON)


async def _bounded_relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Bounded Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive Errorsing
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(await _error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(
            await _error(actual[seasonality:], await _naive_Errorsing(actual, seasonality)))
    else:
        abs_err = np.abs(await _error(actual, predicted))
        abs_err_bench = np.abs(await _error(actual, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)


async def _geometric_mean(a, axis=0, dtype=None):
    """ Geometric mean """

    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the async default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


async def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)
    return mean_squared_error(actual, predicted, squared=True)


async def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    return mean_squared_error(actual, predicted, squared=False)


async def me(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Error """

    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    return np.mean(await _error(actual, predicted))


async def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """

    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    return mean_absolute_error(actual, predicted)


async def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
            + Easy to interpret
            + Scale independent
            - Biased, not symmetric
            - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """

    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    return mean_absolute_percentage_error(actual, predicted)


async def mape_2(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
            + Easy to interpret
            + Scale independent
            - Biased, not symmetric
            - Undefined when actual[t] == 0
    Note: result is multiplied by 100
    """

    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    return np.mean(np.abs(await _percentage_error_2(actual, predicted)))


async def smape(actual: np.ndarray, predicted: np.ndarray):
    """mape = skmetrics.mean_absolute_percentage_error(y_true, y_pred)
    Symmetric Mean Absolute Percentage Error
    Note: result is multiplied by 100
    """
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)
    # Supress/hide the warning
    np.seterr(invalid='ignore')
    smape = 2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
    smape = np.nan_to_num(smape)
    return np.mean(smape)*100


async def wape(actual: np.ndarray, predicted: np.ndarray):
    """
    Weight Absolute Percentage error
    Note: result is multiplied by 100
    """

    sum_errors = 0
    sum_actual = 0
    for i in range(0, len(actual)):
        a = actual[i]
        p = predicted[i]

        sum_errors += np.abs(a - p)
        sum_actual += a

    return (sum_errors / sum_actual) * 100

async def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive Errorsing (shifted by @seasonality)
    """

    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    naive_mae = await mae(actual[seasonality:], await _naive_Errorsing(actual, seasonality))
    if naive_mae == 0:
        return 0
    return await mae(actual, predicted) / await mae(actual[seasonality:], await _naive_Errorsing(actual, seasonality))

    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    mase = mean_squared_error(actual, predicted)

    return np.sqrt(np.mean(np.sqrt(mse)))


async def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Squared Percentage Error
    Note: result is multiplied by 100
    """

    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    mse = mean_squared_error(actual, predicted)

    return np.sqrt(np.mean(np.sqrt(mse)))


async def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """ Root Mean Squared Scaled Error """

    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    q = np.abs(await _error(actual, predicted)) / await mae(actual[seasonality:], await _naive_Errorsing(actual, seasonality))
    return np.sqrt(np.mean(np.square(q)))


async def r2(actual: np.ndarray, predicted: np.ndarray):
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)

    return r2_score(actual, predicted)


async def accuracy_score_(actual, predicted):
    return accuracy_score(actual, predicted)
