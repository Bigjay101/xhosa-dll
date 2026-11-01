"""P10 core synthetic tests: exact table, preservation, reproducible matched edits."""
from collections import Counter
from src.normalise import normalise_text
from src.historicise import historicise_text, estimate_rates
from src.noise import add_letter_noise

RULES = {'accepted_families':['ASP','BH'], 'table':[
    {'hist_form':'teta','modern_form':'thetha','family':'ASP','support':100},
    {'hist_form':'bala','modern_form':'bhala','family':'BH','support':20},
]}
LEX = Counter({'thetha':100, 'bhala':40})


def test_norm_punctuation_case_idempotent():
    """Normalisation changes OOV words only and preserves punctuation."""
    result, logs = normalise_text('Teta, thetha!', LEX, RULES)
    assert result == 'Thetha, thetha!'
    assert len(logs) == 1 and logs[0]['family'] == 'ASP'
    assert normalise_text(result, LEX, RULES)[0] == result


def test_families_and_inverse():
    """A family filter cannot alter other families; inverse dose is deterministic."""
    assert normalise_text('teta bala', LEX, RULES, families=['ASP'])[0] == 'thetha bala'
    x, logs = historicise_text('thetha bhala', RULES, dose=1.0)
    assert x == 'teta bala' and len(logs) == 2
    assert historicise_text('thetha', RULES, dose=0.4, seed=5) == historicise_text('thetha', RULES, dose=0.4, seed=5)


def test_rates_and_noise():
    """Rates use H-dev logs; matched random edits maintain original word count."""
    rates = estimate_rates([[{'family':'ASP'}], []], ['thetha', 'thetha'], RULES)
    assert rates['ASP'] == 0.5
    text = ' '.join(['thetha']*1000)
    noisy, changed = add_letter_noise(text, 0.2, seed=13)
    assert 140 <= changed <= 260 and len(noisy.split()) == 1000
