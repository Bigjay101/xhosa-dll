"""P9.6 selection tests on synthetic, explicitly non-semantic candidate mappings."""
import pandas as pd
from scripts.p9_tune import select_table


def test_best_modern_candidate_and_support():
    """An OOV historical word must map to one lexicon candidate, by frequency."""
    pool = pd.DataFrame([
        {'hist_form':'teta','modern_form':'thetha','family':'ASP','support':21,
         'n_edits':2,'modern_freq':40,'edit_distance':2},
        {'hist_form':'teta','modern_form':'tetha','family':'ASP','support':21,
         'n_edits':1,'modern_freq':2,'edit_distance':1},
    ])
    result = select_table(pool, min_support=20, max_edits=2)
    assert len(result) == 1 and result.iloc[0].modern_form == 'thetha'
    assert select_table(pool, 50, 2).empty
