import pandas as pd
from analysis import cleaning

# Minimal DataFrame to exercise the pipeline (skip SMOTE in the test)
Df = pd.DataFrame({
    'koi_disposition': [1, 0, 1],
    'feat1': [1.0, None, 3.0],
    'feat1_err': [0.1, 0.1, 0.1],
    'koi_fpflag_nt': [0, 1, 0],
    'koi_fpflag_ss': [0, 0, 1],
})

out = cleaning.pipeline(
    Df,
    apply_discards=False,
    apply_row_filter=True,
    apply_correlation_analysis=False,
    apply_separation=True,
    apply_prepare_features=True,
    apply_add_noise=True,
    apply_preprocess_and_smote=False,
    apply_save=False,
)

print('OUTPUT_KEYS', sorted(out.keys()))
if 'X_noised' in out and out['X_noised'] is not None:
    print('X_noised shape:', out['X_noised'].shape)
else:
    print('X_noised: None')
