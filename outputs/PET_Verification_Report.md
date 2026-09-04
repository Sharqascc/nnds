# PET Output Verification Report (Non-Technical)

This report explains, in simple terms, how we check that the PET (Post-Encroachment Time) values are correct and reliable.


## What is PET?

PET is the time gap between when one vehicle leaves a conflict area and another vehicle enters it. A smaller PET means the two vehicles passed through the same spot very close in time, which can indicate a potential conflict.


## How do we know the PET calculation is reliable?

We run a series of automatic checks on every PET event. Each check is like a 'yes/no' question, and we report how many events passed.


### Checks performed

| Check | Result | Details |

|-------|--------|---------|

| Are all PET values positive? | ✅ PASS | 187 of 187 events |

| Are the two vehicles different? | ✅ PASS | 187 of 187 events |

| Is the site label valid? | ✅ PASS | Sites: {'GITI': 153, 'MRC': 34} |

| Does recalculating PET from frame times match stored PET? | ✅ PASS | FPS = 30.0 |


## Summary Statistics

- **Total events**: 187

- **GITI site**: 153

- **MRC site**: 34

- **PET range**: 0.133 s to 2.999 s

- **Average PET**: 1.589 s


### Severity Distribution (based on PET)

- Critical (<1.0 s): 57

- Serious (1.0–1.5 s): 32

- Moderate (1.5–3.0 s): 98

- Safe (>3.0 s): 0


## Example Event (for understanding)

Event 0: Vehicle 2000 left the conflict zone at frame 178, Vehicle 7000 entered at frame 264. The gap is 86 frames, which at 30 FPS equals 2.866 seconds.

## Why this matters

These checks are run automatically by the code and also verified manually in the validation report. They confirm that the PET numbers are mathematically consistent and based on correctly ordered vehicle movements.
