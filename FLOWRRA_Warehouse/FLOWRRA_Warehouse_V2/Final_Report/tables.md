# Final deploy_eval: tables (build log entry 44)

Lead time counts each undelivered order at its minimum wait (end of the 500-step release window to the end of the day); `lead (delivered only)` is the plain mean. Firm = Wilcoxon p < 0.05 on days paired by map, fleet size and seed.

## Clean (128 days per arm)

### Headline (means per day)

| arm       |   delivered % |   lead time |   lead (delivered only) |   collisions/day |   guard stops/day |   L3/day |   planning ms/step |
|:----------|--------------:|------------:|------------------------:|-----------------:|------------------:|---------:|-------------------:|
| ladder4   |         98.87 |      112.09 |                  106.22 |             0    |           1058.62 |    35.88 |             nan    |
| ladder6   |         94.21 |      148.91 |                  123.17 |             0    |           1204.14 |    82.88 |             nan    |
| ladder7   |         93.59 |      154.5  |                  126.01 |             0.01 |           1523.12 |   105.37 |             nan    |
| RULES     |         95.46 |      125.66 |                  102.22 |             0    |            584.22 |   140.36 |             nan    |
| PIBT      |         98.38 |       86.18 |                   76.16 |             0    |              0    |   nan    |               0.89 |
| RHCR-PP   |         94.79 |      103.25 |                   73.83 |             0    |           1974.77 |   nan    |             114.65 |
| RHCR-PIBT |         75.15 |      209.89 |                   63.47 |             0    |           5870.43 |   nan    |              53.47 |

### By map

| arm       |   ('delivered_pct', 'Kiva-style') |   ('delivered_pct', 'large') |   ('delivered_pct', 'medium') |   ('delivered_pct', 'small') |   ('lead_cens', 'Kiva-style') |   ('lead_cens', 'large') |   ('lead_cens', 'medium') |   ('lead_cens', 'small') |
|:----------|----------------------------------:|-----------------------------:|------------------------------:|-----------------------------:|------------------------------:|-------------------------:|--------------------------:|-------------------------:|
| ladder4   |                              95.6 |                         99.9 |                          99.9 |                        100   |                         215.5 |                    108.1 |                      76.5 |                     48.3 |
| ladder6   |                              76.9 |                        100   |                         100   |                         99.9 |                         347.7 |                    114.4 |                      77.6 |                     55.9 |
| ladder7   |                              74.9 |                        100   |                         100   |                         99.4 |                         369.5 |                    112.4 |                      81.8 |                     54.3 |
| RULES     |                              83.5 |                         99.4 |                          99.1 |                         99.9 |                         270   |                    108.6 |                      79.1 |                     45   |
| PIBT      |                              93.5 |                        100   |                         100   |                        100   |                         121.3 |                    106.9 |                      73.3 |                     43.3 |
| RHCR-PP   |                              84.8 |                         99.5 |                          99.1 |                         95.8 |                         177   |                    106.1 |                      74.5 |                     55.4 |
| RHCR-PIBT |                              33.4 |                         96.6 |                          93.5 |                         77   |                         483.2 |                    118.1 |                      96.6 |                    141.7 |

### By fleet size

| arm       |   ('delivered_pct', 20) |   ('delivered_pct', 40) |   ('delivered_pct', 60) |   ('delivered_pct', 80) |   ('lead_cens', 20) |   ('lead_cens', 40) |   ('lead_cens', 60) |   ('lead_cens', 80) |
|:----------|------------------------:|------------------------:|------------------------:|------------------------:|--------------------:|--------------------:|--------------------:|--------------------:|
| ladder4   |                    99.6 |                    99.2 |                    99.1 |                    97.6 |               132   |               117.9 |               104.2 |                94.2 |
| ladder6   |                    89.8 |                    93.9 |                    95.4 |                    97.8 |               191.1 |               161.5 |               136.8 |               106.2 |
| ladder7   |                    91.4 |                    93.9 |                    95.1 |                    94   |               180.9 |               170.7 |               143.8 |               122.6 |
| RULES     |                    95.1 |                    95.3 |                    95.1 |                    96.3 |               150.7 |               141.8 |               117.3 |                92.9 |
| PIBT      |                    98.3 |                    98.2 |                    98.5 |                    98.5 |               114.8 |                95.8 |                75.7 |                58.4 |
| RHCR-PP   |                    93.5 |                    94.8 |                    95.3 |                    95.6 |               136.9 |               110.9 |                91.3 |                73.9 |
| RHCR-PIBT |                    75.5 |                    71.8 |                    74.6 |                    78.7 |               228.8 |               233.9 |               206.5 |               170.3 |

### Paired against ladder4 and against RULES

|                                           |   arm_mean |   ref_mean |     diff |   days_better |   days_worse |   n |        p | firm   |
|:------------------------------------------|-----------:|-----------:|---------:|--------------:|-------------:|----:|---------:|:-------|
| ('ladder4', 'ladder6', 'delivered_pct')   |      94.21 |      98.87 |    -4.66 |             5 |           31 | 128 |   0      | False  |
| ('ladder4', 'ladder6', 'lead_cens')       |     148.91 |     112.09 |    36.82 |            32 |           96 | 128 |   0      | False  |
| ('ladder4', 'ladder6', 'guard_stops')     |    1204.14 |    1058.62 |   145.52 |            46 |           64 | 128 |   0.0171 | False  |
| ('ladder4', 'ladder6', 'collisions')      |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'ladder7', 'delivered_pct')   |      93.59 |      98.87 |    -5.27 |             3 |           38 | 128 |   0      | False  |
| ('ladder4', 'ladder7', 'lead_cens')       |     154.5  |     112.09 |    42.41 |            29 |           99 | 128 |   0      | False  |
| ('ladder4', 'ladder7', 'guard_stops')     |    1523.12 |    1058.62 |   464.49 |            32 |           85 | 128 |   0      | False  |
| ('ladder4', 'ladder7', 'collisions')      |       0.01 |       0    |     0.01 |             0 |            1 | 128 |   0.3173 | False  |
| ('ladder4', 'RULES', 'delivered_pct')     |      95.46 |      98.87 |    -3.4  |             3 |           52 | 128 |   0      | False  |
| ('ladder4', 'RULES', 'lead_cens')         |     125.66 |     112.09 |    13.57 |            53 |           75 | 128 |   0.0068 | False  |
| ('ladder4', 'RULES', 'guard_stops')       |     584.22 |    1058.62 |  -474.41 |            70 |           28 | 128 |   0.0001 | True   |
| ('ladder4', 'RULES', 'collisions')        |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'PIBT', 'delivered_pct')      |      98.38 |      98.87 |    -0.49 |            14 |           18 | 128 |   0.181  | False  |
| ('ladder4', 'PIBT', 'lead_cens')          |      86.18 |     112.09 |   -25.91 |           100 |           28 | 128 |   0      | True   |
| ('ladder4', 'PIBT', 'guard_stops')        |       0    |    1058.62 | -1058.62 |            92 |            0 | 128 |   0      | True   |
| ('ladder4', 'PIBT', 'collisions')         |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'RHCR-PP', 'delivered_pct')   |      94.79 |      98.87 |    -4.08 |             7 |           69 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PP', 'lead_cens')       |     103.25 |     112.09 |    -8.84 |            77 |           51 | 128 |   0.0119 | True   |
| ('ladder4', 'RHCR-PP', 'guard_stops')     |    1974.77 |    1058.62 |   916.15 |            52 |           51 | 128 |   0.0022 | False  |
| ('ladder4', 'RHCR-PP', 'collisions')      |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'RHCR-PIBT', 'delivered_pct') |      75.15 |      98.87 |   -23.72 |             1 |          120 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PIBT', 'lead_cens')     |     209.89 |     112.09 |    97.8  |             8 |          120 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PIBT', 'guard_stops')   |    5870.43 |    1058.62 |  4811.8  |             4 |          118 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PIBT', 'collisions')    |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('RULES', 'ladder4', 'delivered_pct')     |      98.87 |      95.46 |     3.4  |            52 |            3 | 128 |   0      | True   |
| ('RULES', 'ladder4', 'lead_cens')         |     112.09 |     125.66 |   -13.57 |            75 |           53 | 128 |   0.0068 | True   |
| ('RULES', 'ladder4', 'guard_stops')       |    1058.62 |     584.22 |   474.41 |            28 |           70 | 128 |   0.0001 | False  |
| ('RULES', 'ladder4', 'collisions')        |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('RULES', 'ladder6', 'delivered_pct')     |      94.21 |      95.46 |    -1.26 |            35 |           23 | 128 |   0.5017 | False  |
| ('RULES', 'ladder6', 'lead_cens')         |     148.91 |     125.66 |    23.25 |            38 |           90 | 128 |   0      | False  |
| ('RULES', 'ladder6', 'guard_stops')       |    1204.14 |     584.22 |   619.92 |            20 |           81 | 128 |   0      | False  |
| ('RULES', 'ladder6', 'collisions')        |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('RULES', 'ladder7', 'delivered_pct')     |      93.59 |      95.46 |    -1.87 |            31 |           31 | 128 |   0.0579 | False  |
| ('RULES', 'ladder7', 'lead_cens')         |     154.5  |     125.66 |    28.84 |            21 |          107 | 128 |   0      | False  |
| ('RULES', 'ladder7', 'guard_stops')       |    1523.12 |     584.22 |   938.9  |            14 |           94 | 128 |   0      | False  |
| ('RULES', 'ladder7', 'collisions')        |       0.01 |       0    |     0.01 |             0 |            1 | 128 |   0.3173 | False  |
| ('RULES', 'PIBT', 'delivered_pct')        |      98.38 |      95.46 |     2.91 |            50 |            6 | 128 |   0      | True   |
| ('RULES', 'PIBT', 'lead_cens')            |      86.18 |     125.66 |   -39.48 |            97 |           31 | 128 |   0      | True   |
| ('RULES', 'PIBT', 'guard_stops')          |       0    |     584.22 |  -584.22 |            63 |            0 | 128 |   0      | True   |
| ('RULES', 'PIBT', 'collisions')           |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('RULES', 'RHCR-PP', 'delivered_pct')     |      94.79 |      95.46 |    -0.68 |            36 |           49 | 128 |   0.1028 | False  |
| ('RULES', 'RHCR-PP', 'lead_cens')         |     103.25 |     125.66 |   -22.41 |            84 |           43 | 128 |   0.0001 | True   |
| ('RULES', 'RHCR-PP', 'guard_stops')       |    1974.77 |     584.22 |  1390.55 |            24 |           56 | 128 |   0      | False  |
| ('RULES', 'RHCR-PP', 'collisions')        |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('RULES', 'RHCR-PIBT', 'delivered_pct')   |      75.15 |      95.46 |   -20.32 |             4 |          114 | 128 |   0      | False  |
| ('RULES', 'RHCR-PIBT', 'lead_cens')       |     209.89 |     125.66 |    84.24 |            13 |          115 | 128 |   0      | False  |
| ('RULES', 'RHCR-PIBT', 'guard_stops')     |    5870.43 |     584.22 |  5286.21 |             1 |          118 | 128 |   0      | False  |
| ('RULES', 'RHCR-PIBT', 'collisions')      |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |

## Full (128 days per arm)

### Headline (means per day)

| arm             |   delivered % |   lead time |   lead (delivered only) |   collisions/day |   guard stops/day |   L3/day |   near misses with people/day |   failures/day |   rescues/day |   orders lost (naive)/day |   planning ms/step |
|:----------------|--------------:|------------:|------------------------:|-----------------:|------------------:|---------:|------------------------------:|---------------:|--------------:|--------------------------:|-------------------:|
| ladder4         |         87.63 |      192.33 |                  128.92 |             0    |           1175.08 |    41.59 |                         77.41 |          11.53 |         11.34 |                      0    |             nan    |
| ladder6         |         82.14 |      234.37 |                  140.33 |             0    |           1138.11 |    68.62 |                         74.73 |          11.72 |         10.3  |                      0    |             nan    |
| ladder7         |         82.77 |      240.52 |                  144.57 |             0    |           1476.39 |   107.78 |                         61.49 |          11.62 |         11.2  |                      0    |             nan    |
| RULES           |         87.48 |      190.67 |                  126.27 |             0.02 |            499.84 |   117.14 |                         84.02 |          11.51 |         11.26 |                      0    |             nan    |
| PIBT            |         79.17 |      188.36 |                   71.31 |             0    |              0    |   nan    |                         47.22 |          11.51 |          7.81 |                      0    |               0.88 |
| RHCR-PP         |         86.28 |      165.14 |                   89.61 |             0    |           1283.95 |   nan    |                         52.37 |          11.58 |         11.07 |                      0    |             118.6  |
| RHCR-PIBT       |         74.57 |      270.1  |                  106.76 |             0    |           4693.72 |   nan    |                         79.41 |          11.7  |         10.3  |                      0    |              51.81 |
| RHCR-PIBT+naive |         75.25 |      264.76 |                  102.87 |             0    |           4732.34 |   nan    |                         77.27 |          11.69 |          8.27 |                      2.52 |              50.12 |

### By map

| arm             |   ('delivered_pct', 'Kiva-style') |   ('delivered_pct', 'large') |   ('delivered_pct', 'medium') |   ('delivered_pct', 'small') |   ('lead_cens', 'Kiva-style') |   ('lead_cens', 'large') |   ('lead_cens', 'medium') |   ('lead_cens', 'small') |
|:----------------|----------------------------------:|-----------------------------:|------------------------------:|-----------------------------:|------------------------------:|-------------------------:|--------------------------:|-------------------------:|
| ladder4         |                              84.3 |                         86.3 |                          89.8 |                         90.1 |                         355.3 |                    180.4 |                     133.3 |                    100.3 |
| ladder6         |                              70.3 |                         81.6 |                          86.2 |                         90.5 |                         492.8 |                    189.1 |                     145.6 |                    110   |
| ladder7         |                              64.5 |                         84.5 |                          90.5 |                         91.5 |                         540.5 |                    182.9 |                     136.7 |                    101.9 |
| RULES           |                              84   |                         85.6 |                          89.8 |                         90.6 |                         361.6 |                    180.6 |                     129.8 |                     90.7 |
| PIBT            |                              77.2 |                         79.1 |                          82.4 |                         78.1 |                         283.7 |                    187.5 |                     146.2 |                    136   |
| RHCR-PP         |                              83   |                         84.4 |                          89.4 |                         88.3 |                         248   |                    181.3 |                     132.2 |                     99   |
| RHCR-PIBT       |                              46.5 |                         82.9 |                          87.1 |                         81.8 |                         604.5 |                    189.5 |                     142.4 |                    143.9 |
| RHCR-PIBT+naive |                              46.7 |                         85.2 |                          87.9 |                         81.1 |                         604.8 |                    175.8 |                     136.6 |                    141.8 |

### By fleet size

| arm             |   ('delivered_pct', 20) |   ('delivered_pct', 40) |   ('delivered_pct', 60) |   ('delivered_pct', 80) |   ('lead_cens', 20) |   ('lead_cens', 40) |   ('lead_cens', 60) |   ('lead_cens', 80) |
|:----------------|------------------------:|------------------------:|------------------------:|------------------------:|--------------------:|--------------------:|--------------------:|--------------------:|
| ladder4         |                    80.5 |                    89.2 |                    91.3 |                    89.6 |               276.5 |               190.3 |               155   |               147.5 |
| ladder6         |                    64.5 |                    86   |                    88.7 |                    89.4 |               361.1 |               224.7 |               196.2 |               155.4 |
| ladder7         |                    74.4 |                    83.2 |                    86.5 |                    86.9 |               325.5 |               242   |               210   |               184.6 |
| RULES           |                    81.8 |                    89.1 |                    89.9 |                    89   |               261.3 |               196.1 |               161.9 |               143.4 |
| PIBT            |                    62.4 |                    83.6 |                    84.6 |                    86.1 |               305.9 |               171.2 |               154.4 |               121.9 |
| RHCR-PP         |                    77.8 |                    89.1 |                    89.2 |                    89   |               266.6 |               147.5 |               131.4 |               115.1 |
| RHCR-PIBT       |                    69.3 |                    74   |                    76.9 |                    78.1 |               341.3 |               274.3 |               245.4 |               219.3 |
| RHCR-PIBT+naive |                    70.8 |                    74.5 |                    76.7 |                    79   |               330.7 |               268.2 |               244.5 |               215.6 |

### Paired against ladder4 and against RULES

|                                                            |   arm_mean |   ref_mean |     diff |   days_better |   days_worse |   n |        p | firm   |
|:-----------------------------------------------------------|-----------:|-----------:|---------:|--------------:|-------------:|----:|---------:|:-------|
| ('ladder4', 'ladder6', 'delivered_pct')                    |      82.14 |      87.63 |    -5.49 |            40 |           70 | 128 |   0      | False  |
| ('ladder4', 'ladder6', 'lead_cens')                        |     234.37 |     192.33 |    42.04 |            37 |           91 | 128 |   0      | False  |
| ('ladder4', 'ladder6', 'guard_stops')                      |    1138.11 |    1175.08 |   -36.97 |            51 |           65 | 128 |   0.3003 | False  |
| ('ladder4', 'ladder6', 'collisions')                       |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'ladder6', 'obstacle_near_miss_steps')         |      74.73 |      77.41 |    -2.68 |            55 |           58 | 128 |   0.8827 | False  |
| ('ladder4', 'ladder7', 'delivered_pct')                    |      82.77 |      87.63 |    -4.86 |            55 |           56 | 128 |   0.0098 | False  |
| ('ladder4', 'ladder7', 'lead_cens')                        |     240.52 |     192.33 |    48.19 |            40 |           88 | 128 |   0      | False  |
| ('ladder4', 'ladder7', 'guard_stops')                      |    1476.39 |    1175.08 |   301.31 |            41 |           72 | 128 |   0.0005 | False  |
| ('ladder4', 'ladder7', 'collisions')                       |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'ladder7', 'obstacle_near_miss_steps')         |      61.49 |      77.41 |   -15.91 |            58 |           54 | 128 |   0.7704 | False  |
| ('ladder4', 'RULES', 'delivered_pct')                      |      87.48 |      87.63 |    -0.15 |            47 |           53 | 128 |   0.3963 | False  |
| ('ladder4', 'RULES', 'lead_cens')                          |     190.67 |     192.33 |    -1.66 |            71 |           57 | 128 |   0.2709 | False  |
| ('ladder4', 'RULES', 'guard_stops')                        |     499.84 |    1175.08 |  -675.23 |            79 |           21 | 128 |   0      | True   |
| ('ladder4', 'RULES', 'collisions')                         |       0.02 |       0    |     0.02 |             0 |            2 | 128 |   0.1573 | False  |
| ('ladder4', 'RULES', 'obstacle_near_miss_steps')           |      84.02 |      77.41 |     6.61 |            52 |           53 | 128 |   0.4343 | False  |
| ('ladder4', 'PIBT', 'delivered_pct')                       |      79.17 |      87.63 |    -8.46 |             9 |          116 | 128 |   0      | False  |
| ('ladder4', 'PIBT', 'lead_cens')                           |     188.36 |     192.33 |    -3.97 |            49 |           79 | 128 |   0.1028 | False  |
| ('ladder4', 'PIBT', 'guard_stops')                         |       0    |    1175.08 | -1175.08 |            95 |            0 | 128 |   0      | True   |
| ('ladder4', 'PIBT', 'collisions')                          |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'PIBT', 'obstacle_near_miss_steps')            |      47.22 |      77.41 |   -30.19 |            59 |           50 | 128 |   0.192  | False  |
| ('ladder4', 'RHCR-PP', 'delivered_pct')                    |      86.28 |      87.63 |    -1.36 |            40 |           66 | 128 |   0.0007 | False  |
| ('ladder4', 'RHCR-PP', 'lead_cens')                        |     165.14 |     192.33 |   -27.18 |            86 |           42 | 128 |   0      | True   |
| ('ladder4', 'RHCR-PP', 'guard_stops')                      |    1283.95 |    1175.08 |   108.88 |            65 |           36 | 128 |   0.7773 | False  |
| ('ladder4', 'RHCR-PP', 'collisions')                       |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'RHCR-PP', 'obstacle_near_miss_steps')         |      52.37 |      77.41 |   -25.04 |            75 |           33 | 128 |   0.0006 | True   |
| ('ladder4', 'RHCR-PIBT', 'delivered_pct')                  |      74.57 |      87.63 |   -13.06 |            15 |          103 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PIBT', 'lead_cens')                      |     270.1  |     192.33 |    77.78 |            22 |          106 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PIBT', 'guard_stops')                    |    4693.72 |    1175.08 |  3518.64 |            10 |          110 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PIBT', 'collisions')                     |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'RHCR-PIBT', 'obstacle_near_miss_steps')       |      79.41 |      77.41 |     2.01 |            63 |           43 | 128 |   0.1839 | False  |
| ('ladder4', 'RHCR-PIBT+naive', 'delivered_pct')            |      75.25 |      87.63 |   -12.38 |            24 |           96 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PIBT+naive', 'lead_cens')                |     264.76 |     192.33 |    72.43 |            36 |           92 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PIBT+naive', 'guard_stops')              |    4732.34 |    1175.08 |  3557.27 |            10 |          111 | 128 |   0      | False  |
| ('ladder4', 'RHCR-PIBT+naive', 'collisions')               |       0    |       0    |     0    |             0 |            0 | 128 | nan      | False  |
| ('ladder4', 'RHCR-PIBT+naive', 'obstacle_near_miss_steps') |      77.27 |      77.41 |    -0.13 |            67 |           44 | 128 |   0.0755 | False  |
| ('RULES', 'ladder4', 'delivered_pct')                      |      87.63 |      87.48 |     0.15 |            53 |           47 | 128 |   0.3963 | False  |
| ('RULES', 'ladder4', 'lead_cens')                          |     192.33 |     190.67 |     1.66 |            57 |           71 | 128 |   0.2709 | False  |
| ('RULES', 'ladder4', 'guard_stops')                        |    1175.08 |     499.84 |   675.23 |            21 |           79 | 128 |   0      | False  |
| ('RULES', 'ladder4', 'collisions')                         |       0    |       0.02 |    -0.02 |             2 |            0 | 128 |   0.1573 | False  |
| ('RULES', 'ladder4', 'obstacle_near_miss_steps')           |      77.41 |      84.02 |    -6.61 |            53 |           52 | 128 |   0.4343 | False  |
| ('RULES', 'ladder6', 'delivered_pct')                      |      82.14 |      87.48 |    -5.34 |            42 |           70 | 128 |   0.0001 | False  |
| ('RULES', 'ladder6', 'lead_cens')                          |     234.37 |     190.67 |    43.7  |            30 |           98 | 128 |   0      | False  |
| ('RULES', 'ladder6', 'guard_stops')                        |    1138.11 |     499.84 |   638.27 |            17 |           85 | 128 |   0      | False  |
| ('RULES', 'ladder6', 'collisions')                         |       0    |       0.02 |    -0.02 |             2 |            0 | 128 |   0.1573 | False  |
| ('RULES', 'ladder6', 'obstacle_near_miss_steps')           |      74.73 |      84.02 |    -9.29 |            56 |           55 | 128 |   0.7763 | False  |
| ('RULES', 'ladder7', 'delivered_pct')                      |      82.77 |      87.48 |    -4.71 |            53 |           57 | 128 |   0.0118 | False  |
| ('RULES', 'ladder7', 'lead_cens')                          |     240.52 |     190.67 |    49.85 |            32 |           96 | 128 |   0      | False  |
| ('RULES', 'ladder7', 'guard_stops')                        |    1476.39 |     499.84 |   976.55 |            14 |           85 | 128 |   0      | False  |
| ('RULES', 'ladder7', 'collisions')                         |       0    |       0.02 |    -0.02 |             2 |            0 | 128 |   0.1573 | False  |
| ('RULES', 'ladder7', 'obstacle_near_miss_steps')           |      61.49 |      84.02 |   -22.52 |            51 |           58 | 128 |   0.8077 | False  |
| ('RULES', 'PIBT', 'delivered_pct')                         |      79.17 |      87.48 |    -8.31 |            10 |          114 | 128 |   0      | False  |
| ('RULES', 'PIBT', 'lead_cens')                             |     188.36 |     190.67 |    -2.31 |            42 |           86 | 128 |   0.0123 | True   |
| ('RULES', 'PIBT', 'guard_stops')                           |       0    |     499.84 |  -499.84 |            55 |            0 | 128 |   0      | True   |
| ('RULES', 'PIBT', 'collisions')                            |       0    |       0.02 |    -0.02 |             2 |            0 | 128 |   0.1573 | False  |
| ('RULES', 'PIBT', 'obstacle_near_miss_steps')              |      47.22 |      84.02 |   -36.8  |            69 |           38 | 128 |   0.0041 | True   |
| ('RULES', 'RHCR-PP', 'delivered_pct')                      |      86.28 |      87.48 |    -1.2  |            41 |           71 | 128 |   0.036  | False  |
| ('RULES', 'RHCR-PP', 'lead_cens')                          |     165.14 |     190.67 |   -25.52 |            74 |           54 | 128 |   0.0091 | True   |
| ('RULES', 'RHCR-PP', 'guard_stops')                        |    1283.95 |     499.84 |   784.11 |            22 |           48 | 128 |   0      | False  |
| ('RULES', 'RHCR-PP', 'collisions')                         |       0    |       0.02 |    -0.02 |             2 |            0 | 128 |   0.1573 | False  |
| ('RULES', 'RHCR-PP', 'obstacle_near_miss_steps')           |      52.37 |      84.02 |   -31.65 |            70 |           34 | 128 |   0.0014 | True   |
| ('RULES', 'RHCR-PIBT', 'delivered_pct')                    |      74.57 |      87.48 |   -12.91 |            14 |          100 | 128 |   0      | False  |
| ('RULES', 'RHCR-PIBT', 'lead_cens')                        |     270.1  |     190.67 |    79.44 |            18 |          110 | 128 |   0      | False  |
| ('RULES', 'RHCR-PIBT', 'guard_stops')                      |    4693.72 |     499.84 |  4193.88 |             2 |          113 | 128 |   0      | False  |
| ('RULES', 'RHCR-PIBT', 'collisions')                       |       0    |       0.02 |    -0.02 |             2 |            0 | 128 |   0.1573 | False  |
| ('RULES', 'RHCR-PIBT', 'obstacle_near_miss_steps')         |      79.41 |      84.02 |    -4.6  |            62 |           39 | 128 |   0.1915 | False  |
| ('RULES', 'RHCR-PIBT+naive', 'delivered_pct')              |      75.25 |      87.48 |   -12.23 |            26 |           90 | 128 |   0      | False  |
| ('RULES', 'RHCR-PIBT+naive', 'lead_cens')                  |     264.76 |     190.67 |    74.09 |            34 |           94 | 128 |   0      | False  |
| ('RULES', 'RHCR-PIBT+naive', 'guard_stops')                |    4732.34 |     499.84 |  4232.5  |             2 |          115 | 128 |   0      | False  |
| ('RULES', 'RHCR-PIBT+naive', 'collisions')                 |       0    |       0.02 |    -0.02 |             2 |            0 | 128 |   0.1573 | False  |
| ('RULES', 'RHCR-PIBT+naive', 'obstacle_near_miss_steps')   |      77.27 |      84.02 |    -6.74 |            67 |           39 | 128 |   0.0359 | True   |

## Predictions P1-P6, scored

**P1 PASS.** Collisions per day: highest arm mean 0.016 (clean max 0.008, full max 0.016).

**P2 FAIL.** Full, 60-80 robots, delivered %: ladder4 vs PIBT 90.5 / 85.3 (p 0.000); ladder4 vs RHCR-PP 90.5 / 89.1 (p 0.003); ladder4 vs RHCR-PIBT 90.5 / 77.5 (p 0.000); ladder4 vs RHCR-PIBT+naive 90.5 / 77.9 (p 0.000); ladder6 vs PIBT 89.0 / 85.3 (p 0.000); ladder6 vs RHCR-PP 89.0 / 89.1 (p 0.539) -- not firm; ladder6 vs RHCR-PIBT 89.0 / 77.5 (p 0.000); ladder6 vs RHCR-PIBT+naive 89.0 / 77.9 (p 0.000)

**P3 PASS.** Fewest guard stops: clean PIBT (0.0), full PIBT (0.0); fewest near misses with people (full): PIBT (47.2; next RHCR-PP 52.4).

**P4 PASS.** Clean lead time: 20 robots: PIBT 114.8 vs RULES 150.7 (p 0.017); 40 robots: PIBT 95.8 vs RULES 141.8 (p 0.000).

**P5 FAIL.** clean: guard stops 1975 vs 5870 (3.0x); delivered at 60-80 95.4 vs 76.7 (p 0.000); full: guard stops 1284 vs 4694 (3.7x); delivered at 60-80 89.1 vs 77.5 (p 0.000).

**P6 FAIL.** Naive delivered more than RHCR-PIBT on 44 of 128 days; on the 29 days with nothing lost or abandoned it was identical on 29; orders lost by any other arm: 0.
