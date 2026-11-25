# Data

The repository does not contain the data. Put the 15 per-source CSV files in
`data/raw/` (or set `data_dir` in the config):

    {rep} {source}.csv   rep in {dvm, jvm, hybrid}
                         source in {Drebin, Genome, CICInvesAndMal2019, Maldroid2020, Omer}

Columns: `uygulama_adi` (application id, dropped), one 0/1 column per opcode,
`Class` (1 = malware, 0 = benign). The merged tables `dvm.csv`, `jvm.csv`,
`hybrid.csv` may also be placed here; if present, `prepare` checks that the
per-source files reproduce them row for row.

`results/data/data_card.json` records the SHA-256 of every input file, the
number of rows and columns removed at each cleaning step, and the size of the
scenario splits.
