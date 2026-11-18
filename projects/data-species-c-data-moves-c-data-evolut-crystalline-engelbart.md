# Plan: CSV export/import for data/Species.c, data/Moves.c, data/Evolutions.c

## Context
The user wants to edit Pokémon species, move, and evolution data more easily. These live in three large C designated-initializer arrays (`data/Species.c` 2.7MB, `data/Moves.c` 756KB, `data/Evolutions.c` 660KB). Hand-editing C structs is error-prone; a CSV round-trip (export → edit → import) lets users change values in a familiar tabular form and write them back into valid C that recompiles with `make`.

The project already has data generators in `tools/source/*` (C) and helper scripts in `scripts/` (Python). No existing CSV export/import for these three files was found. Node 26 is available; no npm dependency should be needed (use stdlib `fs` only).

## Recommended approach
One Node.js script, stdlib-only, with two modes:
- `export <Species.c|Moves.c|Evolutions.c> <out.csv>` — parse the C array into objects, write one CSV row per entry.
- `import <Species.c|Moves.c|Evolutions.c> <in.csv>` — read CSV, rebuild each entry's C text, splice it back into the original file preserving the includes/typedefs and the array wrapper.

### Parsing strategy
Parse each file's designated-initializer array with a small recursive-descent parser over the entry body:
- Split the file into the array header (`const ... __data[N] = {`), the entries, and the closing `};`.
- Each entry: `[CONSTANT] = { ... }`. Capture the constant name as the row key.
- Within an entry, handle `.substruct = { ... }` (recurse) and `field = value,`.
- Values are captured as *raw C tokens*: numbers verbatim, string literals with surrounding quotes, identifiers/constants (`TYPE_GRASS`, `ABILITY_OVERGROW`, `EVO_LEVEL`, `MOVE_EFFECT_HIT`, …) verbatim, arrays `{ a, b }` preserved.
- This keeps import faithful: a CSV cell holds exactly what appeared in C, so the user edits the literal token and import writes it back. Compiles as long as the user keeps valid constant names/numbers.

### CSV schemas (one file per CSV)
Key column = the array index constant (`species`, `move`, or `species`).

**Species.c** (`SpeciesDataEntry`): species, name, pokedexEntry, classification, height, weight, hp, attack, defense, spAttack, spDefense, speed, type1, type2, catchRate, baseExpRewardPadding, evHp, evAttack, evDefense, evSpAttack, evSpDefense, evSpeed, heldItemCommon, heldItemRare, genderRatio, hatchCycles, baseFriendship, expRate, eggGroup1, eggGroup2, ability1, ability2, safariFleeRate, bodyColor, flipSprite, heightDecimetres, weightHectograms, bodyType, fTrainerScale, fPokemonScale, mTrainerScale, mPokemonScale, fTrainerYOffset, fPokemonYOffset, mTrainerYOffset, mPokemonYOffset.

**Moves.c** (`MoveSourceEntry`): move, name, capsName, fullName, description, effect, split, power, type, accuracy, pp, effectChance, target, priority, flags, appeal, contestType.

**Evolutions.c** (`EvolutionTable`): species, then evo1_method/evo1_param/evo1_target … evo10_method/evo10_param/evo10_target (10 slots per `MAX_EVOS_PER_POKE`).

### Constants used in C values
Read from `include/constants/species.h`, `moves.h`, `types.h`, `move_effects.h`, `include/move_data.h` (SPLIT/FLAG/CONTEST/APPEAL), `include/pokemon.h` (EVO_*), `include/battle.h` (target/priority). The script treats any uppercase identifier token as a pass-through constant (no hard-coded enum table needed), so it adapts to the project's constants automatically.

### Round-trip self-check
`selftest` mode: export each of the 3 files to a temp CSV, import it back, normalize whitespace, and assert the regenerated array body matches the original. This is the required test for the new parser logic.

## Files to create
- `data_csv.js` (repo root) — the export/import/self-check script.

No existing files modified. Build verified with `make` after a sample import.

## Decisions (confirmed by user)
- Script location: repo root `data_csv.js`.
- CSV cells hold raw C tokens verbatim (numbers, string literals with quotes, identifier constants). No reverse constant table.
- Scope: `data/Species.c`, `data/Moves.c`, `data/Evolutions.c` only.

## Verification
1. `node data_csv.js selftest` passes (round-trip identical).
2. Export each file, confirm CSV row counts match species/move counts.
3. Hand-edit one CSV cell (e.g. a base stat), import, run `make` to confirm it compiles, and spot-check the regenerated C.
4. Confirm git diff only touches the intended data values.

## Usage
```bash
# Export to CSV
node data_csv.js export data/Species.c species.csv
node data_csv.js export data/Moves.c moves.csv
node data_csv.js export data/Evolutions.c evolutions.csv

# Import from CSV
node data_csv.js import data/Species.c species.csv
node data_csv.js import data/Moves.c moves.csv
node data_csv.js import data/Evolutions.c evolutions.csv

# Self-test (round-trip verification)
node data_csv.js selftest
```

### CSV Format
- Cells contain **raw C tokens verbatim**: numbers (`45`), string literals (`"Bulbasaur"`), identifiers (`TYPE_GRASS`, `ABILITY_OVERGROW`, `EVO_LEVEL`, `MOVE_EFFECT_HIT`), compound constants (`FLAG_CONTACT | FLAG_PROTECT`)
- No constant name mapping — edit the literal token, import writes it back
- String fields (names, descriptions) keep their quotes on import
