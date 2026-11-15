# HG Engine Data Structure Reference

This document describes the key data structures for modifying Pokemon stats, moves, and understanding the damage calculation system in the HG Engine.

## Pokemon Base Stats

Pokemon base stats are defined in `data/Species.c` within the `sSpeciesData` array.

Each entry in the array is of type `SpeciesDataEntry` (defined in `include/species_data.h`):

```c
typedef struct SpeciesDataEntry {
    SpeciesData speciesData;
    SpeciesMetrics metricsData;
    SpeciesTextData textData;
} SpeciesDataEntry;
```

The `SpeciesData` struct contains the base stats:

```c
typedef struct SpeciesBaseStats {
    u8 hp;
    u8 attack;
    u8 defense;
    u8 spAttack;
    u8 spDefense;
    u8 speed;
} SpeciesBaseStats;
```

To modify a Pokemon's base stats:
1. Locate the Pokemon's entry in `data/Species.c` (indexed by species constant from `include/constants/species.h`).
2. Modify the values in the `.baseStats` substructure.

Example:
```c
[SPECIES_PIKACHU] = {
    .textData = { ... },
    .speciesData = {
        .baseStats = {
            .hp = 35,
            .attack = 55,
            .defense = 40,
            .spAttack = 50,
            .spDefense = 50,
            .speed = 90,
        },
        // ... other fields
    },
    // ... other data
};
```

## Move Data

Move data is defined in `data/Moves.c` within the `sMoveSource` array.

Each entry is of type `MoveSourceEntry` (defined in `include/move_data.h`):

```c
typedef struct MoveSourceEntry {
    MoveNames names;
    MoveDataFields data;
    MoveBattleFields battle;
    MoveContestFields contest;
    const char *description;
} MoveSourceEntry;
```

The key substructures are:

### MoveDataFields
```c
typedef struct MoveDataFields {
    u16 effect;      // Move effect (see constants/move_effects.h)
    u8 split;        // SPLIT_PHYSICAL (0), SPLIT_SPECIAL (1), SPLIT_STATUS (2)
    u8 power;        // Base power (0 for status moves)
    u8 type;         // Move type (see constants/types.h)
    u8 accuracy;     // Accuracy percentage
    u8 pp;           // Power Points
    u8 effectChance; // Chance for secondary effect (0-100)
} MoveDataFields;
```

### MoveBattleFields
```c
typedef struct MoveBattleFields {
    u16 target;      // Target type (see constants/battle.h)
    s8 priority;     // Priority bracket (-6 to +6)
    u8 flags;        // Move flags (see constants/battle.h for flag definitions)
} MoveBattleFields;
```

To modify a move:
1. Locate the move's entry in `data/Moves.c` (indexed by move constant from `include/constants/moves.h`).
2. Modify the desired fields in the substructures.
3. Update the description string if needed.

Example:
```c
[MOVE_THUNDERBOLT] = {
    .names = { "Thunderbolt", "THUNDERBOLT", "Thunderbolt" },
    .data = {
        .effect = MOVE_EFFECT_THUNDERBOLT,
        .split = SPLIT_SPECIAL,
        .power = 90,
        .type = TYPE_ELECTRIC,
        .accuracy = 100,
        .pp = 15,
        .effectChance = 10, // 10% chance to paralyze
    },
    .battle = {
        .target = MOVE_TARGET_SELECTED,
        .priority = 0,
        .flags = FLAG_CONTACT | FLAG_PROTECT | FLAG_MIRROR_MOVE,
    },
    .contest = {
        .appeal = 2,
        .contestType = CONTEST_BEAUTY,
    },
    .description = "A strong electric blast launched at the target. May cause paralysis.",
};
```

## Damage Calculation System

The damage calculation is primarily handled in `src/battle/battle_calc_damage.c`.

Key functions:
- `CalcBaseDamage`: Calculates the base damage before modifiers.
- `CalcDamageOverall`: Orchestrates the overall damage calculation.
- `AdjustDamageForRoll`: Applies the random damage roll (85-100%).

The calculation follows the standard Pokemon damage formula:

```
Damage = (((2 * Level / 5 + 2) * Power * Atk/Def) / 50) + 2
```

Then modified by:
- Critical hits (×1.5)
- Random factor (0.85-1.00)
- STAB (Same Type Attack Bonus: ×1.5 if move type matches user's type)
- Type effectiveness (0, 0.25, 0.5, 1, 2, 4)
- Burn (×0.5 for physical attacks if burned)
- Other modifiers (abilities, items, field effects, etc.)

### Key Data Structures Used in Damage Calculation

In `src/battle/battle_calc_damage.c`, the `DamageCalcStruct` is used to pass data:

```c
struct DamageCalcStruct {
    int maxBattlers;
    int attackerPartySize;
    struct PartyPokemon *attackerParty[6];
    u8 attacker;
    u8 defender;
    u8 critical;
    u8 moveno;
    u8 movetype;
    u8 movesplit;
    u16 movepower;
    // ... other fields for calculation
};
```

### Modifying Damage Calculation

To change how damage is calculated:
1. Edit `src/battle/battle_calc_damage.c`
2. Focus on the `CalcBaseDamage` function for the core formula
3. Be cautious as changes affect all moves

Common modifications:
- Changing the base formula constants
- Modifying how stats are retrieved (attack/defense vs sp. attack/sp. defense based on split)
- Adjusting critical hit mechanics
- Changing type effectiveness calculations

## Other Battle Calculators

`src/battle/other_battle_calculators.c` is a collection of battle helpers used by the damage pipeline and other battle systems; it is not a single calculator. Its main components include:

- **Hit and turn-order calculations:** `CalcAccuracy` applies move accuracy and accuracy/evasion modifiers; `CalcSpeed`, `CalcPriorityAndQuickClawCustapBerry`, and `DynamicSortClientExecutionOrder` determine action ordering.
- **Critical hits:** `CalcCritical` determines whether a move crits, considering critical stage, items, abilities, and move effects. It returns critical state; the critical damage multiplier itself is applied in `battle_calc_damage.c`.
- **Type and damage-stat helpers:** `GetTypeEffectiveness` and `UpdateTypeEffectiveness` handle type matchups and move-specific overrides. `GetDynamicMoveType` resolves variable move types, `GetMoveSplit` resolves a move's physical/special/status split, and `getEquivalentAttackAndDefense` selects effective attack/defense stats and handles split/stat-stage exceptions.
- **Move classification and move lists:** helper predicates such as `IsMultiHitMove`, `IsPowderMove`, `IsBallOrBombMove`, `IsDanceMove`, and `IsMoveWindMove` consult lists near the top of the file. Those lists are part of the behavior: update them when changing a move's classification.
- **Move targeting and execution:** `IsValidMoveTarget`, `IsMoveSpreadMove`, `CanGetNextDefender`, and `SetupCurrentMoveContext` set up targets and move context; multi-hit and move-end helpers manage repeated hits and cleanup.
- **Battle-state, ability, and item helpers:** includes weather lookup, type changes, held-item removal checks, damage-reduction berry eligibility, substitute checks, switching/run checks, and ability interaction predicates.
- **Other battle systems:** includes legal move selection (such as Mimic and Metronome), capture restrictions, transform handling, replacement-mon checks, and trainer victory music selection.

For a damage change, trace the call sites as well as the helper: the main formula/modifier pipeline is in `battle_calc_damage.c`, while accuracy, critical-stage calculation, type effectiveness, move type/split, and effective attack/defense selection are supported here. Avoid treating similarly named helpers as the final damage multiplier; check where their results are consumed.

## Related Files

- `include/constants/move_effects.h`: Move effect IDs
- `include/constants/moves.h`: Move ID definitions
- `include/constants/species.h`: Species/Pokemon ID definitions
- `include/constants/types.h`: Type ID definitions
- `include/battle.h`: Battle system structures and constants
- `src/battle/other_battle_calculators.c`: Battle calculation and move/battle helper functions
- `src/battle/`: Other battle system files (status effects, abilities, items)

## Making Changes

When modifying stats or moves:
1. Make your changes to the appropriate `.c` file
2. Recompile the project using `make`
3. Test in-game to verify changes work as expected
4. For move effect changes, you may need to modify battle scripts in `data/battle_scripts/`

Remember that some values are limited by their data types (e.g., base stats are u8, so max 255).