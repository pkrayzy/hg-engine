.include "asm/include/battle_commands.inc"

.data

_000:
    TryFutureSight _010
    UpdateVar OPCODE_SET, BSCRIPT_VAR_SIDE_EFFECT_FLAGS_INDIRECT, MOVE_SIDE_EFFECT_ON_HIT|MOVE_SUBSCRIPT_PTR_PRINT_MESSAGE_AND_PLAY_ANIMATION
    UpdateVar OPCODE_FLAG_ON, BSCRIPT_VAR_BATTLE_STATUS, BATTLE_STATUS_SHADOW_FORCE|BATTLE_STATUS_IGNORE_TYPE_IMMUNITY|BATTLE_STATUS_FLAT_HIT_RATE|BATTLE_STATUS_HIT_DIVE|BATTLE_STATUS_HIT_DIG|BATTLE_STATUS_HIT_FLY
    End 

_010:
    PrintAttackMessage 
    Wait 
    WaitButtonABTime 30
    Call BATTLE_SUBSCRIPT_BUT_IT_FAILED
    End 
