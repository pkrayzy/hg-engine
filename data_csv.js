#!/usr/bin/env node
/**
 * data_csv.js - Export/import C data arrays to/from CSV
 * Usage:
 *   node data_csv.js export data/Species.c species.csv
 *   node data_csv.js import data/Species.c species.csv
 *   node data_csv.js selftest
 */

const fs = require('fs');
const path = require('path');

// ============================================================================
// CSV UTILITIES
// ============================================================================

function escapeCsvCell(cell) {
    if (cell === undefined || cell === null) return '';
    const str = String(cell);
    if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
        return '"' + str.replace(/"/g, '""') + '"';
    }
    return str;
}

function unescapeCsvCell(cell) {
    if (cell.startsWith('"') && cell.endsWith('"')) {
        return cell.slice(1, -1).replace(/""/g, '"');
    }
    return cell;
}

function parseCsvLine(line) {
    const cells = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '"') {
            if (inQuotes && line[i + 1] === '"') {
                current += '"';
                i++;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c === ',' && !inQuotes) {
            cells.push(current);
            current = '';
        } else {
            current += c;
        }
    }
    cells.push(current);
    return cells.map(unescapeCsvCell);
}

function writeCsv(filePath, headers, rows) {
    const lines = [headers.join(',')];
    for (const row of rows) {
        lines.push(headers.map(h => escapeCsvCell(row[h])).join(','));
    }
    fs.writeFileSync(filePath, lines.join('\n') + '\n', 'utf8');
}

function readCsv(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.trim().split('\n');
    if (lines.length === 0) return { headers: [], rows: [] };
    const headers = parseCsvLine(lines[0]);
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
        const cells = parseCsvLine(lines[i]);
        const row = {};
        for (let j = 0; j < headers.length; j++) {
            row[headers[j]] = cells[j] ?? '';
        }
        rows.push(row);
    }
    return { headers, rows };
}

// ============================================================================
// C PARSER - Tokenizer and recursive descent for designated initializers
// ============================================================================

const TokenType = {
    IDENT: 'IDENT',
    NUMBER: 'NUMBER',
    STRING: 'STRING',
    LBRACE: '{',
    RBRACE: '}',
    LBRACKET: '[',
    RBRACKET: ']',
    EQUALS: '=',
    COMMA: ',',
    DOT: '.',
    PIPE: '|',
    EOF: 'EOF',
};

function tokenize(str) {
    const tokens = [];
    let i = 0;
    while (i < str.length) {
        const c = str[i];
        if (c === '{') { tokens.push({ type: TokenType.LBRACE, value: '{' }); i++; }
        else if (c === '}') { tokens.push({ type: TokenType.RBRACE, value: '}' }); i++; }
        else if (c === '[') { tokens.push({ type: TokenType.LBRACKET, value: '[' }); i++; }
        else if (c === ']') { tokens.push({ type: TokenType.RBRACKET, value: ']' }); i++; }
        else if (c === '=') { tokens.push({ type: TokenType.EQUALS, value: '=' }); i++; }
        else if (c === ',') { tokens.push({ type: TokenType.COMMA, value: ',' }); i++; }
        else if (c === '.') { tokens.push({ type: TokenType.DOT, value: '.' }); i++; }
        else if (c === '|') { tokens.push({ type: TokenType.PIPE, value: '|' }); i++; }
        else if (/\s/.test(c)) { i++; }
        else if (c === '"') {
            let val = '"';
            i++;
            while (i < str.length && str[i] !== '"') {
                val += str[i];
                i++;
            }
            if (i < str.length) { val += '"'; i++; }
            tokens.push({ type: TokenType.STRING, value: val });
        }
        else if (/[A-Za-z_]/.test(c)) {
            let val = '';
            while (i < str.length && /[A-Za-z0-9_]/.test(str[i])) {
                val += str[i];
                i++;
            }
            tokens.push({ type: TokenType.IDENT, value: val });
        }
        else if (/[0-9]/.test(c)) {
            let val = '';
            while (i < str.length && /[0-9]/.test(str[i])) {
                val += str[i];
                i++;
            }
            tokens.push({ type: TokenType.NUMBER, value: val });
        }
        else {
            i++;
        }
    }
    tokens.push({ type: TokenType.EOF, value: '' });
    return tokens;
}

class Parser {
    constructor(tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }

    peek() { return this.tokens[this.pos]; }
    consume(type) {
        const t = this.peek();
        if (t.type === type) { this.pos++; return t; }
        throw new Error(`Expected ${type}, got ${t.type} (${t.value}) at pos ${this.pos}`);
    }
    match(type) {
        if (this.peek().type === type) { this.pos++; return true; }
        return false;
    }

    parseValue() {
        const t = this.peek();
        if (t.type === TokenType.STRING) return this.consume(TokenType.STRING).value;
        if (t.type === TokenType.NUMBER) return this.consume(TokenType.NUMBER).value;
        if (t.type === TokenType.IDENT) {
            let ident = this.consume(TokenType.IDENT).value;
            while (this.match(TokenType.PIPE)) {
                ident += ' | ' + this.consume(TokenType.IDENT).value;
            }
            return ident;
        }
        if (t.type === TokenType.LBRACE) return this.parseBraceInit();
        throw new Error(`Unexpected token ${t.type} (${t.value}) at pos ${this.pos}`);
    }

    // Parse { ... } - can be struct init (.field = val) or array init (val1, val2)
    parseBraceInit() {
        this.consume(TokenType.LBRACE);
        const parts = [];
        while (!this.match(TokenType.RBRACE)) {
            const t = this.peek();
            if (t.type === TokenType.DOT) {
                // Designated initializer: .field = value
                const field = this.parseDesignatedField();
                parts.push(field);
            } else {
                // Regular value in array init
                const val = this.parseValue();
                parts.push(val);
            }
            this.match(TokenType.COMMA);
        }
        return '{' + parts.join(',') + '}';
    }

    // Parse .field = value
    parseDesignatedField() {
        this.consume(TokenType.DOT);
        const field = this.consume(TokenType.IDENT).value;
        let suffix = '';
        if (this.match(TokenType.LBRACKET)) {
            const idx = this.consume(TokenType.NUMBER).value;
            this.consume(TokenType.RBRACKET);
            suffix = `[${idx}]`;
        }
        this.consume(TokenType.EQUALS);
        const value = this.parseValue();
        this.match(TokenType.COMMA);
        return `.${field}${suffix} = ${value}`;
    }

    // Parse top-level entry: [KEY] = { ... }
    parseEntry() {
        this.consume(TokenType.LBRACKET);
        const key = this.consume(TokenType.IDENT).value;
        this.consume(TokenType.RBRACKET);
        this.consume(TokenType.EQUALS);
        const body = this.parseBraceInit();
        this.match(TokenType.COMMA);
        return { key, body };
    }

    parseArray() {
        const entries = [];
        while (!this.match(TokenType.EOF)) {
            if (this.peek().type === TokenType.LBRACKET) {
                entries.push(this.parseEntry());
            } else {
                this.pos++;
            }
        }
        return entries;
    }
}

// ============================================================================
// ENTRY PROCESSING - Convert between flat CSV and nested C
// ============================================================================

// Parse entry body (string like "{ .textData = { ... }, .speciesData = { ... } }")
// into nested object structure
function parseEntryBody(bodyStr) {
    const tokens = tokenize(bodyStr);
    const parser = new Parser(tokens);
    // Skip outer braces if present
    if (parser.peek().type === TokenType.LBRACE) {
        parser.consume(TokenType.LBRACE);
    }
    return parseBraceToObject(parser);
}

function parseBraceToObject(parser) {
    const result = {};
    let isPositionalArray = false;
    const positionalValues = [];

    while (!parser.match(TokenType.RBRACE)) {
        const t = parser.peek();
        if (t.type === TokenType.DOT) {
            // Designated initializer
            parser.consume(TokenType.DOT);
            const field = parser.consume(TokenType.IDENT).value;
            let suffix = '';
            if (parser.match(TokenType.LBRACKET)) {
                const idx = parser.consume(TokenType.NUMBER).value;
                parser.consume(TokenType.RBRACKET);
                suffix = `[${idx}]`;
            }
            parser.consume(TokenType.EQUALS);
            let value;
            if (parser.peek().type === TokenType.LBRACE) {
                parser.consume(TokenType.LBRACE);
                value = parseBraceToObject(parser);
            } else {
                const t2 = parser.peek();
                if (t2.type === TokenType.STRING) value = parser.consume(TokenType.STRING).value;
                else if (t2.type === TokenType.NUMBER) value = parser.consume(TokenType.NUMBER).value;
                else if (t2.type === TokenType.IDENT) {
                    let ident = parser.consume(TokenType.IDENT).value;
                    while (parser.match(TokenType.PIPE)) {
                        ident += ' | ' + parser.consume(TokenType.IDENT).value;
                    }
                    value = ident;
                }
            }
            parser.match(TokenType.COMMA);
            result[field + suffix] = value;
        } else if (t.type === TokenType.LBRACE) {
            // Nested brace - could be positional array entry or nested struct
            parser.consume(TokenType.LBRACE);
            const value = parseBraceToObject(parser);
            isPositionalArray = true;
            positionalValues.push(value);
            parser.match(TokenType.COMMA);
        } else if (t.type === TokenType.IDENT || t.type === TokenType.NUMBER || t.type === TokenType.STRING) {
            // Positional array value
            isPositionalArray = true;
            const value = parser.parseValue();
            positionalValues.push(value);
            parser.match(TokenType.COMMA);
        } else {
            parser.pos++;
        }
    }

    if (isPositionalArray && Object.keys(result).length === 0) {
        return positionalValues;
    }
    return result;
}

function entryToFlat(entry, config) {
    const obj = parseEntryBody(entry.body);
    const flat = { [config.keyColumn]: entry.key };

    // Special handling for Evolutions.c: entries is a positional array of [method, param, target]
    if (config.fieldMapping && config.fieldMapping.evo1_method === 'entries[0].method') {
        const entriesArr = obj.entries;
        if (Array.isArray(entriesArr)) {
            for (let i = 0; i < 10; i++) {
                const entry = entriesArr[i] || [];
                flat[`evo${i + 1}_method`] = entry[0] || '';
                flat[`evo${i + 1}_param`] = entry[1] || '';
                flat[`evo${i + 1}_target`] = entry[2] || '';
            }
        }
        return flat;
    }

    for (const [csvField, cPath] of Object.entries(config.fieldMapping)) {
        // Check if this is an array element path: entries[0].method or speciesData.types[0]
        const m = cPath.match(/^(.+)\[(\d+)\](.*)$/);
        if (m) {
            const basePath = m[1];
            const index = parseInt(m[2]);
            const subPath = m[3] ? m[3].replace(/^\./, '') : '';
            const arr = getArrayValues(obj, basePath);
            if (arr && index < arr.length) {
                if (subPath) {
                    flat[csvField] = arr[index][subPath] || '';
                } else {
                    flat[csvField] = arr[index] || '';
                }
            } else {
                flat[csvField] = '';
            }
        } else {
            flat[csvField] = getNested(obj, cPath) || '';
        }
    }
    return flat;
}

function getNested(obj, path) {
    const parts = path.split('.').map(p => {
        const m = p.match(/^(\w+)\[(\d+)\]$/);
        if (m) return { key: m[1], index: parseInt(m[2]) };
        return { key: p };
    });
    let cur = obj;
    for (const part of parts) {
        if (cur === undefined || cur === null) return null;
        if (part.index !== undefined) {
            cur = cur[part.key];
            if (!Array.isArray(cur)) return null;
            cur = cur[part.index];
        } else {
            cur = cur[part.key];
        }
    }
    return cur;
}

// Get all values for an array field (for CSV export)
function getArrayValues(obj, path) {
    const basePath = path.replace(/\[\d+\]$/, '');
    const arr = getNested(obj, basePath);
    if (!Array.isArray(arr)) return null;
    return arr;
}

function setNested(obj, path, value) {
    const parts = path.split('.').map(p => {
        const m = p.match(/^(\w+)\[(\d+)\]$/);
        if (m) return { key: m[1], index: parseInt(m[2]) };
        return { key: p };
    });
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) {
        const part = parts[i];
        if (part.index !== undefined) {
            if (!cur[part.key]) cur[part.key] = [];
            if (!cur[part.key][part.index]) cur[part.key][part.index] = {};
            cur = cur[part.key][part.index];
        } else {
            if (!cur[part.key]) cur[part.key] = {};
            cur = cur[part.key];
        }
    }
    const last = parts[parts.length - 1];
    if (last.index !== undefined) {
        if (!cur[last.key]) cur[last.key] = [];
        cur[last.key][last.index] = value;
    } else {
        cur[last.key] = value;
    }
}

function flatToEntry(flat, config) {
    const obj = {};
    // Collect array element fields (paths containing [N])
    const arraySubFields = {}; // basePath -> index -> { subFieldName -> val }

    for (const [csvField, cPath] of Object.entries(config.fieldMapping)) {
        const val = flat[csvField];
        if (val === undefined) continue;

        // Check if this is an array element path: entries[0].method or speciesData.types[0]
        const m = cPath.match(/^(.+)\[(\d+)\](.*)$/);
        if (m) {
            const basePath = m[1];
            const index = parseInt(m[2]);
            const subField = m[3] ? m[3].replace(/^\./, '') : '';
            if (!arraySubFields[basePath]) arraySubFields[basePath] = {};
            if (!arraySubFields[basePath][index]) arraySubFields[basePath][index] = {};
            if (subField) {
                arraySubFields[basePath][index][subField] = val;
            } else {
                // Direct array element like speciesData.types[0]
                arraySubFields[basePath][index][''] = val;
            }
        } else {
            // Simple non-array field
            setNested(obj, cPath, val);
        }
    }

    // Apply array fields
    for (const [basePath, indices] of Object.entries(arraySubFields)) {
        const arr = [];
        for (const [index, subFields] of Object.entries(indices)) {
            if (subFields[''] !== undefined) {
                // Direct value array element
                arr[parseInt(index)] = subFields[''];
            } else if (basePath === 'entries' && subFields.method !== undefined) {
                // Evolution entry as positional array [method, param, target]
                arr[parseInt(index)] = [
                    subFields.method || 'EVO_NONE',
                    subFields.param || '0',
                    subFields.target || 'SPECIES_NONE'
                ];
            } else {
                arr[parseInt(index)] = subFields;
            }
        }
        setNested(obj, basePath, arr);
    }

    const body = objectToCBody(obj, config);
    return { key: flat[config.keyColumn], body };
}

function isArrayField(path) {
    return /\[\d+\]$/.test(path);
}

function getArrayBasePath(path) {
    return path.replace(/\[\d+\]$/, '');
}

function getArrayIndex(path) {
    const m = path.match(/\[(\d+)\]$/);
    return m ? parseInt(m[1]) : null;
}

// ============================================================================
// C OUTPUT - Build C array text from flat CSV rows
// ============================================================================

function objectToCBody(obj, config) {
    const parts = [];

    // First, find all top-level structs in fieldMapping
    // Skip paths like entries[0] - those are positional array elements
    const topLevelStructs = new Set();
    for (const cPath of Object.values(config.fieldMapping)) {
        const top = cPath.split('.')[0];
        if (top.match(/\[\d+\]$/)) continue;
        topLevelStructs.add(top);
    }

    // For each top-level struct, output its fields
    for (const topLevel of topLevelStructs) {
        // Skip positional arrays at top level - handled separately
        if (config.positionalArrays && config.positionalArrays.has(topLevel)) {
            continue;
        }

        const structStr = formatStruct(obj, config, topLevel);
        if (structStr) {
            parts.push(`.${topLevel} = { ${structStr} }`);
        }
    }

    // Handle arrays that are direct children of root (Evolutions.c entries)
    if (config.positionalArrays) {
        for (const arrayName of config.positionalArrays) {
            // Skip if formatStruct already output this via isArrayField paths
            let alreadyOutput = false;
            for (const cPath of Object.values(config.fieldMapping)) {
                if (cPath.startsWith(arrayName + '[') && isArrayField(cPath)) {
                    alreadyOutput = true;
                    break;
                }
            }
            if (alreadyOutput) continue;

            const arr = getNested(obj, arrayName);
            if (Array.isArray(arr)) {
                const entryStrs = [];
                for (const entry of arr) {
                    if (Array.isArray(entry)) {
                        entryStrs.push(`{ ${entry.join(', ')} }`);
                    } else if (entry && typeof entry === 'object') {
                        const entryParts = [];
                        for (const [k, v] of Object.entries(entry)) {
                            if (v !== undefined && v !== null) entryParts.push(`${v}`);
                            else entryParts.push('0');
                        }
                        entryStrs.push(`{ ${entryParts.join(', ')} }`);
                    } else {
                        entryStrs.push('{ 0, 0, 0 }');
                    }
                }
                parts.push(`.${arrayName} = { ${entryStrs.join(', ')} }`);
            }
        }
    }

    return parts.join(', ');
}

function formatStruct(obj, config, path) {
    // If path itself is a positional array, skip - objectToCBody handles it
    if (config.positionalArrays && config.positionalArrays.has(path)) {
        return '';
    }

    const parts = [];
    const processedKeys = new Set();

    // Get all fields under this path
    for (const [csvField, cPath] of Object.entries(config.fieldMapping)) {
        if (!cPath.startsWith(path + '.')) continue;

        const relPath = cPath.substring(path.length + 1);
        const pathParts = relPath.split('.');
        const firstPart = pathParts[0];

        if (processedKeys.has(firstPart)) continue;

        // Skip positional arrays - let objectToCBody handle them
        if (config.positionalArrays && config.positionalArrays.has(firstPart)) {
            processedKeys.add(firstPart);
            continue;
        }

        // Also skip if this firstPart looks like entries[0] (positional array element)
        if (/\[\d+\]$/.test(firstPart)) {
            processedKeys.add(firstPart);
            continue;
        }

        if (isArrayField(cPath)) {
            // Check if this is the base array
            const basePath = getArrayBasePath(cPath);
            if (basePath === cPath.replace(/\[\d+\]$/, '')) {
                if (!processedKeys.has(basePath.split('.').pop())) {
                    processedKeys.add(basePath.split('.').pop());
                    const arr = getArrayValues(obj, basePath);
                    if (arr) {
                        // Output as positional array entries
                        const entryStrs = [];
                        for (const entry of arr) {
                            if (Array.isArray(entry)) {
                                entryStrs.push(`{ ${entry.join(', ')} }`);
                            } else if (entry && typeof entry === 'object') {
                                const entryParts = [];
                                for (const [k, v] of Object.entries(entry)) {
                                    if (v !== undefined && v !== null) entryParts.push(`${v}`);
                                    else entryParts.push('0');
                                }
                                entryStrs.push(`{ ${entryParts.join(', ')} }`);
                            } else {
                                entryStrs.push('{ 0, 0, 0 }');
                            }
                        }
                        parts.push(`.${basePath.split('.').pop()} = { ${entryStrs.join(', ')} }`);
                    }
                }
            }
            continue;
        }

        if (pathParts.length === 1) {
            // Direct field
            const val = getNested(obj, cPath);
            if (val !== undefined && val !== null) {
                let value = val;
                if (config.stringFields && config.stringFields.has(firstPart)) {
                    if (!value.startsWith('"')) value = '"' + value + '"';
                }
                parts.push(`.${firstPart} = ${value}`);
                processedKeys.add(firstPart);
            }
        } else {
            // Nested struct - recursively format
            const nestedPath = path + '.' + firstPart;
            if (!processedKeys.has(firstPart)) {
                processedKeys.add(firstPart);
                const nestedStr = formatStruct(obj, config, nestedPath);
                if (nestedStr) {
                    parts.push(`.${firstPart} = { ${nestedStr} }`);
                }
            }
        }
    }

    return parts.join(', ');
}

// ============================================================================
// C FILE READ/WRITE - Preserve headers/footers
// ============================================================================

function splitCFile(content) {
    const arrayStart = content.indexOf('const ');
    if (arrayStart === -1) throw new Error('No array declaration found');
    let bracePos = content.indexOf('{', arrayStart);
    if (bracePos === -1) throw new Error('No array opening brace found');
    const header = content.slice(0, bracePos + 1);
    let depth = 1;
    let i = bracePos + 1;
    for (; i < content.length && depth > 0; i++) {
        if (content[i] === '{') depth++;
        else if (content[i] === '}') depth--;
    }
    const body = content.slice(bracePos + 1, i - 1);
    const footer = content.slice(i - 1);
    return { header, body, footer };
}

function parseCArray(content, config) {
    const { header, body, footer } = splitCFile(content);
    const tokens = tokenize(body);
    const parser = new Parser(tokens);
    const entries = parser.parseArray();
    const rows = entries.map(e => entryToFlat(e, config));
    return { header, footer, rows };
}

function writeCArray(header, footer, rows, config) {
    const lines = [''];
    for (const row of rows) {
        const entry = flatToEntry(row, config);
        lines.push(`    [${entry.key}] = { ${entry.body} },`);
    }
    lines.push('');
    return header + '\n' + lines.join('\n') + '\n' + footer;
}

// ============================================================================
// FILE-SPECIFIC CONFIGURATIONS
// ============================================================================

const FILE_CONFIGS = {
    'Species.c': {
        filePath: 'data/Species.c',
        arrayName: 'sSpeciesData',
        keyColumn: 'species',
        fieldOrder: [
            'name', 'pokedexEntry', 'classification', 'height', 'weight',
            'hp', 'attack', 'defense', 'spAttack', 'spDefense', 'speed',
            'type1', 'type2', 'catchRate', 'baseExpRewardPadding',
            'evHp', 'evAttack', 'evDefense', 'evSpAttack', 'evSpDefense', 'evSpeed',
            'heldItemCommon', 'heldItemRare',
            'genderRatio', 'hatchCycles', 'baseFriendship', 'expRate',
            'eggGroup1', 'eggGroup2',
            'ability1', 'ability2',
            'safariFleeRate', 'bodyColor', 'flipSprite',
            'heightDecimetres', 'weightHectograms', 'bodyType',
            'fTrainerScale', 'fPokemonScale', 'mTrainerScale', 'mPokemonScale',
            'fTrainerYOffset', 'fPokemonYOffset', 'mTrainerYOffset', 'mPokemonYOffset',
        ],
        fieldMapping: {
            'name': 'textData.name',
            'pokedexEntry': 'textData.pokedexEntry',
            'classification': 'textData.classification',
            'height': 'textData.height',
            'weight': 'textData.weight',
            'hp': 'speciesData.baseStats.hp',
            'attack': 'speciesData.baseStats.attack',
            'defense': 'speciesData.baseStats.defense',
            'spAttack': 'speciesData.baseStats.spAttack',
            'spDefense': 'speciesData.baseStats.spDefense',
            'speed': 'speciesData.baseStats.speed',
            'type1': 'speciesData.types[0]',
            'type2': 'speciesData.types[1]',
            'catchRate': 'speciesData.catchRate',
            'baseExpRewardPadding': 'speciesData.baseExpRewardPadding',
            'evHp': 'speciesData.evYields.hp',
            'evAttack': 'speciesData.evYields.attack',
            'evDefense': 'speciesData.evYields.defense',
            'evSpAttack': 'speciesData.evYields.spAttack',
            'evSpDefense': 'speciesData.evYields.spDefense',
            'evSpeed': 'speciesData.evYields.speed',
            'heldItemCommon': 'speciesData.wildHeldItems.common',
            'heldItemRare': 'speciesData.wildHeldItems.rare',
            'genderRatio': 'speciesData.genderRatio',
            'hatchCycles': 'speciesData.hatchCycles',
            'baseFriendship': 'speciesData.baseFriendship',
            'expRate': 'speciesData.expRate',
            'eggGroup1': 'speciesData.eggGroups[0]',
            'eggGroup2': 'speciesData.eggGroups[1]',
            'ability1': 'speciesData.abilities[0]',
            'ability2': 'speciesData.abilities[1]',
            'safariFleeRate': 'speciesData.safariFleeRate',
            'bodyColor': 'speciesData.bodyColor',
            'flipSprite': 'speciesData.flipSprite',
            'heightDecimetres': 'metricsData.heightDecimetres',
            'weightHectometers': 'metricsData.weightHectometers',
            'bodyType': 'metricsData.bodyType',
            'fTrainerScale': 'metricsData.femaleTrainerScale',
            'fPokemonScale': 'metricsData.femalePokemonScale',
            'mTrainerScale': 'metricsData.maleTrainerScale',
            'mPokemonScale': 'metricsData.malePokemonScale',
            'fTrainerYOffset': 'metricsData.femaleTrainerYOffset',
            'fPokemonYOffset': 'metricsData.femalePokemonYOffset',
            'mTrainerYOffset': 'metricsData.maleTrainerYOffset',
            'mPokemonYOffset': 'metricsData.malePokemonYOffset',
        },
        stringFields: new Set([
            'name', 'pokedexEntry', 'classification', 'height', 'weight',
        ]),
    },
    'Moves.c': {
        filePath: 'data/Moves.c',
        arrayName: 'sMoveSource',
        keyColumn: 'move',
        fieldOrder: [
            'name', 'capsName', 'fullName', 'description',
            'effect', 'split', 'power', 'type', 'accuracy', 'pp', 'effectChance',
            'target', 'priority', 'flags',
            'appeal', 'contestType',
        ],
        fieldMapping: {
            'name': 'names.name',
            'capsName': 'names.capsName',
            'fullName': 'names.fullName',
            'effect': 'data.effect',
            'split': 'data.split',
            'power': 'data.power',
            'type': 'data.type',
            'accuracy': 'data.accuracy',
            'pp': 'data.pp',
            'effectChance': 'data.effectChance',
            'target': 'battle.target',
            'priority': 'battle.priority',
            'flags': 'battle.flags',
            'appeal': 'contest.appeal',
            'contestType': 'contest.contestType',
            'description': 'description',
        },
        stringFields: new Set([
            'name', 'capsName', 'fullName', 'description',
        ]),
    },
    'Evolutions.c': {
        filePath: 'data/Evolutions.c',
        arrayName: '__data',
        keyColumn: 'species',
        fieldOrder: (() => {
            const order = [];
            for (let i = 1; i <= 10; i++) {
                order.push(`evo${i}_method`, `evo${i}_param`, `evo${i}_target`);
            }
            return order;
        })(),
        fieldMapping: (() => {
            const map = {};
            for (let i = 1; i <= 10; i++) {
                map[`evo${i}_method`] = `entries[${i - 1}].method`;
                map[`evo${i}_param`] = `entries[${i - 1}].param`;
                map[`evo${i}_target`] = `entries[${i - 1}].target`;
            }
            return map;
        })(),
        stringFields: new Set(),
        // entries is a positional array of [method, param, target]
        positionalArrays: new Set(['entries']),
    },
};

// ============================================================================
// HIGH-LEVEL OPERATIONS
// ============================================================================

function exportFile(config, outPath) {
    const content = fs.readFileSync(config.filePath, 'utf8');
    const { rows } = parseCArray(content, config);
    writeCsv(outPath, [config.keyColumn, ...config.fieldOrder], rows);
    console.log(`Exported ${rows.length} entries to ${outPath}`);
}

function importFile(config, inPath) {
    const { headers, rows } = readCsv(inPath);
    if (!headers.includes(config.keyColumn)) {
        throw new Error(`CSV missing key column: ${config.keyColumn}`);
    }
    const content = fs.readFileSync(config.filePath, 'utf8');
    const { header, footer } = splitCFile(content);
    const newContent = writeCArray(header, footer, rows, config);
    fs.writeFileSync(config.filePath, newContent, 'utf8');
    console.log(`Imported ${rows.length} entries from ${inPath} to ${config.filePath}`);
}

function selftest() {
    console.log('Running selftest (round-trip)...');
    const tmpDir = fs.mkdtempSync(path.join('/tmp', 'data_csv_'));
    let allPassed = true;
    for (const [name, config] of Object.entries(FILE_CONFIGS)) {
        const tmpCsv = path.join(tmpDir, `${name}.csv`);
        try {
            const origContent = fs.readFileSync(config.filePath, 'utf8');
            exportFile(config, tmpCsv);
            const { header, footer } = splitCFile(origContent);
            const { rows } = readCsv(tmpCsv);
            const newContent = writeCArray(header, footer, rows, config);
            const norm = s => s.replace(/\s+/g, ' ').trim();
            if (norm(origContent) !== norm(newContent)) {
                console.error(`FAIL: ${name} round-trip mismatch`);
                const o = norm(origContent);
                const n = norm(newContent);
                for (let i = 0; i < Math.min(o.length, n.length); i++) {
                    if (o[i] !== n[i]) {
                        console.error(`  Diff at ${i}: orig="${o.slice(i, i+80)}" new="${n.slice(i, i+80)}"`);
                        break;
                    }
                }
                allPassed = false;
            } else {
                console.log(`PASS: ${name}`);
            }
        } catch (e) {
            console.error(`FAIL: ${name} - ${e.message}`);
            console.error(e.stack);
            allPassed = false;
        }
    }
    fs.rmSync(tmpDir, { recursive: true, force: true });
    if (allPassed) {
        console.log('All selftests passed!');
        process.exit(0);
    } else {
        console.error('Some selftests failed!');
        process.exit(1);
    }
}

// ============================================================================
// CLI
// ============================================================================

function main() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.log(`Usage:
  node data_csv.js export <Species.c|Moves.c|Evolutions.c> <out.csv>
  node data_csv.js import <Species.c|Moves.c|Evolutions.c> <in.csv>
  node data_csv.js selftest`);
        process.exit(1);
    }

    const cmd = args[0];
    if (cmd === 'selftest') {
        selftest();
        return;
    }

    if (args.length !== 3) {
        console.error('Expected 3 arguments: command file csv');
        process.exit(1);
    }

    const fileArg = args[1];
    const csvPath = args[2];

    let config;
    if (fileArg === 'Species.c' || fileArg === 'data/Species.c') config = FILE_CONFIGS['Species.c'];
    else if (fileArg === 'Moves.c' || fileArg === 'data/Moves.c') config = FILE_CONFIGS['Moves.c'];
    else if (fileArg === 'Evolutions.c' || fileArg === 'data/Evolutions.c') config = FILE_CONFIGS['Evolutions.c'];
    else {
        console.error(`Unknown file: ${fileArg}`);
        process.exit(1);
    }

    if (cmd === 'export') {
        exportFile(config, csvPath);
    } else if (cmd === 'import') {
        importFile(config, csvPath);
    } else {
        console.error(`Unknown command: ${cmd}`);
        process.exit(1);
    }
}

main();