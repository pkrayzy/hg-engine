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
        else if (c === '(') {
            let val = '(';
            let depth = 1;
            i++;
            while (i < str.length && depth > 0) {
                val += str[i];
                if (str[i] === '(') depth++;
                else if (str[i] === ')') depth--;
                i++;
            }
            tokens.push({ type: TokenType.STRING, value: val });
        }
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
            // Function-like macro: IDENT(...) — keep as one token
            if (i < str.length && str[i] === '(') {
                let depth = 1;
                val += '(';
                i++;
                while (i < str.length && depth > 0) {
                    val += str[i];
                    if (str[i] === '(') depth++;
                    else if (str[i] === ')') depth--;
                    i++;
                }
            }
            tokens.push({ type: TokenType.IDENT, value: val });
        }
        else if (c === '0' && str[i + 1] === 'x') {
            let val = '0x';
            i += 2;
            while (i < str.length && /[0-9a-fA-F]/.test(str[i])) {
                val += str[i];
                i++;
            }
            tokens.push({ type: TokenType.NUMBER, value: val });
        }
        else if (/[0-9]/.test(c)) {
            let val = '';
            while (i < str.length && /[0-9]/.test(str[i])) {
                val += str[i];
                i++;
            }
            tokens.push({ type: TokenType.NUMBER, value: val });
        }
        else if (c === '-' && /[0-9]/.test(str[i + 1])) {
            let val = '-';
            i++;
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

    parseBraceInit() {
        this.consume(TokenType.LBRACE);
        const parts = [];
        while (!this.match(TokenType.RBRACE)) {
            const t = this.peek();
            if (t.type === TokenType.DOT) {
                const field = this.parseDesignatedField();
                parts.push(field);
            } else {
                const val = this.parseValue();
                parts.push(val);
            }
            this.match(TokenType.COMMA);
        }
        return '{' + parts.join(',') + '}';
    }

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
// ENTRY PROCESSING
// ============================================================================

function parseEntryBody(bodyStr) {
    const tokens = tokenize(bodyStr);
    const parser = new Parser(tokens);
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
            parser.consume(TokenType.LBRACE);
            const value = parseBraceToObject(parser);
            isPositionalArray = true;
            positionalValues.push(value);
            parser.match(TokenType.COMMA);
        } else if (t.type === TokenType.IDENT || t.type === TokenType.NUMBER || t.type === TokenType.STRING) {
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

    if (config.fieldMapping && config.fieldMapping.evo1_method === 'entries[0].method') {
        const entriesArr = obj.entries;
        if (Array.isArray(entriesArr)) {
            for (let i = 0; i < 10; i++) {
                const entry = entriesArr[i];
                if (!entry) continue;
                flat[`evo${i + 1}_method`] = entry[0] || '';
                flat[`evo${i + 1}_param`] = entry[1] || '';
                flat[`evo${i + 1}_target`] = entry[2] || '';
            }
        }
        return { ...flat, obj, key: flat[config.keyColumn] };
    }

    for (const [csvField, cPath] of Object.entries(config.fieldMapping)) {
        if (csvField === config.keyColumn) continue;

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
    return { ...flat, obj, key: flat[config.keyColumn] };
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
    const key = flat[config.keyColumn];
    const arraySubFields = {};

    for (const [csvField, cPath] of Object.entries(config.fieldMapping)) {
        const val = flat[csvField];
        if (val === undefined || csvField === config.keyColumn) continue;

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
                arraySubFields[basePath][index][''] = val;
            }
        } else {
            setNested(obj, cPath, val);
        }
    }

    for (const [basePath, indices] of Object.entries(arraySubFields)) {
        const arr = [];
        for (const [index, subFields] of Object.entries(indices)) {
            if (subFields[''] !== undefined) {
                arr[parseInt(index)] = subFields[''];
            } else if (basePath === 'entries') {
                if (subFields.method === undefined || subFields.method === '') continue;
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
    return { obj, key };
}

// ============================================================================
// C OUTPUT - Pretty-printed C array text
// ============================================================================

const INDENT4 = '    ';

function isArrayField(path) {
    return /\[\d+\]$/.test(path);
}

function getArrayBasePath(path) {
    return path.replace(/\[\d+\]$/, '');
}

function objectToCBody(obj, config) {
    const topLevelStructs = [];
    for (const cPath of Object.values(config.fieldMapping)) {
        const top = cPath.split('.')[0];
        if (top.match(/\[\d+\]$/)) continue;
        if (!topLevelStructs.includes(top)) topLevelStructs.push(top);
    }

    const structIndent = INDENT4 + INDENT4; // 8 spaces: fields inside entry

    const structSections = [];
    for (const topLevel of topLevelStructs) {
        if (config.positionalArrays && config.positionalArrays.has(topLevel)) {
            continue;
        }

        const structBody = formatStruct(obj, config, topLevel, structIndent);
        if (structBody) {
            structSections.push(`        .${topLevel} = {\n${structBody}\n        }`);
        }
    }

    // Top-level scalar fields (direct entry fields, not inside a sub-struct)
    const topLevelScalars = [];
    for (const [csvField, cPath] of Object.entries(config.fieldMapping)) {
        if (cPath.includes('.')) continue;
        const val = getNested(obj, cPath);
        if (val !== undefined && val !== null) {
            let value = val;
            if (config.stringFields && config.stringFields.has(csvField)) {
                if (!value.startsWith('"')) value = '"' + value + '"';
            }
            topLevelScalars.push(`        .${csvField} = ${value}`);
        }
    }

    const bodyParts = [...structSections, ...topLevelScalars].map(p => p + ',');

    const entryLines = [`    [__FLAT_KEY__] = {`];
    if (bodyParts.length > 0) {
        entryLines.push(bodyParts.join('\n'));
    }
    entryLines.push('    },');

    if (config.positionalArrays) {
        for (const arrayName of config.positionalArrays) {
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
                const arrayIndent = INDENT4 + INDENT4 + INDENT4; // 12 spaces inside .entries
                const entriesText = entryStrs.join(',\n' + arrayIndent);
                entryLines.splice(entryLines.length - 1, 0, `        .${arrayName} = {\n${arrayIndent}${entriesText}${entriesText ? ',\n' : '\n'}        },`);
            }
        }
    }

    return entryLines.join('\n');
}

const FLAT_KEY_PLACEHOLDER = '__FLAT_KEY__';

function writeCArray(header, footer, rows, config) {
    const lines = [''];
    for (const row of rows) {
        let result;
        if (row.obj !== undefined) {
            result = row;
        } else {
            result = flatToEntry(row, config);
        }
        const body = objectToCBody(result.obj, config);
        const entryText = body.replace(FLAT_KEY_PLACEHOLDER, result.key || '<EMPTY_KEY>');
        lines.push(entryText);
    }
    lines.push('');
    return header + '\n' + lines.join('\n') + '\n' + footer;
}

function formatStruct(obj, config, path, indent) {
    if (config.positionalArrays && config.positionalArrays.has(path)) {
        return '';
    }

    const parts = [];
    const processedKeys = new Set();
    const childIndent = indent + '    ';

    for (const [csvField, cPath] of Object.entries(config.fieldMapping)) {
        if (!cPath.startsWith(path + '.')) continue;

        const relPath = cPath.substring(path.length + 1);
        const pathParts = relPath.split('.');
        const firstPart = pathParts[0];

        if (processedKeys.has(firstPart)) continue;

        if (config.positionalArrays && config.positionalArrays.has(firstPart)) {
            processedKeys.add(firstPart);
            continue;
        }

        if (config.positionalArrays && config.positionalArrays.some(arrName => cPath.startsWith(arrName + '['))) {
            processedKeys.add(firstPart);
            continue;
        }

        if (isArrayField(cPath)) {
            const basePath = getArrayBasePath(cPath);
            if (basePath === cPath.replace(/\[\d+\]$/, '')) {
                if (!processedKeys.has(basePath.split('.').pop())) {
                    processedKeys.add(basePath.split('.').pop());
                    const arr = getArrayValues(obj, basePath);
                    if (arr) {
                        const entryStrs = [];
                        for (const entry of arr) {
                            if (typeof entry === 'string') {
                                entryStrs.push(entry);
                            } else if (Array.isArray(entry)) {
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
                        parts.push(`${childIndent}.${basePath.split('.').pop()} = { ${entryStrs.join(', ')} },`);
                    }
                }
            }
            continue;
        }

        if (pathParts.length === 1) {
            const val = getNested(obj, cPath);
            if (val !== undefined && val !== null) {
                let value = val;
                if (config.stringFields && config.stringFields.has(firstPart)) {
                    if (!value.startsWith('"')) value = '"' + value + '"';
                }
                parts.push(`${childIndent}.${firstPart} = ${value},`);
                processedKeys.add(firstPart);
            }
        } else {
            const nestedPath = path + '.' + firstPart;
            if (!processedKeys.has(firstPart)) {
                processedKeys.add(firstPart);
                const nestedStr = formatStruct(obj, config, nestedPath, childIndent);
                if (nestedStr) {
                    parts.push(`${childIndent}.${firstPart} = {\n${nestedStr}\n${childIndent}},`);
                }
            }
        }
    }

    return parts.join('\n');
}

// ============================================================================
// C FILE READ/WRITE
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
            'weightHectograms': 'metricsData.weightHectograms',
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
        positionalArrays: new Set(['entries']),
    },
};

// ============================================================================
// HIGH-LEVEL OPERATIONS
// ============================================================================

function exportFile(config, outPath) {
    const content = fs.readFileSync(config.filePath, 'utf8');
    const { rows } = parseCArray(content, config);
    const headers = [config.keyColumn, ...config.fieldOrder];
    writeCsv(outPath, headers, rows);
    console.log(`Exported ${rows.length} entries to ${outPath}`);
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
                        console.error(`  Diff at ${i}: orig="${o.slice(i, i+100)}" new="${n.slice(i, i+100)}"`);
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

function getConfig(fileArg) {
    if (fileArg === 'Species.c' || fileArg === 'data/Species.c') return FILE_CONFIGS['Species.c'];
    if (fileArg === 'Moves.c' || fileArg === 'data/Moves.c') return FILE_CONFIGS['Moves.c'];
    if (fileArg === 'Evolutions.c' || fileArg === 'data/Evolutions.c') return FILE_CONFIGS['Evolutions.c'];
    return null;
}

function csvNameFor(config) {
    return path.basename(config.filePath, '.c') + '.csv';
}

function main() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.log(`Usage:
  node data_csv.js export <Species.c|Moves.c|Evolutions.c> <out.csv>
  node data_csv.js import <Species.c|Moves.c|Evolutions.c> <in.csv>
  node data_csv.js export       (exports all three to the root directory)
  node data_csv.js import       (imports all three from the root directory)
  node data_csv.js selftest`);
        process.exit(1);
    }

    const cmd = args[0];

    if (cmd === 'selftest') {
        selftest();
        return;
    }

    if (cmd === 'export' && args.length === 1) {
        for (const config of Object.values(FILE_CONFIGS)) {
            exportFile(config, csvNameFor(config));
        }
        return;
    }

    if (cmd === 'import' && args.length === 1) {
        for (const config of Object.values(FILE_CONFIGS)) {
            importFile(config, csvNameFor(config));
        }
        return;
    }

    if (args.length !== 3) {
        console.error('Expected 3 arguments: command file csv');
        process.exit(1);
    }

    const fileArg = args[1];
    const csvPath = args[2];

    const config = getConfig(fileArg);
    if (!config) {
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