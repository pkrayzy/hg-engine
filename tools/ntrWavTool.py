"""
Written by Turtleisaac - https://github.com/turtleisaac
Created through the assistance of RoadrunnerWMC and based on their strm-xq.py
Usages should be credited to Turtleisaac
Requires compiling adpcm-xq and having it in the same directory as this file
"""

import argparse
import io
import math
import os
from pathlib import Path
import subprocess
from typing import List, Optional
import wave
import ndspy
import ndspy.soundWave

DEFAULT_LOOKAHEAD = 3  # same as adpcm-xq itself


def split_stereo_wav(in_wav: Path, left_out_wav: Path, right_out_wav: Path) -> None:
    """
    Use ffmpeg to split a stereo wav to two mono wavs
    """
    subprocess.run([
        'ffmpeg',
        '-y',
        '-i', str(in_wav),
        '-map_channel', '0.0.0', str(left_out_wav),
        '-map_channel', '0.0.1', str(right_out_wav)])


def do_wav_mangling(in_wav: Path, out_wav: Path, block_size_samples: int) -> None:
    """
    Split the wav data into blocks of specified size, then double the
    first sample of each block. This tricks adpcm-xq into creating ADPCM
    data in the format expected by the Nintendo DS (i.e. the block
    header does *not* count as a sample)
    """

    with wave.open(str(in_wav), 'r') as in_wav:
        if in_wav.getnchannels() != 1:
            raise ValueError('can only mangle mono wavs')

        sampwidth = in_wav.getsampwidth()

        with wave.open(str(out_wav), 'w') as out_wav:
            out_wav.setparams(in_wav.getparams())
            block = True
            while block:
                block = in_wav.readframes(block_size_samples)
                out_wav.writeframes(block[:sampwidth])
                out_wav.writeframes(block)


def run_adpcm_xq(in_wav: Path, out_wav: Path, adpcm_xq: Path,
                 *, lookahead: Optional[int] = None, block_size_pow: Optional[int] = None) -> None:
    """
    Helper function to run adpcm-xq
    """
    cmd = [os.getcwd() + os.path.sep + str(adpcm_xq)]
    cmd.append('-e')  # "encode only (fail on WAV file already ADPCM)"
    cmd.append('-q')  # "quiet mode (display errors only)"
    cmd.append('-y')  # "overwrite outfile if it exists"

    if lookahead:
        cmd.append(f'-{lookahead}')

    if block_size_pow:
        cmd.append(f'-b{block_size_pow}')

    cmd += [str(in_wav), str(out_wav)]

    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = str(result.stdout).replace('b\'\'', '').strip()
    if len(output) > 3:
        print(output)


def create_swav_from_adpcm_wavs(wav_path: str, *, loop_start: Optional[int] = None) -> ndspy.soundWave:
    """
    Build a SWAV from a path to an IMA-ADPCM WAV
    """
    data = b''
    with Path(wav_path).open('rb') as f:
        f.seek(0, io.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        assert f.read(4) == b'RIFF'

        f.seek(0x14)
        sample_format = int.from_bytes(f.read(2), 'little')
        if sample_format != 17:
            raise ValueError(f'Expecting 17 (ADPCM) for WAV type, but found {sample_format}')

        num_channels = int.from_bytes(f.read(2), 'little')
        if num_channels != 1:
            raise ValueError(f'Expecting mono WAV, got {num_channels} channels')

        sample_rate = int.from_bytes(f.read(4), 'little')

        f.seek(0x20)
        block_size = int.from_bytes(f.read(2), 'little')
        last_block_size = block_size

        f.seek(0x28)
        assert f.read(4) == b'fact'

        f.seek(0x34)
        assert f.read(4) == b'data'
        total_data_bytes = int.from_bytes(f.read(4), 'little')

        while f.tell() < file_size:
            block = f.read(block_size)
            data = block

    swav = ndspy.soundWave.SWAV.fromData(data)
    swav.waveType = ndspy.WaveType.ADPCM
    swav.sampleRate = sample_rate
    swav.isLooped = (loop_start is not None)
    swav.loopOffset = 0 if loop_start is None else loop_start
    return swav


def main(argv: List[str] = None) -> None:
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description='swav-xq: WAV -> SWAV converter powered by ndspy and adpcm-xq')

    parser.add_argument('wav', type=Path,
                        help='input wav')
    parser.add_argument('swav', nargs='?', type=Path,
                        help='output swav')
    parser.add_argument('block_size', type=lambda val: int(val, 0),
                        help='adpcm block size (must be a power of 2)')
    parser.add_argument('--temp-file-dir', type=Path,
                        help='location of dir to put temp files generated during conversion process')
    parser.add_argument('--adpcm-xq', type=Path,
                        help='location of the adpcm-xq executable')
    parser.add_argument('--lookahead', type=int, metavar='N', default=DEFAULT_LOOKAHEAD,
                        help='adpcm-xq lookahead value (larger values make encoding exponentially slower)')
    parser.add_argument('--skip-conversion', action='store_true',
                        help='assume that [swavname]_[blocksize]_[lookahead].wav already exists, and skip creating it')
    parser.add_argument('--loop-start', type=int,
                        help='loop start point (measured in samples)')
    parser.add_argument('--shadow-buffer-size', type=int,
                        help="fail if the SWAV length and loop start point wouldn't be multiples of this number of blocks")

    args = parser.parse_args(argv)

    swav_path = args.swav
    if swav_path is None:
        swav_path = args.wav.with_suffix('.swav')

    temp_file_dir = args.temp_file_dir
    if temp_file_dir is None:
        temp_file_dir = Path(os.getcwd())
    else:
        if not os.path.exists(temp_file_dir):
            os.mkdir(temp_file_dir)

    adpcm_xq_path = args.adpcm_xq
    if adpcm_xq_path is None:
        adpcm_xq_path = Path('adpcm-xq')

    if bin(args.block_size).count('1') != 1:
        raise ValueError('block size must be a power of 2')
    block_size_samples = (args.block_size - 4) * 2

    # Split wav if needed
    with wave.open(str(args.wav), 'r') as in_wav:
        num_channels = in_wav.getnchannels()
        num_samples = in_wav.getnframes()

    # Enforce restrictions
    if args.shadow_buffer_size:
        multiple_of = block_size_samples * args.shadow_buffer_size

        def enforce_is_multiple_of(value: int, name: str, multiple_of: int) -> None:
            if value % multiple_of:
                prev = (value // multiple_of) * multiple_of
                next = prev + multiple_of
                raise ValueError(
                    f'with block size {args.block_size:x} and shadow-buffer size {args.shadow_buffer_size}, {name} must be a multiple of {multiple_of} samples, which {value} is not.\nMaybe consider changing it to {prev} (-{value - prev}) or {next} (+{next - value})?')

        enforce_is_multiple_of(num_samples, 'total length', multiple_of)
        if args.loop_start:
            enforce_is_multiple_of(args.loop_start, 'loop start', multiple_of)

    if num_channels == 1:
        mono_input_wavs = [args.wav]

    elif num_channels == 2:
        mono_input_wavs = [
            args.wav.parent / f'{args.wav.stem}_L.wav',
            args.wav.parent / f'{args.wav.stem}_R.wav']

        if not args.skip_conversion:
            split_stereo_wav(args.wav, *mono_input_wavs)

    else:
        raise ValueError(f'More than 2 channels in a wav?? {num_channels}')

    mono_adpcm_wavs = []

    for mono_input_wav in mono_input_wavs:
        mono_input_wav_mangled = temp_file_dir / f'{mono_input_wav.stem}_mangled_{args.block_size:x}.wav'
        mono_adpcm_wav = temp_file_dir / f'{mono_input_wav.stem}_adpcm_{args.block_size:x}_{args.lookahead}.wav'
        mono_adpcm_wavs.append(mono_adpcm_wav)

        if not args.skip_conversion:
            do_wav_mangling(mono_input_wav, mono_input_wav_mangled, block_size_samples)
            run_adpcm_xq(mono_input_wav_mangled, mono_adpcm_wav, adpcm_xq_path,
                         lookahead=args.lookahead, block_size_pow=round(math.log2(args.block_size)))

    print('Creating SWAV: %s' % str(swav_path))
    swav = create_swav_from_adpcm_wavs(mono_adpcm_wavs[0], loop_start=args.loop_start)
    swav.saveToFile(swav_path, updateTime=True, updateTotalLength=True)


if __name__ == '__main__':
    main()
