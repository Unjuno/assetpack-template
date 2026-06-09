import json
import sys
from pathlib import Path

REPORTS = Path('reports')

REQUIRED_REPORTS = [
    'bonsai-tokenizer-smoke/report.json',
    'bonsai-text-embedding-smoke/report.json',
    'bonsai-prompt-context-select/report.json',
    'bonsai-prompt-projection-contract/report.json',
    'bonsai-prompt-projection-shape-contract/report.json',
    'bonsai-prompt-projection-load-contract/report.json',
    'bonsai-prompt-projection-load-fixture-contract/report.json',
    'bonsai-prompt-adapter-contract/report.json',
    'bonsai-prompt-staged-smoke/report.json',
    'bonsai-prompt-staged-vae-smoke/report.json',
]

REQUIRED_FILES = [
    'bonsai-prompt-staged-vae-smoke/prompt_staged_vae.png',
]


def load_report(rel):
    path = REPORTS / rel
    if not path.is_file():
        raise AssertionError(f'missing report: {path}')
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise AssertionError(f'invalid json: {path}: {exc}') from exc
    if data.get('ok') is not True:
        raise AssertionError(f'report not ok: {path}: {data}')
    return data


def require_file(rel):
    path = REPORTS / rel
    if not path.is_file():
        raise AssertionError(f'missing file: {path}')
    if path.stat().st_size <= 0:
        raise AssertionError(f'empty file: {path}')
    return path


def main():
    reports = {rel: load_report(rel) for rel in REQUIRED_REPORTS}
    files = {rel: require_file(rel) for rel in REQUIRED_FILES}

    vae = reports['bonsai-prompt-staged-vae-smoke/report.json']
    png = files['bonsai-prompt-staged-vae-smoke/prompt_staged_vae.png']
    if vae.get('decode_success') is not True:
        raise AssertionError(f'vae decode_success not true: {vae}')
    if int(vae.get('png_size_bytes', -1)) != png.stat().st_size:
        raise AssertionError(
            f'png size mismatch: report={vae.get("png_size_bytes")} actual={png.stat().st_size}'
        )

    projection = reports['bonsai-prompt-projection-contract/report.json']
    if projection.get('projection_wired') is not False:
        raise AssertionError(f'projection contract unexpectedly wired: {projection}')

    shape = reports['bonsai-prompt-projection-shape-contract/report.json']
    if shape.get('shape') != [2, 3, 7680]:
        raise AssertionError(f'projection shape contract mismatch: {shape}')

    load = reports['bonsai-prompt-projection-load-contract/report.json']
    if load.get('projection_wired') is not False:
        raise AssertionError(f'projection load contract unexpectedly wired: {load}')
    if load.get('used_in_generation_path') is not False:
        raise AssertionError(f'projection load contract unexpectedly used in generation path: {load}')

    fixture = reports['bonsai-prompt-projection-load-fixture-contract/report.json']
    if fixture.get('load_available') is not True:
        raise AssertionError(f'projection load fixture was not available: {fixture}')
    if fixture.get('load_attempted') is not True:
        raise AssertionError(f'projection load fixture was not attempted: {fixture}')
    if fixture.get('projection_wired') is not False:
        raise AssertionError(f'projection load fixture unexpectedly wired: {fixture}')
    if fixture.get('used_in_generation_path') is not False:
        raise AssertionError(f'projection load fixture unexpectedly used in generation path: {fixture}')

    print(json.dumps({
        'ok': True,
        'reports_checked': sorted(REQUIRED_REPORTS),
        'files_checked': sorted(REQUIRED_FILES),
        'png_size_bytes': png.stat().st_size,
        'projection_wired': projection.get('projection_wired'),
        'projection_shape': shape.get('shape'),
        'projection_load_available': load.get('load_available'),
        'projection_load_fixture_available': fixture.get('load_available'),
    }, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, sort_keys=True))
        sys.exit(1)
