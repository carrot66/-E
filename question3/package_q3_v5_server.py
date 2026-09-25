"""Create the v5 Q3 server source bundle, excluding data and checkpoints."""
import zipfile

from q3v2_common import ROOT, dump, sha


def main():
    folder = ROOT / 'question3'
    files = []
    for pattern in ('q3v2_*.py', 'q3v3_*.py', 'q3v4_*.py', 'q3v5_*.py'):
        files.extend(sorted(folder.glob(pattern)))
    files += [folder / name for name in (
        'test_q3v2.py', 'test_q3v3.py', 'test_q3v5.py', 'requirements_v2.txt',
        'README_Q3_V2_SERVER.md', 'README_Q3_V3_SERVER.md',
        'README_Q3_V5_SERVER.md', 'package_q3_v5_server.py')]
    files += [ROOT / 'run_q3_v5_server.sh']
    files = sorted(set(files))
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))

    output = ROOT / 'outputs/question3'
    manifest = output / '问题3_v5_服务器代码清单.json'
    dump(manifest, {
        p.relative_to(ROOT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in files
    })
    archive = output / '问题3_v5_服务器代码包.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.write(manifest, manifest.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('ZIP contains duplicate paths')
    print(archive)
    print(f'{archive.stat().st_size} bytes; {len(files)} source files plus manifest')


if __name__ == '__main__':
    main()
