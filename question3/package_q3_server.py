"""Create the Q3 server source bundle, excluding data and checkpoints."""
import zipfile

from q3_common import ROOT, dump, sha


def main():
    folder = ROOT / 'question3'
    files = []
    files.extend(sorted(folder.glob('q3_*.py')))
    files += [folder / name for name in (
        'test_q3.py', 'requirements.txt', 'README_Q3_SERVER.md',
        'README.md', 'package_q3_server.py', 'run_q3_server.sh')]
    files = sorted(set(files))
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))

    output = ROOT / 'outputs/question3'
    manifest = output / '问题3_服务器代码清单.json'
    dump(manifest, {
        p.relative_to(ROOT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in files
    })
    archive = output / '问题3_服务器代码包.zip'
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
