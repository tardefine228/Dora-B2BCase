"""Generate reproducible synthetic fixtures against the repository's frozen catalog."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIELDS = ['id', 'product_name', 'true_okpd2', 'appendix', 'description_length']


def generate():
    with (ROOT / 'data/data.csv').open(encoding='utf-8-sig', newline='') as f:
        catalog = list(csv.DictReader(f, delimiter=';'))
    products = [line.split('|') for line in (HERE / 'products.txt').read_text().splitlines()]
    assert len(products) == 100
    # Interleave the ten subject areas instead of making subject-specific batches.
    products = [products[area * 10 + offset] for offset in range(10) for area in range(10)]
    all_rows = []
    provenance = []
    for batch in range(5):
        rows = []
        for index, (code, appendix, base, variants, medium, details) in enumerate(products):
            matching = [r for r in catalog if r['okpd2'] == code and r['appendix'] == appendix
                        and r['code_role'] == 'Включён']
            assert matching, (code, appendix)
            variant = variants.split(';')[batch]
            slot = (index + 2 * batch) % 10
            length = 'short' if slot < 3 else 'medium' if slot < 7 else 'long'
            name = f'{base} {variant}'
            if length != 'short':
                name += ' ' + medium
            if length == 'long':
                # Append whole specification clauses, never truncate inside a clause.
                for clause in details.split(', '):
                    name += ', ' + clause
                    if len(name.split()) >= 20:
                        break
            if length == 'short':
                assert 2 <= len(name.split()) <= 4, name
            if length == 'long':
                assert len(name.split()) >= 20, name
            row = dict(zip(FIELDS, [batch * 100 + index + 1, name, code, appendix, length]))
            rows.append(row)
            provenance.append({
                'id': row['id'], 'record_id': matching[0]['record_id'],
                'position': matching[0]['position'],
                'pdf_page_start': matching[0]['pdf_page_start'],
                'pdf_page_end': matching[0]['pdf_page_end'],
                'catalog_name': matching[0]['name'],
            })
        assert Counter(r['description_length'] for r in rows) == {'short': 30, 'medium': 40, 'long': 30}
        assert {r['appendix'] for r in rows} == {'1', '2', '3'}
        with (HERE / f'test_batch_{batch + 1}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        all_rows.extend(rows)
    assert len({r['product_name'].casefold() for r in all_rows}) == 500
    assert [r['id'] for r in all_rows] == list(range(1, 501))
    with (HERE / 'provenance.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(provenance[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(provenance)
    manifest = {
        'dataset_kind': 'synthetic_positive_fixtures',
        'catalog_sha256': hashlib.sha256((ROOT / 'data/data.csv').read_bytes()).hexdigest(),
        'source_text_sha256': hashlib.sha256((ROOT / 'data/source_text.txt').read_bytes()).hexdigest(),
        'source_report': json.loads((ROOT / 'data/report.json').read_text()),
        'rows': len(all_rows), 'unique_product_names': len({r['product_name'] for r in all_rows}),
        'description_length_counts': dict(Counter(r['description_length'] for r in all_rows)),
        'appendix_counts': dict(Counter(r['appendix'] for r in all_rows)),
        'files': {f'test_batch_{i}.csv': hashlib.sha256((HERE / f'test_batch_{i}.csv').read_bytes()).hexdigest() for i in range(1, 6)},
    }
    (HERE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['rows', 'unique_product_names', 'description_length_counts', 'appendix_counts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    generate()
