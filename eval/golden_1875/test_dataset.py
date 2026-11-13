import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIELDS = ['id', 'product_name', 'true_okpd2', 'appendix', 'description_length']


class DatasetTests(unittest.TestCase):
    def test_all_batches(self):
        with (ROOT / 'data/data.csv').open(encoding='utf-8-sig', newline='') as f:
            catalog = list(csv.DictReader(f, delimiter=';'))
        allowed = {(r['okpd2'], r['appendix']) for r in catalog if r['code_role'] == 'Включён'}
        manifest = json.loads((HERE / 'manifest.json').read_text())
        all_rows = []
        for batch in range(1, 6):
            path = HERE / f'test_batch_{batch}.csv'
            with path.open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, FIELDS)
                rows = list(reader)
            self.assertEqual(len(rows), 100)
            self.assertEqual(Counter(r['description_length'] for r in rows),
                             {'short': 30, 'medium': 40, 'long': 30})
            self.assertEqual({r['appendix'] for r in rows}, {'1', '2', '3'})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, manifest['files'][path.name])
            for row in rows:
                self.assertNotIn(None, row)
                self.assertTrue(all(row.values()))
                self.assertIn((row['true_okpd2'], row['appendix']), allowed)
                words = len(row['product_name'].split())
                if row['description_length'] == 'short':
                    self.assertTrue(2 <= words <= 4)
                elif row['description_length'] == 'long':
                    self.assertGreaterEqual(words, 20)
                else:
                    self.assertTrue(5 <= words < 20)
            all_rows.extend(rows)
        self.assertEqual([int(r['id']) for r in all_rows], list(range(1, 501)))
        self.assertEqual(len({r['product_name'].casefold() for r in all_rows}), 500)
        self.assertEqual(hashlib.sha256((ROOT / 'data/data.csv').read_bytes()).hexdigest(),
                         manifest['catalog_sha256'])
        with (HERE / 'provenance.csv').open(encoding='utf-8', newline='') as f:
            provenance = list(csv.DictReader(f))
        self.assertEqual(len(provenance), 500)
        by_record = {r['record_id']: r for r in catalog}
        for row, source in zip(all_rows, provenance, strict=True):
            self.assertEqual(row['id'], source['id'])
            original = by_record[source['record_id']]
            self.assertEqual((row['true_okpd2'], row['appendix']),
                             (original['okpd2'], original['appendix']))
            self.assertEqual(source['catalog_name'], original['name'])


if __name__ == '__main__':
    unittest.main()
