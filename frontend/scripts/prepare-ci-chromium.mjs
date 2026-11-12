// CI downloads this pinned npm package instead of relying on the Playwright CDN.
// The binary and its Amazon Linux compatibility libraries are bundled as brotli
// archives, so the runner needs no browser download or apt/snap package.
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const { default: Chromium, inflate } = require('@sparticuz/chromium');
const packageEntry = require.resolve('@sparticuz/chromium');
const bin = join(dirname(packageEntry), '..', 'bin');
await inflate(join(bin, 'al2023.tar.br'));
process.stdout.write(await Chromium.executablePath());
process.stderr.write(`Chromium compatibility libraries: ${join(tmpdir(), 'al2023', 'lib')}\n`);
