// Persistence is optional: blocked storage or exhausted quota must not break API work.
export const storage = {
  getItem(key: string): string | null {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  setItem(key: string, value: string): boolean {
    try { localStorage.setItem(key, value); return true; } catch { return false; }
  },
  removeItem(key: string): boolean {
    try { localStorage.removeItem(key); return true; } catch { return false; }
  },
};
