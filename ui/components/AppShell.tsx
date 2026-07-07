'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const activeNav = pathname?.startsWith('/bench') ? 'bench' : 'triage';

  return (
    <>
      <header className="app-header">
        <div className="app-header__inner">
          <Link href="/" className="brand">
            <div className="brand__mark">DF</div>
            <div className="brand__text">
              <p className="brand__eyebrow">Governed RAG + PEFT</p>
              <p className="brand__title">DomainForge</p>
            </div>
          </Link>
          <nav className="app-nav" aria-label="Primary">
            <Link href="/" className={activeNav === 'triage' ? 'is-active' : undefined}>
              Support triage
            </Link>
            <Link href="/bench" className={activeNav === 'bench' ? 'is-active' : undefined}>
              Local AI bench
            </Link>
          </nav>
          <div className="header-actions">
            <span className="status-badge">Live API</span>
            <a
              href="https://github.com/vpeetla-ai/domainforge-rag-peft"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-ghost"
              style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem', textDecoration: 'none' }}
            >
              GitHub
            </a>
          </div>
        </div>
      </header>
      <main className="app-main">{children}</main>
      <footer className="app-footer">
        DomainForge · S0→S4 eval ladder ·{' '}
        <a href="https://github.com/vpeetla-ai/domainforge-rag-peft">vpeetla-ai</a>
      </footer>
    </>
  );
}
