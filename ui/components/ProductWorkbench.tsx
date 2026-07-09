"use client";

import { useState, type ReactNode } from "react";

export type WorkbenchTab = "product" | "architecture";

type Props = {
  eyebrow: string;
  productName: string;
  subtitle: string;
  headerActions?: ReactNode;
  productPanel: ReactNode;
  architecturePanel: ReactNode;
  defaultTab?: WorkbenchTab;
};

export function ProductWorkbench({
  eyebrow,
  productName,
  subtitle,
  headerActions,
  productPanel,
  architecturePanel,
  defaultTab = "product",
}: Props) {
  const [tab, setTab] = useState<WorkbenchTab>(defaultTab);

  return (
    <div className="workbench">
      <div className="workbench-hero">
        <div className="workbench-hero__top">
          <div>
            <p className="workbench-eyebrow">{eyebrow}</p>
            <h1 className="workbench-title">{productName}</h1>
          </div>
          {headerActions ? <div className="workbench-actions">{headerActions}</div> : null}
        </div>
        <p className="workbench-subtitle">{subtitle}</p>
        <nav className="workbench-tabs" role="tablist" aria-label="Product workbench">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "product"}
            className={`workbench-tab${tab === "product" ? " is-active" : ""}`}
            onClick={() => setTab("product")}
          >
            <span className="workbench-tab__label">Live product</span>
            <span className="workbench-tab__hint">Run the demo</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "architecture"}
            className={`workbench-tab${tab === "architecture" ? " is-active" : ""}`}
            onClick={() => setTab("architecture")}
          >
            <span className="workbench-tab__label">Architecture & metrics</span>
            <span className="workbench-tab__hint">Stack, tradeoffs, SLOs</span>
          </button>
        </nav>
      </div>
      <div className="workbench-body" role="tabpanel">
        {tab === "product" ? productPanel : architecturePanel}
      </div>
    </div>
  );
}
