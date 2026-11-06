import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import "./App.css";

const initialCustomers = [
  { id: "C001", ite: 24.5, discount: 15, decision: "TARGET" },
  { id: "C002", ite: 18.2, discount: 10, decision: "TARGET" },
  { id: "C003", ite: 8.4, discount: 5, decision: "SKIP" },
  { id: "C004", ite: 31.7, discount: 20, decision: "TARGET" },
  { id: "C005", ite: 4.1, discount: 0, decision: "SKIP" },
];

function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [budget, setBudget] = useState(10000);
  const [fileName, setFileName] = useState("");
  const [customers, setCustomers] = useState(initialCustomers);

  const targeted = customers.filter((c) => c.decision === "TARGET").length;
  const averageITE = customers.length
    ? customers.reduce((sum, c) => sum + c.ite, 0) / customers.length
    : 0;

  const navigation = [
    ["Overview", "◫"],
    ["Data Upload", "⇧"],
    ["Causal Analysis", "⌁"],
    ["Budget Allocation", "▦"],
  ];

  function handleFileUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);

    if (file.name.toLowerCase().endsWith(".csv")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const lines = String(e.target.result).trim().split(/\r?\n/);
        const rows = lines.slice(1).map((line, index) => {
          const values = line.split(",");
          return {
            id: values[0] || `Customer-${index + 1}`,
            ite: Number(values[1]) || 0,
            discount: Number(values[2]) || 0,
            decision: Number(values[1]) > 0 ? "TARGET" : "SKIP",
          };
        });
        if (rows.length) setCustomers(rows);
      };
      reader.readAsText(file);
    }
  }

  const chartData = useMemo(() => {
    const sorted = [...customers].sort((a, b) => b.ite - a.ite);
    return sorted.map((c, i) => ({
      x: i + 1,
      y: c.ite,
    }));
  }, [customers]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">EC</div>
          <div>
            <h2>EconoCausal</h2>
            <span>CAUSAL INTELLIGENCE</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav>
          {navigation.map(([name, icon]) => (
            <button
              key={name}
              className={`nav-item ${activePage === name ? "active" : ""}`}
              onClick={() => setActivePage(name)}
            >
              <span>{icon}</span>{name}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <span>Frontend environment ready</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">DYNAMIC PRICING PLATFORM</div>
            <h1>{activePage}</h1>
          </div>
          <div className="topbar-right">
            <span className="environment">DEVELOPMENT</span>
            <div className="avatar">P</div>
          </div>
        </header>

        <section className="welcome-row">
          <div>
            <h2>Campaign intelligence overview</h2>
            <p>Use causal effects to make smarter promotional decisions.</p>
          </div>
          <label className="upload-button">
            + Upload campaign data
            <input type="file" accept=".csv,.xlsx" onChange={handleFileUpload} />
          </label>
        </section>

        {fileName && (
          <div className="file-notice">
            Selected file: <strong>{fileName}</strong>
            <span> CSV preview loaded locally. Backend processing is not connected yet.</span>
          </div>
        )}

        <section className="metric-grid">
          <Metric title="CUSTOMER RECORDS" value={customers.length.toLocaleString()} note="Current frontend data" />
          <Metric title="AVERAGE ITE" value={averageITE.toFixed(2)} note="Illustrative local values" />
          <Metric title="TARGETED CUSTOMERS" value={targeted.toLocaleString()} note="Based on current table" />
          <Metric title="MARKETING BUDGET" value={`₹${budget.toLocaleString("en-IN")}`} note="Your selected budget" />
        </section>

        <section className="content-grid">
          <div className="panel chart-panel">
            <div className="panel-heading">
              <div>
                <h3>Individual Treatment Effect</h3>
                <p>Customer-level estimated uplift</p>
              </div>
              <span className="tag">ITE ANALYSIS</span>
            </div>
            <Plot
              data={[{
                x: chartData.map((p) => p.x),
                y: chartData.map((p) => p.y),
                type: "scatter",
                mode: "lines+markers",
                marker: { size: 8 },
                line: { width: 3 },
                name: "ITE",
              }]}
              layout={{
                autosize: true,
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#aab5ca" },
                margin: { l: 48, r: 20, t: 20, b: 45 },
                xaxis: { title: "Customer rank", gridcolor: "#263248" },
                yaxis: { title: "Estimated ITE", gridcolor: "#263248" },
                showlegend: false,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%", height: "300px" }}
              useResizeHandler
            />
            <p className="chart-disclaimer">Illustrative frontend values only — not yet connected to EconML.</p>
          </div>

          <div className="panel budget-panel">
            <div className="panel-heading">
              <div>
                <h3>Campaign budget</h3>
                <p>Set your available marketing budget</p>
              </div>
              <span className="budget-icon">₹</span>
            </div>
            <div className="budget-value">₹{budget.toLocaleString("en-IN")}</div>
            <input
              className="budget-slider"
              type="range"
              min="1000"
              max="100000"
              step="1000"
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
            />
            <div className="range-labels"><span>₹1,000</span><span>₹1,00,000</span></div>
            <div className="budget-info">
              <span>Optimization status</span>
              <strong>Awaiting backend connection</strong>
            </div>
          </div>
        </section>

        <section className="panel table-panel">
          <div className="panel-heading">
            <div>
              <h3>Customer allocation preview</h3>
              <p>Individual uplift and targeting decisions</p>
            </div>
            <span className="tag">ALLOCATION MATRIX</span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>CUSTOMER ID</th>
                  <th>ESTIMATED ITE</th>
                  <th>DISCOUNT (%)</th>
                  <th>DECISION</th>
                </tr>
              </thead>
              <tbody>
                {customers.slice(0, 10).map((customer) => (
                  <tr key={customer.id}>
                    <td>{customer.id}</td>
                    <td>{customer.ite.toFixed(2)}</td>
                    <td>{customer.discount}%</td>
                    <td>
                      <span className={`decision ${customer.decision === "TARGET" ? "target" : "skip"}`}>
                        {customer.decision}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="chart-disclaimer">Preview only. Final allocation will come from the existing SciPy optimizer through FastAPI.</p>
        </section>

        <footer>ECONOCAUSAL · DYNAMIC PRICING VIA DOUBLE MACHINE LEARNING</footer>
      </main>
    </div>
  );
}

function Metric({ title, value, note }) {
  return (
    <div className="panel metric-card">
      <span className="metric-title">{title}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

export default App;