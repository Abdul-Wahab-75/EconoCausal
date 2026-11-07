
import { useState } from "react";
import Plot from "react-plotly.js";
import Papa from "papaparse";
import {
  checkBackend,
  trainModel,
  predictITE,
  evaluateUplift,
  optimizeBudget,
} from "./api";
import "./App.css";

export default function App() {
  const [rows, setRows] = useState([]);
  const [numericColumns, setNumericColumns] = useState([]);
  const [treatment, setTreatment] = useState("");
  const [outcome, setOutcome] = useState("");
  const [features, setFeatures] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [evaluationRows, setEvaluationRows] = useState([]);
  const [message, setMessage] = useState("Upload a CSV file to begin.");
  const [loading, setLoading] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [upliftResult, setUpliftResult] = useState(null);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [budget, setBudget] = useState(10000);
  const [discountRate, setDiscountRate] = useState(15);

  const handleFileUpload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setRows([]);
    setPredictions([]);
    setEvaluationRows([]);
    setModelInfo(null);
    setUpliftResult(null);
    setOptimizationResult(null);
    setMessage("Reading CSV...");

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        const data = result.data;
        const headers = (result.meta.fields || []).map((c) => c.trim());

        if (!headers.length || !data.length) {
          setMessage("CSV is empty or has no valid headers.");
          return;
        }

        const numeric = headers.filter((column) =>
          data.every((row) => {
            const value = row[column];
            return (
              value !== undefined &&
              String(value).trim() !== "" &&
              Number.isFinite(Number(value))
            );
          })
        );

        const treatmentColumn = headers.find((c) =>
          ["discount", "treatment", "promo"].includes(c.toLowerCase())
        );
        const outcomeColumn = headers.find((c) =>
          ["sales", "outcome", "revenue"].includes(c.toLowerCase())
        );

        setRows(data);
        setNumericColumns(numeric);
        setTreatment(treatmentColumn || "");
        setOutcome(outcomeColumn || "");
        setFeatures(
          numeric.filter(
            (c) =>
              c !== treatmentColumn &&
              c !== outcomeColumn &&
              !/(^id$|_id$|^customerid$|^store$|^date$)/i.test(c)
          )
        );

        setMessage(
          `CSV loaded: ${data.length} records and ${headers.length} columns.`
        );
      },
      error: (error) => setMessage(`CSV error: ${error.message}`),
    });
  };

  const handleTrain = async () => {
    try {
      setLoading(true);
      setPredictions([]);
      setEvaluationRows([]);
      setModelInfo(null);
      setUpliftResult(null);
      setOptimizationResult(null);

      if (!rows.length) throw new Error("Upload a CSV first.");
      if (!treatment || !outcome || !features.length) {
        throw new Error("Select Treatment, Outcome and Features.");
      }
      if (
        features.includes(treatment) ||
        features.includes(outcome) ||
        treatment === outcome
      ) {
        throw new Error("Treatment, Outcome and Features must be different.");
      }

      setMessage("Connecting to backend...");
      await checkBackend();

      const required = [...features, treatment, outcome];
      const validRows = rows.filter((row) =>
        required.every(
          (column) =>
            row[column] !== undefined &&
            String(row[column]).trim() !== "" &&
            Number.isFinite(Number(row[column]))
        )
      );

      if (validRows.length < 50) {
        throw new Error("At least 50 valid records are required.");
      }

      const X = validRows.map((row) =>
        features.map((column) => Number(row[column]))
      );
      const T = validRows.map((row) => Number(row[treatment]));
      const Y = validRows.map((row) => Number(row[outcome]));

      if (
        !T.every((v) => v === 0 || v === 1) ||
        !T.includes(0) ||
        !T.includes(1)
      ) {
        throw new Error("Treatment must contain both 0 and 1.");
      }

      // Deterministic 80:20 split.
      const trainIndexes = validRows
        .map((_, i) => i)
        .filter((i) => i % 5 !== 0);
      const testIndexes = validRows
        .map((_, i) => i)
        .filter((i) => i % 5 === 0);

      const pick = (array, indexes) =>
        indexes.map((i) => array[i]);

      const trainT = pick(T, trainIndexes);
      const testT = pick(T, testIndexes);

      if (
        !trainT.includes(0) ||
        !trainT.includes(1) ||
        !testT.includes(0) ||
        !testT.includes(1)
      ) {
        throw new Error("Both treatment groups must exist in both splits.");
      }

      setMessage(`Training model on ${trainIndexes.length} records...`);

      const fit = await trainModel({
        feature_names: features,
        features: pick(X, trainIndexes),
        treatment: trainT,
        outcome: pick(Y, trainIndexes),
        n_splits: 3,
        nuisance_type: "random_forest",
        random_state: 42,
      });

      setMessage("Predicting ITE on evaluation records...");

      const predictionResponse = await predictITE({
        feature_names: features,
        features: pick(X, testIndexes),
      });

      const ite = predictionResponse.predictions || [];

      if (
        ite.length !== testIndexes.length ||
        ite.some((v) => !Number.isFinite(Number(v)))
      ) {
        throw new Error("Invalid ITE predictions.");
      }

      setMessage("Calculating Qini and uplift...");

      const testY = pick(Y, testIndexes);

      const uplift = await evaluateUplift({
        treatment: testT,
        outcome: testY,
        ite_predictions: ite,
      });

      setPredictions(ite.map(Number));
      setEvaluationRows(testIndexes.map((i) => i + 1));
      setModelInfo({
        ...fit,
        evaluation_observations: testIndexes.length,
      });
      setUpliftResult(uplift);

      setMessage(
        `Success! Trained on ${trainIndexes.length} records and evaluated on ${testIndexes.length} separate records.`
      );
    } catch (error) {
      setMessage(
        `Error: ${
          error.response?.data?.detail ||
          error.message ||
          "Something went wrong."
        }`
      );
    } finally {
      setLoading(false);
    }
  };

  const handleOptimize = async () => {
    try {
      setOptimizing(true);
      setOptimizationResult(null);

      if (!predictions.length) {
        throw new Error("Train the model before optimization.");
      }

      if (!Number.isFinite(Number(budget)) || Number(budget) <= 0) {
        throw new Error("Enter a budget greater than zero.");
      }

      if (
        !Number.isFinite(Number(discountRate)) ||
        Number(discountRate) <= 0 ||
        Number(discountRate) > 100
      ) {
        throw new Error("Discount rate must be between 1 and 100.");
      }

      setMessage("Running SciPy budget optimization...");

      const assumedPrice = 50;

      const result = await optimizeBudget({
        ite_estimates: predictions,
        item_prices: predictions.map(() => assumedPrice),
        budget: Number(budget),
        discount_rate: Number(discountRate) / 100,
      });

      setOptimizationResult(result);
      setMessage("Budget optimization completed successfully.");
    } catch (error) {
      setMessage(
        `Optimization error: ${
          error.response?.data?.detail ||
          error.message ||
          "Something went wrong."
        }`
      );
    } finally {
      setOptimizing(false);
    }
  };

  const ranked = predictions
    .map((ite, index) => ({
      row: evaluationRows[index],
      ite,
    }))
    .sort((a, b) => b.ite - a.ite);

  return (
    <main style={{ maxWidth: 1200, margin: "0 auto", padding: 24 }}>
      <h1>EconoCausal</h1>
      <p>Dynamic Pricing via Double Machine Learning</p>

      <section>
        <h2>1. Upload Historical Data</h2>
        <input
          type="file"
          accept=".csv"
          onChange={handleFileUpload}
          disabled={loading || optimizing}
        />
        <p>{message}</p>
        {rows.length > 0 && (
          <p>
            Loaded {rows.length} records. Numeric columns:{" "}
            {numericColumns.length}
          </p>
        )}
      </section>

      {rows.length > 0 && (
        <section>
          <h2>2. Configure Causal Model</h2>

          <label>
            Treatment column{" "}
            <select
              value={treatment}
              onChange={(e) => setTreatment(e.target.value)}
            >
              <option value="">Select treatment</option>
              {numericColumns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>

          <br /><br />

          <label>
            Outcome column{" "}
            <select
              value={outcome}
              onChange={(e) => setOutcome(e.target.value)}
            >
              <option value="">Select outcome</option>
              {numericColumns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>

          <br /><br />

          <label>
            Feature columns (hold Ctrl to select multiple)
            <br />
            <select
              multiple
              value={features}
              onChange={(e) =>
                setFeatures(
                  Array.from(e.target.selectedOptions, (o) => o.value)
                )
              }
              size={Math.min(Math.max(numericColumns.length, 4), 10)}
              style={{ minWidth: 260 }}
            >
              {numericColumns
                .filter((c) => c !== treatment && c !== outcome)
                .map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
            </select>
          </label>

          <p>Selected features: {features.join(", ") || "None"}</p>

          <button onClick={handleTrain} disabled={loading || optimizing}>
            {loading ? "Please wait..." : "Train Model & Predict ITE"}
          </button>
        </section>
      )}

      {modelInfo && (
        <section>
          <h2>3. Model Results</h2>
          <p>Training observations: {modelInfo.observations}</p>
          <p>
            Evaluation observations: {modelInfo.evaluation_observations}
          </p>
          <p>ATE: {Number(modelInfo.ate).toFixed(4)}</p>
          <p>Standard Error: {Number(modelInfo.standard_error).toFixed(4)}</p>
        </section>
      )}

      {predictions.length > 0 && (
        <section>
          <h2>4. Individual Treatment Effect (ITE)</h2>
          <p>
            Positive ITE indicates a predicted positive treatment effect;
            negative ITE indicates a predicted negative effect.
          </p>

          <Plot
            data={[
              {
                x: ranked.map((v) => v.row),
                y: ranked.map((v) => v.ite),
                type: "scatter",
                mode: "lines",
                name: "Predicted ITE",
              },
            ]}
            layout={{
              title: "Predicted Individual Treatment Effects",
              xaxis: { title: "Evaluation record" },
              yaxis: { title: "Predicted ITE" },
              autosize: true,
            }}
            style={{ width: "100%" }}
            useResizeHandler
          />

          <h3>Top 10 Predicted Records</h3>
          <table>
            <thead>
              <tr><th>CSV row</th><th>Predicted ITE</th></tr>
            </thead>
            <tbody>
              {ranked.slice(0, 10).map((item) => (
                <tr key={item.row}>
                  <td>{item.row}</td>
                  <td>{item.ite.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {upliftResult && (
        <section>
          <h2>5. Qini / Uplift Evaluation</h2>
          <p>Qini Score: {Number(upliftResult.qini_score).toFixed(4)}</p>

          <Plot
            data={[
              {
                x: upliftResult.uplift_curve.map(
                  (p) => p.FractionTargeted
                ),
                y: upliftResult.uplift_curve.map(
                  (p) => p.CumulativeQini
                ),
                type: "scatter",
                mode: "lines+markers",
                name: "Model Qini",
              },
              {
                x: upliftResult.uplift_curve.map(
                  (p) => p.FractionTargeted
                ),
                y: upliftResult.uplift_curve.map(
                  (p) => p.RandomBaseline
                ),
                type: "scatter",
                mode: "lines",
                name: "Random Baseline",
              },
            ]}
            layout={{
              title: "Qini / Uplift Curve (Held-out Data)",
              xaxis: { title: "Fraction of records targeted" },
              yaxis: { title: "Cumulative Qini" },
              autosize: true,
            }}
            style={{ width: "100%" }}
            useResizeHandler
          />
        </section>
      )}

      {predictions.length > 0 && (
        <section>
          <h2>6. SciPy Budget Optimization</h2>
          <p>
            The Rossmann dataset does not provide an actual item price in this
            sample. The optimizer therefore uses an assumed price of ₹50 per
            record. This is an illustrative assumption, not observed pricing.
          </p>

          <label>
            Available budget (₹)
            <br />
            <input
              type="number"
              min="1"
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
              disabled={optimizing}
            />
          </label>

          <br /><br />

          <label>
            Discount rate (%)
            <br />
            <input
              type="number"
              min="1"
              max="100"
              value={discountRate}
              onChange={(e) => setDiscountRate(e.target.value)}
              disabled={optimizing}
            />
          </label>

          <p>Assumed item price: ₹50 per record</p>

          <button
            onClick={handleOptimize}
            disabled={optimizing || loading}
          >
            {optimizing ? "Optimizing..." : "Optimize Budget with SciPy"}
          </button>
        </section>
      )}

      {optimizationResult && (
        <section>
          <h2>7. Budget Allocation Results</h2>
          <p>
            Selected records: {optimizationResult.targeted_customers}
          </p>
          <p>
            Total budget spent: ₹
            {Number(optimizationResult.total_cost).toFixed(2)}
          </p>
          <p>
            Budget limit: ₹
            {Number(optimizationResult.budget_limit).toFixed(2)}
          </p>
          <p>
            Remaining budget: ₹
            {(
              Number(optimizationResult.budget_limit) -
              Number(optimizationResult.total_cost)
            ).toFixed(2)}
          </p>
          <p>
            Estimated incremental value:{" "}
            {Number(optimizationResult.total_incremental_value).toFixed(2)}
          </p>
          <p>
            Estimated ROI:{" "}
            {Number(optimizationResult.estimated_roi).toFixed(4)}
          </p>

          <h3>Allocation Matrix</h3>
          <table>
            <thead>
              <tr>
                <th>Evaluation record</th>
                <th>Predicted ITE</th>
                <th>Decision</th>
                <th>Estimated cost (₹)</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((ite, index) => (
                <tr key={index}>
                  <td>{evaluationRows[index]}</td>
                  <td>{Number(ite).toFixed(4)}</td>
                  <td>
                    {optimizationResult.selected_flags[index]
                      ? "SELECTED"
                      : "NOT SELECTED"}
                  </td>
                  <td>
                    {Number(
                      optimizationResult.customer_costs[index]
                    ).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </main>
  );
}