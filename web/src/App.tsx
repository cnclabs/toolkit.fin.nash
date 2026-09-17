import { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  FlaskConical,
  Layers3,
  SlidersHorizontal,
  Table2,
  Upload,
  X
} from "lucide-react";
import { fetchBaselineComparison, fetchModels } from "./api";
import demoCasesData from "./demo_data/demo_cases.generated.json";
import type {
  BaselineComparisonResponse,
  ModelOption,
  NashTrace
} from "./types";

type TabId = "batch" | "example" | "cases";
type SortKey = "baseline" | "nash";
type ReactNode = unknown;

type BatchPair = {
  id: string;
  title: string;
  sentence1: string;
  sentence2: string;
  task?: string;
  role?: string;
};

type UploadPairResult = {
  label: string;
  sentence1: string;
  sentence2: string;
  masked_sentence1?: string;
  masked_sentence2?: string;
  baseline_score: number;
  nash_score: number;
  text_score: number;
  numeric_score: number;
  numbers_sentence1?: GeneratedNumber[];
  numbers_sentence2?: GeneratedNumber[];
  alignment?: GeneratedAlignment;
};

type BatchResultRow = BatchPair & {
  baseline_id: string;
  baseline_label: string;
  backend: string;
  baseline_score: number;
  nash_score: number;
  text_score: number;
  numeric_score: number;
  pair_results?: UploadPairResult[];
};

type EvaluationSummary = {
  task: string;
  model: string;
  threshold: string;
  records: string;
  metricLabel: string;
  metricValue: string;
  runtime: string;
};

type GeneratedNumber = {
  index: number;
  raw: string;
  value: number;
  start: number;
  end: number;
};

type GeneratedAlignmentEdge = {
  direction: string;
  source_index: number;
  target_index: number;
  context_similarity: number;
  aligned: boolean;
  magnitude_similarity?: number;
  numeric_contribution?: number;
  source?: GeneratedNumber;
  target?: GeneratedNumber;
};

type GeneratedAlignment = {
  matrix?: number[][];
  selected_alignment_edges?: GeneratedAlignmentEdge[];
};

type GeneratedPairResult = {
  model: string;
  model_key: string;
  display_name: string;
  sentence1: string;
  sentence2: string;
  masked_sentence1?: string;
  masked_sentence2?: string;
  baseline_score: number;
  nash_score: number;
  text_score: number;
  numeric_score: number;
  numbers_sentence1?: GeneratedNumber[];
  numbers_sentence2?: GeneratedNumber[];
  alignment?: GeneratedAlignment;
};

type GeneratedSentencePairCase = {
  id: string;
  title: string;
  sentence1: string;
  sentence2: string;
  model_results: Record<string, GeneratedPairResult>;
};

type GeneratedTripletCase = {
  id: string;
  title: string;
  anchor: string;
  positive: string;
  negative: string;
  model_results: Record<string, {
    positive_pair: GeneratedPairResult;
    negative_pair: GeneratedPairResult;
  }>;
};

type GeneratedCrossPairCase = {
  id: string;
  title: string;
  pair_a: { sentence1: string; sentence2: string };
  pair_b: { sentence1: string; sentence2: string };
  model_results: Record<string, {
    pair_a: GeneratedPairResult;
    pair_b: GeneratedPairResult;
  }>;
};

type GeneratedCasesData = {
  sentence_pair_cases: GeneratedSentencePairCase[];
  triplet_cases: GeneratedTripletCase[];
  cross_pair_cases: GeneratedCrossPairCase[];
};

const generatedCases = demoCasesData as GeneratedCasesData;

const scoreFmt = new Intl.NumberFormat("en", {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3
});

const compactFmt = new Intl.NumberFormat("en", {
  maximumFractionDigits: 4
});

const fallbackModels: ModelOption[] = [
  { id: "bert-base-uncased", label: "baseline / bert-base-uncased", backend: "bertscore", requires_model_dependencies: true },
  { id: "ProsusAI/finbert", label: "FinBERT", backend: "bertscore", requires_model_dependencies: true },
  { id: "sentence-transformers/all-MiniLM-L6-v2", label: "Sentence-BERT / all-MiniLM-L6-v2", backend: "sentence-transformer", requires_model_dependencies: true }
];

const defaultModelIds = ["bert-base-uncased", "ProsusAI/finbert", "sentence-transformers/all-MiniLM-L6-v2"];

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const initialTab = parseInitialTab(params.get("tab"), window.location.pathname);
  const [activeTab, setActiveTab] = useState<TabId>(initialTab);
  const [models, setModels] = useState<ModelOption[]>(fallbackModels);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>(defaultModelIds);
  const [globalError, setGlobalError] = useState("");

  useEffect(() => {
    fetchModels()
      .then((items) => {
        setModels(items.length ? items : fallbackModels);
        const available = defaultModelIds.filter((id) => items.some((model) => model.id === id));
        if (available.length) setSelectedModelIds(available);
      })
      .catch(() => {
        setModels(fallbackModels);
      });
  }, []);

  return (
    <main className="min-h-screen bg-stone-50 text-slate-950">
      <div className="mx-auto grid max-w-[1600px] gap-4 px-4 py-4 lg:px-6">
        <header className="flex flex-col gap-3 border-b border-slate-200 pb-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase text-teal-700">NASH</p>
            <h1 className="text-2xl font-semibold tracking-normal md:text-3xl">NASH DEMO</h1>
          </div>
          <nav className="grid gap-2 rounded-md border border-slate-200 bg-white p-1 shadow-panel sm:grid-cols-3">
            <TabButton id="batch" active={activeTab} onClick={setActiveTab} label="Batch Upload" icon={<Upload size={16} />} />
            <TabButton id="example" active={activeTab} onClick={setActiveTab} label="Interactive Example" icon={<SlidersHorizontal size={16} />} />
            <TabButton id="cases" active={activeTab} onClick={setActiveTab} label="Cases" icon={<Layers3 size={16} />} />
          </nav>
        </header>

        {globalError && <Notice tone="error">{globalError}</Notice>}

        {activeTab === "batch" && (
          <BatchUploadTab
            models={models}
            onError={setGlobalError}
          />
        )}
        {activeTab === "example" && (
          <InteractiveExampleTab
            models={models}
            selectedModelIds={selectedModelIds}
            onSelectedModelIds={setSelectedModelIds}
            initialSentenceA={params.get("sentence-a") ?? params.get("sentenceA") ?? "Revenue increased by 3.5% in Q2."}
            initialSentenceB={params.get("sentence-b") ?? params.get("sentenceB") ?? "Revenue increased by 35% in Q2."}
            onError={setGlobalError}
          />
        )}
        {activeTab === "cases" && <CasesTab />}
      </div>
    </main>
  );
}

function BatchUploadTab({
  models,
  onError
}: {
  models: ModelOption[];
  onError: (message: string) => void;
}) {
  const [pairs, setPairs] = useState<BatchPair[]>([]);
  const [rows, setRows] = useState<BatchResultRow[]>([]);
  const [traces, setTraces] = useState<NashTrace[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [uploadTask, setUploadTask] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("nash");
  const [loading, setLoading] = useState(false);
  const [sourceName, setSourceName] = useState("");
  const [selectedRow, setSelectedRow] = useState<BatchResultRow | null>(null);

  useEffect(() => {
    const payload = new URLSearchParams(window.location.search).get("payload");
    if (!payload) return;
    fetch(payload)
      .then((response) => response.text())
      .then((text) => {
        const parsed = parseBatchPayload(text, payload);
        setPairs(parsed.pairs);
        setRows(parsed.rows ?? []);
        setTraces(parsed.traces ?? []);
        setEvaluation(parsed.evaluation ?? null);
        setUploadTask(parsed.task ?? "");
        setSourceName(payload);
        setSelectedRow(null);
        if (!parsed.rows && !parsed.evaluation && parsed.pairs.length) void scoreBatch(parsed.pairs, defaultBaselines(models), setRows, setTraces, setLoading, onError);
      })
      .catch((error: Error) => {
        setPairs([]);
        setRows([]);
        setTraces([]);
        setEvaluation(null);
        setUploadTask("");
        setSelectedRow(null);
        onError(error.message);
      });
  }, []);

  async function uploadBatch(event: { target: { files: FileList | null; value: string } }) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseBatchPayload(text, file.name);
      setPairs(parsed.pairs);
      setRows(parsed.rows ?? []);
      setTraces(parsed.traces ?? []);
      setEvaluation(parsed.evaluation ?? null);
      setUploadTask(parsed.task ?? "");
      setSourceName(file.name);
      setSelectedRow(null);
      onError("");
      if (!parsed.rows && !parsed.evaluation && parsed.pairs.length) {
        await scoreBatch(parsed.pairs, defaultBaselines(models), setRows, setTraces, setLoading, onError);
      }
    } catch (error) {
      setPairs([]);
      setRows([]);
      setTraces([]);
      setEvaluation(null);
      setUploadTask("");
      setSelectedRow(null);
      onError(error instanceof Error ? error.message : "Could not parse uploaded file.");
    }
  }

  const sortedRows = [...rows].sort((a, b) => {
    const key = sortBy === "baseline" ? "baseline_score" : "nash_score";
    return b[key] - a[key];
  });
  const summary = summarizeBatch(rows);
  const effectiveTask = uploadTask || rows[0]?.task || pairs[0]?.task || "";
  const unitLabel = isGroupedUploadTask(effectiveTask) ? "Groups" : "Pairs";
  const unitCount = uploadUnitCount(rows.length ? rows : pairs, effectiveTask);

  return (
    <div className="grid gap-4">
      <Panel title="Batch Upload" icon={<Upload size={16} />}>
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="grid gap-2">
            <label className="inline-flex h-10 w-fit cursor-pointer items-center justify-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-semibold text-white hover:bg-slate-800">
              <Upload size={16} />
              Upload JSON
              <input className="hidden" type="file" accept=".json,application/json" onChange={uploadBatch} />
            </label>
            <p className="text-xs text-slate-600">Upload NASH evaluation result JSON. Row results and triplet/crosspair/listwise summaries are supported.</p>
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <label className="grid gap-1 text-xs font-medium text-slate-600">
              Sort by
              <select className="h-10 rounded-md border border-slate-300 bg-white px-3 text-sm" value={sortBy} onChange={(event: { target: { value: string } }) => setSortBy(event.target.value === "baseline" ? "baseline" : "nash")}>
                <option value="nash">NASH</option>
                <option value="baseline">baseline</option>
              </select>
            </label>
          </div>
        </div>
      </Panel>

      {evaluation && (
        <Panel title="Evaluation Summary" icon={<BarChart3 size={16} />}>
          <div className="grid gap-2 md:grid-cols-6">
            <MiniMetric label="Task" value={evaluation.task} />
            <MiniMetric label="Model" value={evaluation.model} />
            <MiniMetric label="Threshold" value={evaluation.threshold} />
            <MiniMetric label="Records" value={evaluation.records} />
            <MiniMetric label={evaluation.metricLabel} value={evaluation.metricValue} />
            <MiniMetric label="Runtime" value={evaluation.runtime} />
          </div>
        </Panel>
      )}

      <Panel title="Batch Model Comparison" icon={<Table2 size={16} />}>
        <div className="mb-3 grid gap-2 md:grid-cols-4">
          <MiniMetric label="Source" value={sourceName || "No upload"} />
          <MiniMetric label={unitLabel} value={String(unitCount)} />
          <MiniMetric label="Avg baseline" value={formatScore(summary.avgBaseline)} />
          <MiniMetric label="Avg NASH" value={formatScore(summary.avgNash)} />
        </div>
        {loading && <Notice tone="info">Scoring uploaded batch across baseline settings...</Notice>}
        <ResultTable rows={sortedRows} onRowClick={setSelectedRow} />
        {rows.length ? <p className="mt-2 text-xs text-slate-500">Click a row to open its NASH visualization.</p> : null}
      </Panel>

      {selectedRow && (
        <UploadVisualizationModal
          row={selectedRow}
          rows={rows}
          task={uploadTask}
          traces={traces}
          onClose={() => setSelectedRow(null)}
        />
      )}
    </div>
  );
}

function InteractiveExampleTab({
  models,
  selectedModelIds,
  onSelectedModelIds,
  initialSentenceA,
  initialSentenceB,
  onError
}: {
  models: ModelOption[];
  selectedModelIds: string[];
  onSelectedModelIds: (ids: string[]) => void;
  initialSentenceA: string;
  initialSentenceB: string;
  onError: (message: string) => void;
}) {
  const [sentenceA, setSentenceA] = useState(initialSentenceA);
  const [sentenceB, setSentenceB] = useState(initialSentenceB);
  const threshold = 0.3763;
  const [comparison, setComparison] = useState<BaselineComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function scoreToy() {
    setLoading(true);
    onError("");
    try {
      const response = await fetchBaselineComparison({
        sentence1: sentenceA,
        sentence2: sentenceB,
        staticMode: false,
        threshold,
        baselines: selectedModels(models, selectedModelIds)
      });
      setComparison(response);
    } catch (error) {
      onError(error instanceof Error ? error.message : "Toy scoring failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-4">
      <Panel title="Interactive Example: Single-Pair Analysis" icon={<SlidersHorizontal size={16} />}>
        <div className="grid gap-3 lg:grid-cols-2">
          <SentenceBox label="Sentence A" value={sentenceA} onChange={setSentenceA} />
          <SentenceBox label="Sentence B" value={sentenceB} onChange={setSentenceB} />
        </div>
        <div className="mt-3 flex flex-wrap items-end justify-between gap-3">
          <div className="flex flex-wrap items-end gap-3">
            <MiniMetric label="Alignment threshold" value={formatScore(threshold)} />
            <ModelSelector models={models} selectedIds={selectedModelIds} onChange={onSelectedModelIds} />
          </div>
          <button className="inline-flex h-10 items-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-semibold text-white disabled:bg-slate-400" disabled={loading} onClick={scoreToy}>
            <Activity size={16} className={loading ? "animate-spin" : ""} />
            {loading ? "Scoring..." : "Score pair"}
          </button>
        </div>
      </Panel>

      {comparison && (
        <Panel title="baseline / NASH Comparison" icon={<BarChart3 size={16} />}>
          <ResultTable rows={comparisonToRows(comparison)} />
        </Panel>
      )}

      <TraceVisualizationPanel traces={comparison?.traces ?? []} title="NASH Visualization" />
    </div>
  );
}

function CasesTab() {
  return (
    <div className="grid gap-4">
      <SentencePairCasesSection cases={generatedCases.sentence_pair_cases} />
      <TripletCasesSection cases={generatedCases.triplet_cases} />
      <CrossPairCasesSection cases={generatedCases.cross_pair_cases} />
    </div>
  );
}

function SentencePairCasesSection({ cases }: { cases: GeneratedSentencePairCase[] }) {
  const [caseId, setCaseId] = useState(cases[0]?.id ?? "");
  const selectedCase = cases.find((item) => item.id === caseId) ?? cases[0];
  const [modelKey, setModelKey] = useState(firstModelKey(selectedCase));
  const result = selectedCase?.model_results[modelKey] ?? firstModelResult(selectedCase);

  function selectCase(id: string) {
    const next = cases.find((item) => item.id === id);
    setCaseId(id);
    setModelKey(firstModelKey(next));
  }

  if (!selectedCase || !result) return null;
  return (
    <Panel title="Sentence Pair Cases" icon={<Layers3 size={16} />}>
      <CaseControls
        caseLabel="Case"
        caseValue={selectedCase.id}
        cases={cases.map((item) => ({ id: item.id, title: item.title }))}
        onCaseChange={selectCase}
        modelValue={modelKey}
        modelOptions={modelOptionsFor(selectedCase.model_results)}
        onModelChange={setModelKey}
      />
      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <ReadOnlySentence label="Sentence 1" value={selectedCase.sentence1} />
        <ReadOnlySentence label="Sentence 2" value={selectedCase.sentence2} />
      </div>
      <GeneratedPairBlock result={result} />
    </Panel>
  );
}

function TripletCasesSection({ cases }: { cases: GeneratedTripletCase[] }) {
  const [caseId, setCaseId] = useState(cases[0]?.id ?? "");
  const selectedCase = cases.find((item) => item.id === caseId) ?? cases[0];
  const [modelKey, setModelKey] = useState(firstModelKey(selectedCase));
  const result = selectedCase?.model_results[modelKey] ?? firstModelResult(selectedCase);

  function selectCase(id: string) {
    const next = cases.find((item) => item.id === id);
    setCaseId(id);
    setModelKey(firstModelKey(next));
  }

  if (!selectedCase || !result) return null;
  return (
    <Panel title="Triplet Cases" icon={<FlaskConical size={16} />}>
      <CaseControls
        caseLabel="Case"
        caseValue={selectedCase.id}
        cases={cases.map((item) => ({ id: item.id, title: item.title }))}
        onCaseChange={selectCase}
        modelValue={modelKey}
        modelOptions={modelOptionsFor(selectedCase.model_results)}
        onModelChange={setModelKey}
      />
      <div className="mt-3 grid gap-3 lg:grid-cols-3">
        <ReadOnlySentence label="Anchor" value={selectedCase.anchor} />
        <ReadOnlySentence label="S1" value={selectedCase.positive} />
        <ReadOnlySentence label="S2" value={selectedCase.negative} />
      </div>
      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <GeneratedPairBlock title="Anchor vs S1" result={result.positive_pair} compact />
        <GeneratedPairBlock title="Anchor vs S2" result={result.negative_pair} compact />
      </div>
    </Panel>
  );
}

function CrossPairCasesSection({ cases }: { cases: GeneratedCrossPairCase[] }) {
  const [caseId, setCaseId] = useState(cases[0]?.id ?? "");
  const selectedCase = cases.find((item) => item.id === caseId) ?? cases[0];
  const [modelKey, setModelKey] = useState(firstModelKey(selectedCase));
  const result = selectedCase?.model_results[modelKey] ?? firstModelResult(selectedCase);

  function selectCase(id: string) {
    const next = cases.find((item) => item.id === id);
    setCaseId(id);
    setModelKey(firstModelKey(next));
  }

  if (!selectedCase || !result) return null;
  return (
    <Panel title="Cross-pair Cases" icon={<ArrowRight size={16} />}>
      <CaseControls
        caseLabel="Case"
        caseValue={selectedCase.id}
        cases={cases.map((item) => ({ id: item.id, title: item.title }))}
        onCaseChange={selectCase}
        modelValue={modelKey}
        modelOptions={modelOptionsFor(selectedCase.model_results)}
        onModelChange={setModelKey}
      />
      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <div className="grid gap-2">
          <ReadOnlySentence label="Pair A sentence 1" value={selectedCase.pair_a.sentence1} />
          <ReadOnlySentence label="Pair A sentence 2" value={selectedCase.pair_a.sentence2} />
        </div>
        <div className="grid gap-2">
          <ReadOnlySentence label="Pair B sentence 1" value={selectedCase.pair_b.sentence1} />
          <ReadOnlySentence label="Pair B sentence 2" value={selectedCase.pair_b.sentence2} />
        </div>
      </div>
      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <GeneratedPairBlock title="Pair A" result={result.pair_a} compact />
        <GeneratedPairBlock title="Pair B" result={result.pair_b} compact />
      </div>
    </Panel>
  );
}

function CaseControls({
  caseLabel,
  caseValue,
  cases,
  onCaseChange,
  modelValue,
  modelOptions,
  onModelChange
}: {
  caseLabel: string;
  caseValue: string;
  cases: Array<{ id: string; title: string }>;
  onCaseChange: (id: string) => void;
  modelValue: string;
  modelOptions: Array<{ key: string; label: string }>;
  onModelChange: (key: string) => void;
}) {
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <label className="grid gap-1 text-xs font-medium text-slate-600">
        {caseLabel}
        <select className="h-10 rounded-md border border-slate-300 bg-white px-3 text-sm" value={caseValue} onChange={(event: { target: { value: string } }) => onCaseChange(event.target.value)}>
          {cases.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
        </select>
      </label>
      <label className="grid gap-1 text-xs font-medium text-slate-600">
        Model
        <select className="h-10 rounded-md border border-slate-300 bg-white px-3 text-sm" value={modelValue} onChange={(event: { target: { value: string } }) => onModelChange(event.target.value)}>
          {modelOptions.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
        </select>
      </label>
    </div>
  );
}

function GeneratedPairBlock({ title, result, compact = false }: { title?: string; result: GeneratedPairResult; compact?: boolean }) {
  return (
    <div className={compact ? "rounded-md border border-slate-200 p-3" : "mt-3 grid gap-3"}>
      {title && <div className="mb-3 text-sm font-semibold text-slate-900">{title}</div>}
      {(result.masked_sentence1 || result.masked_sentence2) && (
        <div className="mb-3 grid gap-2 md:grid-cols-2">
          {result.masked_sentence1 && <CodeLine label="Masked 1" value={result.masked_sentence1} />}
          {result.masked_sentence2 && <CodeLine label="Masked 2" value={result.masked_sentence2} />}
        </div>
      )}
      <GeneratedScoreGrid result={result} />
      <div className={`mt-3 grid gap-3 ${compact ? "" : "xl:grid-cols-2"}`}>
        <GeneratedAlignmentTable result={result} />
        <GeneratedAlignmentHeatmap result={result} />
      </div>
    </div>
  );
}

function GeneratedScoreGrid({ result }: { result: GeneratedPairResult }) {
  return (
    <div className="grid gap-2 md:grid-cols-4">
      <MiniMetric label="Baseline" value={formatScore(result.baseline_score)} />
      <MiniMetric label="NASH" value={formatScore(result.nash_score)} />
      <MiniMetric label="Text" value={formatScore(result.text_score)} />
      <MiniMetric label="Numeric" value={formatScore(result.numeric_score)} />
    </div>
  );
}

function GeneratedAlignmentTable({ result }: { result: GeneratedPairResult }) {
  const numbers1 = result.numbers_sentence1 ?? [];
  const numbers2 = result.numbers_sentence2 ?? [];
  const edges = result.alignment?.selected_alignment_edges ?? [];
  return (
    <div className="grid gap-3">
      <div className="grid gap-2 md:grid-cols-2">
        <GeneratedNumberList title="Sentence 1 numbers" numbers={numbers1} />
        <GeneratedNumberList title="Sentence 2 numbers" numbers={numbers2} />
      </div>
      <div className="overflow-x-auto rounded-md border border-slate-200">
        <table className="w-full min-w-[560px] text-left text-xs">
          <thead className="bg-slate-100 text-slate-600">
            <tr>
              <th className="px-2 py-2">Direction</th>
              <th className="px-2 py-2">Source</th>
              <th className="px-2 py-2">Target</th>
              <th className="px-2 py-2 text-right">Alignment</th>
              <th className="px-2 py-2">Selected</th>
            </tr>
          </thead>
          <tbody>
            {edges.length ? edges.map((edge, index) => (
              <tr key={`${edge.direction}-${edge.source_index}-${edge.target_index}-${index}`} className="border-t border-slate-100">
                <td className="px-2 py-2">{directionLabel(edge.direction)}</td>
                <td className="px-2 py-2 font-mono">{edge.source?.raw ?? String(edge.source_index)}</td>
                <td className="px-2 py-2 font-mono">{edge.target?.raw ?? String(edge.target_index)}</td>
                <td className="px-2 py-2 text-right font-mono">{formatScore(edge.context_similarity)}</td>
                <td className="px-2 py-2"><AlignmentState aligned={edge.aligned} /></td>
              </tr>
            )) : (
              <tr>
                <td className="px-2 py-3 text-slate-500" colSpan={5}>No selected alignment edges</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GeneratedAlignmentHeatmap({ result }: { result: GeneratedPairResult | UploadPairResult }) {
  const rows = result.numbers_sentence1 ?? [];
  const columns = result.numbers_sentence2 ?? [];
  const matrix = result.alignment?.matrix ?? [];
  const forwardEdges = new Map<string, GeneratedAlignmentEdge>();
  (result.alignment?.selected_alignment_edges ?? [])
    .forEach((edge) => {
      const key = edge.direction === "sentence2_to_sentence1"
        ? `${edge.target_index}-${edge.source_index}`
        : `${edge.source_index}-${edge.target_index}`;
      const existing = forwardEdges.get(key);
      if (!existing || (!existing.aligned && edge.aligned)) forwardEdges.set(key, edge);
    });

  return (
    <div className="overflow-x-auto">
      <div className="mb-2 text-xs font-semibold uppercase text-slate-500">Matrix</div>
      {rows.length && columns.length ? (
        <div className="grid min-w-max gap-1" style={{ gridTemplateColumns: `96px repeat(${Math.max(1, columns.length)}, minmax(92px, 1fr))` }}>
          <div />
          {columns.map((column) => <div key={`${column.index}-${column.raw}`} className="rounded bg-slate-100 px-2 py-2 text-center font-mono text-xs font-semibold">{column.raw}</div>)}
          {rows.map((row, rowIndex) => (
            <div key={`${row.index}-${row.raw}`} className="contents">
              <div className="rounded bg-slate-100 px-2 py-2 font-mono text-xs font-semibold">{row.raw}</div>
              {columns.map((column, colIndex) => {
                const value = matrix[rowIndex]?.[colIndex] ?? 0;
                const edge = forwardEdges.get(`${rowIndex}-${colIndex}`);
                const edgeClass = edge ? edge.aligned ? "border-teal-700 ring-2 ring-teal-400" : "border-slate-500 ring-2 ring-slate-300" : "border-slate-200";
                return (
                  <div
                    key={`${row.index}-${column.index}`}
                    className={`rounded border px-2 py-3 text-center font-mono text-xs font-semibold ${edgeClass}`}
                    style={{ backgroundColor: heatColor(value) }}
                  >
                    <div>{formatScore(value)}</div>
                    {edge && <div className="mt-1 text-[10px] uppercase text-slate-800">{edge.aligned ? "aligned" : "not aligned"}</div>}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      ) : (
        <Notice tone="info">No numeric mentions were available for an alignment matrix.</Notice>
      )}
    </div>
  );
}

function GeneratedNumberList({ title, numbers }: { title: string; numbers: GeneratedNumber[] }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-2">
      <div className="mb-2 text-xs font-semibold uppercase text-slate-500">{title}</div>
      <div className="grid gap-1">
        {numbers.length ? numbers.map((number) => (
          <div key={`${number.index}-${number.raw}-${number.start}`} className="rounded bg-white px-2 py-1 text-xs">
            <span className="font-mono font-semibold">{number.raw}</span>
          </div>
        )) : <span className="text-xs text-slate-500">No numeric mentions</span>}
      </div>
    </div>
  );
}

function AlignmentState({ aligned }: { aligned: boolean }) {
  const className = aligned ? "bg-teal-100 text-teal-900" : "bg-slate-100 text-slate-700";
  return <span className={`rounded px-2 py-1 text-xs font-semibold uppercase ${className}`}>{aligned ? "aligned" : "not aligned"}</span>;
}

function firstModelKey(caseItem?: { model_results: Record<string, unknown> }) {
  return caseItem ? Object.keys(caseItem.model_results)[0] ?? "" : "";
}

function firstModelResult<T extends { model_results: Record<string, unknown> }>(caseItem?: T) {
  return caseItem ? Object.values(caseItem.model_results)[0] as any : undefined;
}

function modelOptionsFor(results: Record<string, unknown>) {
  return Object.entries(results).map(([key, value]) => {
    const result = value as { display_name?: string; positive_pair?: GeneratedPairResult; pair_a?: GeneratedPairResult };
    return {
      key,
      label: result.display_name ?? result.positive_pair?.display_name ?? result.pair_a?.display_name ?? key
    };
  });
}

function directionLabel(direction: string) {
  if (direction === "sentence1_to_sentence2") return "S1 to S2";
  if (direction === "sentence2_to_sentence1") return "S2 to S1";
  return direction.replace(/_/g, " ");
}

function TabButton({ id, active, label, icon, onClick }: { id: TabId; active: TabId; label: string; icon: ReactNode; onClick: (id: TabId) => void }) {
  const selected = id === active;
  return (
    <button className={`inline-flex h-10 items-center justify-center gap-2 rounded px-3 text-sm font-semibold ${selected ? "bg-slate-950 text-white" : "text-slate-700 hover:bg-slate-100"}`} onClick={() => onClick(id)}>
      {icon}
      {label}
    </button>
  );
}

function Panel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-md border border-slate-200 bg-white shadow-panel">
      <div className="flex min-h-10 items-center gap-2 border-b border-slate-200 px-3 py-2 text-sm font-semibold text-slate-900">
        <span className="text-teal-700">{icon}</span>
        {title}
      </div>
      <div className="p-3">{children}</div>
    </section>
  );
}

function ModelSelector({ models, selectedIds, onChange }: { models: ModelOption[]; selectedIds: string[]; onChange: (ids: string[]) => void }) {
  const visible = models.filter((model) => defaultModelIds.includes(model.id));
  return (
    <div className="grid gap-1 text-xs font-medium text-slate-600">
      Baselines
      <div className="flex flex-wrap gap-2">
        {visible.map((model) => {
          const checked = selectedIds.includes(model.id);
          return (
            <label key={model.id} className={`inline-flex h-10 items-center gap-2 rounded-md border px-3 text-sm ${checked ? "border-teal-300 bg-teal-50 text-teal-950" : "border-slate-300 bg-white text-slate-700"}`}>
              <input
                type="checkbox"
                checked={checked}
                onChange={() => {
                  const next = checked ? selectedIds.filter((id) => id !== model.id) : [...selectedIds, model.id];
                  onChange(next.length ? next : [model.id]);
                }}
              />
              {model.label}
            </label>
          );
        })}
      </div>
    </div>
  );
}

function ResultTable({ rows, onRowClick }: { rows: BatchResultRow[]; onRowClick?: (row: BatchResultRow) => void }) {
  return (
    <div className="overflow-x-auto rounded-md border border-slate-200">
      <table className="w-full min-w-[920px] text-left text-sm">
        <thead className="bg-slate-100 text-xs uppercase text-slate-500">
          <tr>
            <th className="px-3 py-2">Pair</th>
            <th className="px-3 py-2">Baseline</th>
            <th className="px-3 py-2 text-right">baseline</th>
            <th className="px-3 py-2 text-right">NASH</th>
            <th className="px-3 py-2 text-right">Text</th>
            <th className="px-3 py-2 text-right">Numeric</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={`${row.id}-${row.baseline_id}-${index}`}
              className={`border-t border-slate-100 ${onRowClick ? "cursor-pointer hover:bg-teal-50" : ""}`}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
            >
              <td className="px-3 py-2">
                <div className="font-semibold">{row.title}</div>
                <div className="max-w-[420px] truncate text-xs text-slate-500">{sentencePreview(row)}</div>
              </td>
              <td className="px-3 py-2">{row.baseline_label}</td>
              <td className="px-3 py-2 text-right font-mono">{formatScore(row.baseline_score)}</td>
              <td className="px-3 py-2 text-right font-mono font-semibold">{formatScore(row.nash_score)}</td>
              <td className="px-3 py-2 text-right font-mono">{formatScore(row.text_score)}</td>
              <td className="px-3 py-2 text-right font-mono">{formatScore(row.numeric_score)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SentenceBox({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="grid gap-1 text-xs font-medium text-slate-600">
      {label}
      <textarea className="min-h-24 resize-y rounded-md border border-slate-300 bg-white p-3 text-sm leading-6 text-slate-950 shadow-panel focus:border-teal-600 focus:outline-none" value={value} onChange={(event: { target: { value: string } }) => onChange(event.target.value)} />
    </label>
  );
}

function ReadOnlySentence({ label, value, compact = false }: { label: string; value: string; compact?: boolean }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="mb-1 text-xs font-semibold uppercase text-slate-500">{label}</div>
      <div className="text-sm leading-6">{compact ? truncateText(value, 130) : value}</div>
    </div>
  );
}

function TraceVisualizationPanel({ traces, title }: { traces: NashTrace[]; title: string }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const trace = traces[selectedIndex] ?? traces[0];

  useEffect(() => {
    if (selectedIndex >= traces.length) setSelectedIndex(0);
  }, [selectedIndex, traces.length]);

  if (!trace) {
    return (
      <Panel title={title} icon={<Table2 size={16} />}>
        <Notice tone="info">Score a pair to show the NASH numerical alignment matrix.</Notice>
      </Panel>
    );
  }

  const sentence1Numbers = trace.numeric_extraction.sentence1_numbers;
  const sentence2Numbers = trace.numeric_extraction.sentence2_numbers;
  const columns = sentence2Numbers.map((number) => number.text);
  const rows = sentence1Numbers.map((number) => number.text);
  const matrix = trace.numeric_alignment.similarity_matrix;
  const tau = trace.numeric_alignment.threshold;
  const textMass = trace.aggregation.idf_text_mass;
  const numericMass = trace.aggregation.idf_numeric_mass;
  const alpha = textMass / Math.max(textMass + numericMass, 1e-12);
  const numericWeight = 1 - alpha;
  const forward = trace.numeric_alignment.alignments_s1_to_s2;
  const reverse = trace.numeric_alignment.alignments_s2_to_s1;
  const acceptedCount = [...forward, ...reverse].filter((alignment) => alignment.valid).length;
  const rejectedCount = [...forward, ...reverse].filter((alignment) => !alignment.valid).length;
  const forwardCells = new Set(
    trace.numeric_alignment.valid_alignments_s1_to_s2
      .filter((alignment) => alignment.valid)
      .map((alignment) => `${alignment.source_index}-${alignment.target_index}`)
  );
  const forwardArgmax = new Map(trace.numeric_alignment.alignments_s1_to_s2.map((alignment) => [`${alignment.source_index}-${alignment.target_index}`, alignment]));

  return (
    <Panel title={title} icon={<Table2 size={16} />}>
      <div className="mb-4 grid gap-3 lg:grid-cols-[1fr_320px]">
        <div className="grid gap-2 md:grid-cols-2">
          <ReadOnlySentence label="Sentence A" value={trace.input.sentence1} />
          <ReadOnlySentence label="Sentence B" value={trace.input.sentence2} />
        </div>
        <div className="rounded-lg border-2 border-slate-900 bg-slate-950 p-3 text-white shadow-panel">
          {traces.length > 1 && (
            <label className="grid gap-1 text-xs font-medium text-slate-600">
              Trace
              <select className="h-10 rounded-md border border-slate-300 bg-white px-3 text-sm" value={selectedIndex} onChange={(event: { target: { value: string } }) => setSelectedIndex(Number(event.target.value))}>
                {traces.map((item, index) => {
                  const label = String(item.metadata?.baseline_label ?? item.metadata?.baseline_id ?? `baseline ${index + 1}`);
                  return <option key={`${label}-${index}`} value={index}>{label}</option>;
                })}
              </select>
            </label>
          )}
          <div className="mb-2 text-xs font-bold uppercase tracking-wider text-teal-300">Final score audit</div>
          <div className="grid grid-cols-2 gap-2">
            <AuditMetric label="Baseline score" value={trace.baseline_comparison.baseline_score} />
            <AuditMetric label="Masked-text score" value={trace.textual_similarity.score} />
            <AuditMetric label="Final numeric score" value={trace.numeric_similarity.score} accent />
            <AuditMetric label="Final NASH score" value={trace.aggregation.final_score} accent />
          </div>
        </div>
      </div>

      <div className="mb-4 rounded-lg border border-teal-200 bg-teal-50 p-3">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div><span className="text-sm font-bold text-teal-950">Numerical alignment</span><span className="ml-2 text-xs text-teal-800">Contextual cosine matrix · directional argmax · τ gating</span></div>
          <div className="rounded-full bg-teal-700 px-3 py-1 font-mono text-xs font-bold text-white">Active τ = {formatScore(tau)}</div>
        </div>
        <div className="flex flex-wrap gap-2 text-xs"><LegendChip tone="accepted" label={`Accepted ${acceptedCount}`} /><LegendChip tone="rejected" label={`Rejected by τ ${rejectedCount}`} /><LegendChip tone="neutral" label="Cells = contextual cosine similarity" /></div>
      </div>

      <div className="mb-4 grid gap-3 md:grid-cols-2">
        <TraceNumberList title="Extracted Numbers — Sentence A" numbers={sentence1Numbers} />
        <TraceNumberList title="Extracted Numbers — Sentence B" numbers={sentence2Numbers} />
      </div>

      <div className="mb-2 flex items-end justify-between gap-3"><div><h3 className="text-sm font-bold text-slate-950">Alignment Matrix: Sentence A → Sentence B</h3><p className="text-xs text-slate-600">Rows are extracted numbers from Sentence A. Columns are extracted numbers from Sentence B. Selected argmax cells are outlined; red cells are selected then rejected by τ.</p></div><span className="whitespace-nowrap text-xs font-semibold text-slate-600">τ = {formatScore(tau)}</span></div>

      {rows.length && columns.length ? (
        <div className="overflow-x-auto">
          <div className="grid min-w-max gap-1" style={{ gridTemplateColumns: `112px repeat(${Math.max(1, columns.length)}, minmax(92px, 1fr))` }}>
            <div className="rounded bg-slate-900 px-2 py-2 text-center text-[10px] font-bold uppercase text-white">A ↓ / B →</div>
            {columns.map((column, index) => <div key={`${column}-${index}`} className="rounded bg-slate-900 px-2 py-2 text-center font-mono text-xs font-semibold text-white"><div className="text-[10px] font-sans uppercase text-slate-300">B{index + 1}</div>{column}</div>)}
            {rows.map((row, rowIndex) => (
              <div key={`${row}-${rowIndex}`} className="contents">
                <div className="rounded bg-slate-900 px-2 py-2 font-mono text-xs font-semibold text-white"><span className="mr-1 font-sans text-[10px] text-slate-300">A{rowIndex + 1}</span>{row}</div>
                {columns.map((column, colIndex) => {
                  const value = matrix[rowIndex]?.[colIndex] ?? 0;
                  const key = `${rowIndex}-${colIndex}`;
                  const aligned = forwardCells.has(key);
                  const selection = forwardArgmax.get(key);
                  const cellTitle = `${rows[rowIndex]} → ${columns[colIndex]}\ncontextual cosine: ${formatScore(value)}\ndirectional argmax: ${selection ? "yes" : "no"}\npasses τ: ${selection?.valid ? "yes" : "no"}${selection ? `\nmagnitude similarity: ${selection.valid ? formatScore(selection.magnitude_similarity) : "does not contribute"}` : ""}`;
                  return (
                    <div
                      key={`${row}-${column}-${colIndex}`}
                      className={`rounded border px-2 py-2 text-center font-mono text-xs font-semibold ${aligned ? "border-teal-800 ring-2 ring-teal-500" : selection ? "border-rose-700 ring-2 ring-rose-300" : "border-slate-200"}`}
                      style={{ backgroundColor: heatColor(value) }}
                      title={cellTitle}
                    >
                      <div>{formatScore(value)}</div>
                      {aligned && <div className="mt-1 text-[10px] uppercase text-teal-950">A→B accepted</div>}
                      {selection && !selection.valid && <div className="mt-1 text-[10px] uppercase text-rose-950">A→B rejected</div>}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <Notice tone="info">No numeric mentions were available for an alignment matrix.</Notice>
      )}
      <AlignmentDetails trace={trace} />
      <div className="mt-4 rounded-lg border-2 border-indigo-200 bg-indigo-50 p-4">
        <div className="mb-3 text-sm font-bold text-indigo-950">Final NASH score decomposition</div>
        <div className="grid gap-2 sm:grid-cols-3"><AuditTile label="A → B numeric score" value={trace.numeric_similarity.directional_s1_to_s2} /><AuditTile label="B → A numeric score" value={trace.numeric_similarity.directional_s2_to_s1} /><AuditTile label="Bidirectional numeric score" value={trace.numeric_similarity.score} /></div>
        <div className="my-3 grid gap-2 sm:grid-cols-2"><AuditTile label="α (text weight)" value={alpha} /><AuditTile label="1 − α (numeric weight)" value={numericWeight} /></div>
        <div className="rounded-md bg-indigo-950 p-3 font-mono text-xs leading-6 text-indigo-50">NASH = α × text_score + (1 − α) × numeric_score<br />= {formatScore(alpha)} × {formatScore(trace.textual_similarity.score)} + {formatScore(numericWeight)} × {formatScore(trace.numeric_similarity.score)}<br />= <strong className="text-teal-300">{formatScore(trace.aggregation.final_score)}</strong></div>
      </div>
    </Panel>
  );
}

function AlignmentDetails({ trace }: { trace: NashTrace }) {
  const directions = [["A → B", trace.numeric_alignment.alignments_s1_to_s2], ["B → A", trace.numeric_alignment.alignments_s2_to_s1]] as const;
  return <div className="mt-5 overflow-x-auto rounded-lg border border-slate-300 bg-white"><div className="border-b border-slate-200 bg-slate-100 px-4 py-3"><div className="text-sm font-bold text-slate-950">Selected Alignment Details</div><div className="text-xs text-slate-600">One directional argmax row per source mention. Rejected rows remain visible and contribute zero.</div></div><table className="min-w-full text-sm"><thead className="bg-slate-900 text-left text-xs uppercase tracking-wide text-white"><tr><th className="px-3 py-2">Direction</th><th className="px-3 py-2">Source number</th><th className="px-3 py-2">Matched target number</th><th className="px-3 py-2">Contextual similarity</th><th className="px-3 py-2">Status at τ={formatScore(trace.numeric_alignment.threshold)}</th><th className="px-3 py-2">Magnitude score</th></tr></thead><tbody>{directions.flatMap(([direction, items]) => items.map((item, index) => <tr key={`${direction}-${index}`} className="border-b border-slate-200 last:border-0"><td className="px-3 py-3 font-semibold">{direction}</td><td className="px-3 py-3 font-mono">{item.source_text}</td><td className="px-3 py-3 font-mono">{item.target_text}</td><td className="px-3 py-3 font-mono">{formatScore(item.contextual_similarity)}</td><td className="px-3 py-3"><StatusBadge accepted={item.valid} /></td><td className="px-3 py-3 font-mono font-bold">{item.valid ? formatScore(item.magnitude_similarity) : "0.000 (rejected)"}</td></tr>))}</tbody></table></div>;
}

function AuditMetric({ label, value, accent = false }: { label: string; value: number; accent?: boolean }) {
  return <div className={`rounded border p-2 ${accent ? "border-teal-400 bg-teal-500/15" : "border-slate-700 bg-slate-800"}`}><div className="text-[10px] font-bold uppercase tracking-wide text-slate-300">{label}</div><div className={`mt-1 font-mono text-lg font-bold ${accent ? "text-teal-300" : "text-white"}`}>{formatScore(value)}</div></div>;
}

function AuditTile({ label, value }: { label: string; value: number }) {
  return <div className="rounded-md border border-indigo-200 bg-white px-3 py-2"><div className="text-[10px] font-bold uppercase tracking-wide text-indigo-700">{label}</div><div className="mt-1 font-mono text-base font-bold text-indigo-950">{formatScore(value)}</div></div>;
}

function LegendChip({ tone, label }: { tone: "accepted" | "rejected" | "neutral"; label: string }) {
  const colors = { accepted: "bg-teal-100 text-teal-900 ring-teal-300", rejected: "bg-rose-100 text-rose-900 ring-rose-300", neutral: "bg-white text-slate-700 ring-slate-300" }[tone];
  return <span className={`rounded-full px-2 py-1 text-xs font-semibold ring-1 ${colors}`}>{label}</span>;
}

function StatusBadge({ accepted }: { accepted: boolean }) {
  return accepted ? <span className="rounded-full bg-teal-100 px-2 py-1 text-xs font-bold text-teal-900">Accepted · contributes</span> : <span className="rounded-full bg-rose-100 px-2 py-1 text-xs font-bold text-rose-900">Rejected by τ · contributes 0</span>;
}

function EvaluationResultVisualizationPanel({ rows, task }: { rows: BatchResultRow[]; task: string }) {
  const normalizedTask = task.toLowerCase();
  if (normalizedTask === "triplet") return <TripletUploadVisualization rows={rows} />;
  if (normalizedTask === "crosspair") return <CrosspairUploadVisualization rows={rows} />;
  if (normalizedTask === "listwise") return <ListwiseUploadVisualization rows={rows} />;
  return <GenericUploadVisualization rows={rows} />;
}

function TripletUploadVisualization({ rows }: { rows: BatchResultRow[] }) {
  const groups = groupRows(rows, (row) => uploadVisualizationGroupKey(row, "triplet"));
  return (
    <Panel title="NASH Visualization: Triplet" icon={<FlaskConical size={16} />}>
      <div className="grid gap-3">
        {groups.map((group) => {
          const taskRow = group.rows.find((row) => row.pair_results?.length);
          if (taskRow?.pair_results?.length) {
            const s1 = taskRow.pair_results.find((result) => /^s1$/i.test(result.label)) ?? taskRow.pair_results[0];
            const s2 = taskRow.pair_results.find((result) => /^s2$/i.test(result.label)) ?? taskRow.pair_results[1];
            return (
              <div key={group.key} className="rounded-md border border-slate-200 p-3">
                <div className="mb-2 text-sm font-semibold">{taskRow.title}</div>
                <div className="mb-3 grid gap-2 md:grid-cols-3">
                  <ReadOnlySentence label="Anchor" value={s1?.sentence1 ?? taskRow.sentence1} compact />
                  <ReadOnlySentence label="S1" value={s1?.sentence2 ?? ""} compact />
                  <ReadOnlySentence label="S2" value={s2?.sentence2 ?? ""} compact />
                </div>
                <div className="grid gap-3 xl:grid-cols-2">
                  {s1 && <UploadPairResultBlock result={s1} tone="teal" showSentences={false} showMasked={false} />}
                  {s2 && <UploadPairResultBlock result={s2} tone="rose" showSentences={false} showMasked={false} />}
                </div>
              </div>
            );
          }
          const s1 = group.rows.find(isTripletS1Row);
          const s2 = group.rows.find(isTripletS2Row);
          return (
            <div key={group.key} className="rounded-md border border-slate-200 p-3">
              <div className="mb-2 text-sm font-semibold">{tripletTitle(s1, s2, group.key)}</div>
              <div className="mb-3 grid gap-2 md:grid-cols-3">
                <ReadOnlySentence label="Anchor" value={s1?.sentence1 ?? s2?.sentence1 ?? ""} compact />
                <ReadOnlySentence label="S1" value={s1?.sentence2 ?? ""} compact />
                <ReadOnlySentence label="S2" value={s2?.sentence2 ?? ""} compact />
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                <UploadScoreCard label="S1" row={s1} tone="teal" />
                <UploadScoreCard label="S2" row={s2} tone="rose" />
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

function CrosspairUploadVisualization({ rows }: { rows: BatchResultRow[] }) {
  const groups = groupRows(rows, (row) => uploadVisualizationGroupKey(row, "crosspair"));
  return (
    <Panel title="NASH Visualization: Crosspair" icon={<ArrowRight size={16} />}>
      <div className="grid gap-3">
        {groups.map((group) => {
          const taskRow = group.rows.find((row) => row.pair_results?.length);
          if (taskRow?.pair_results?.length) {
            const pairA = taskRow.pair_results.find((result) => /pair a/i.test(result.label)) ?? taskRow.pair_results[0];
            const pairB = taskRow.pair_results.find((result) => /pair b/i.test(result.label)) ?? taskRow.pair_results[1];
            return (
              <div key={group.key} className="rounded-md border border-slate-200 p-3">
                <div className="mb-2 text-sm font-semibold">{taskRow.title}</div>
                <div className="mb-3 grid gap-2 md:grid-cols-2">
                  {pairA && (
                    <div className="grid gap-2">
                      <ReadOnlySentence label="Pair A sentence 1" value={pairA.sentence1} compact />
                      <ReadOnlySentence label="Pair A sentence 2" value={pairA.sentence2} compact />
                    </div>
                  )}
                  {pairB && (
                    <div className="grid gap-2">
                      <ReadOnlySentence label="Pair B sentence 1" value={pairB.sentence1} compact />
                      <ReadOnlySentence label="Pair B sentence 2" value={pairB.sentence2} compact />
                    </div>
                  )}
                </div>
                <div className="grid gap-3 xl:grid-cols-2">
                  {pairA && <UploadPairResultBlock result={pairA} tone="teal" showSentences={false} showMasked={false} />}
                  {pairB && <UploadPairResultBlock result={pairB} tone="indigo" showSentences={false} showMasked={false} />}
                </div>
              </div>
            );
          }
          const pairA = group.rows.find((row) => /pair_a|_a$|_ab$/i.test(`${row.role ?? ""}_${row.id}`));
          const pairB = group.rows.find((row) => /pair_b|_b$|_cd$/i.test(`${row.role ?? ""}_${row.id}`));
          return (
            <div key={group.key} className="rounded-md border border-slate-200 p-3">
              <div className="mb-2 text-sm font-semibold">{crosspairTitle(pairA, pairB, group.key)}</div>
              <div className="mb-3 grid gap-2 md:grid-cols-2">
                <div className="grid gap-2">
                  <ReadOnlySentence label="Pair A sentence 1" value={pairA?.sentence1 ?? ""} compact />
                  <ReadOnlySentence label="Pair A sentence 2" value={pairA?.sentence2 ?? ""} compact />
                </div>
                <div className="grid gap-2">
                  <ReadOnlySentence label="Pair B sentence 1" value={pairB?.sentence1 ?? ""} compact />
                  <ReadOnlySentence label="Pair B sentence 2" value={pairB?.sentence2 ?? ""} compact />
                </div>
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                <UploadScoreCard label="Pair A" row={pairA} tone="teal" />
                <UploadScoreCard label="Pair B" row={pairB} tone="indigo" />
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

function ListwiseUploadVisualization({ rows }: { rows: BatchResultRow[] }) {
  const groups = groupRows(rows, (row) => row.id.replace(/_candidate_\d+$/i, ""));
  return (
    <Panel title="NASH Visualization: Listwise" icon={<BarChart3 size={16} />}>
      <div className="grid gap-3">
        {groups.map((group) => {
          const taskRow = group.rows.find((row) => row.pair_results?.length);
          if (taskRow?.pair_results?.length) {
            return (
              <div key={group.key} className="rounded-md border border-slate-200 p-3">
                <div className="mb-2 text-sm font-semibold">{taskRow.title}</div>
                <div className="mb-3 grid gap-2">
                  <ReadOnlySentence label="Anchor" value={taskRow.pair_results[0]?.sentence1 ?? taskRow.sentence1} compact />
                  <CandidateList candidates={taskRow.pair_results.map((result) => result.sentence2)} />
                </div>
                <div className="grid gap-2">
                  {taskRow.pair_results.map((result, index) => (
                    <CandidateScoreBar key={`${taskRow.id}-${result.label}-${index}`} label={result.label || `Candidate ${index + 1}`} value={result.nash_score} />
                  ))}
                </div>
              </div>
            );
          }
          return (
            <div key={group.key} className="rounded-md border border-slate-200 p-3">
              <div className="mb-2 text-sm font-semibold">{group.rows[0]?.title.replace(/: candidate \d+$/i, "") || group.key}</div>
              <div className="mb-3 grid gap-2">
                <ReadOnlySentence label="Anchor" value={group.rows[0]?.sentence1 ?? ""} compact />
                <CandidateList candidates={candidateRows(group.rows).map((row) => row.sentence2)} />
              </div>
              <div className="grid gap-2">
                {candidateRows(group.rows).map((row, index) => (
                  <CandidateScoreBar key={row.id} label={`Candidate ${index + 1}`} value={row.nash_score} />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

function GenericUploadVisualization({ rows }: { rows: BatchResultRow[] }) {
  return (
    <Panel title="NASH Visualization: Uploaded Results" icon={<BarChart3 size={16} />}>
      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        {rows.slice(0, 9).map((row) => (
          <div key={`${row.id}-${row.baseline_id}`}>
            <UploadScoreCard label={row.title} row={row} tone="teal" />
          </div>
        ))}
      </div>
    </Panel>
  );
}

function UploadVisualizationModal({
  row,
  rows,
  task,
  traces,
  onClose
}: {
  row: BatchResultRow;
  rows: BatchResultRow[];
  task: string;
  traces: NashTrace[];
  onClose: () => void;
}) {
  const matchingTraces = traces.filter((trace) => traceMatchesRow(trace, row));
  const rowTask = row.task ?? task;
  const relatedRows = relatedRowsForVisualization(row, rows, rowTask);
  const sentenceContext = modalSentenceContext(row, rowTask, relatedRows);
  const groupedTask = isGroupedUploadTask(rowTask);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/40 p-4" onClick={onClose}>
      <div className="max-h-[85vh] w-full max-w-5xl overflow-auto rounded-md bg-white shadow-xl" onClick={(event: { stopPropagation: () => void }) => event.stopPropagation()}>
        <div className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b border-slate-200 bg-white px-3 py-2">
          <div className="min-w-0">
            <div className="text-sm font-semibold text-slate-900">NASH Visualization</div>
            <div className="text-xs text-slate-500">{row.title} / {row.baseline_label}</div>
            <div className="mt-1 grid gap-0.5">
              {sentenceContext.map((item) => (
                <div key={item.label} className="max-w-4xl truncate text-[11px] leading-4 text-slate-500">
                  <span className="font-semibold">{item.label}:</span> {item.text}
                </div>
              ))}
            </div>
          </div>
          <button className="inline-flex h-8 w-8 items-center justify-center rounded border border-slate-300 bg-white text-slate-700 hover:bg-slate-100" aria-label="Close visualization" onClick={onClose}>
            <X size={16} />
          </button>
        </div>
        <div className="p-3">
          <div className="grid gap-3">
            {groupedTask && <EvaluationResultVisualizationPanel rows={relatedRows} task={rowTask} />}
            {matchingTraces.length ? <TraceVisualizationPanel traces={matchingTraces} title={groupedTask ? "Selected Pair Matrix" : "NASH Visualization"} /> : null}
            {!groupedTask && !matchingTraces.length ? <EvaluationResultVisualizationPanel rows={relatedRows} task={rowTask} /> : null}
          </div>
        </div>
      </div>
    </div>
  );
}

function isGroupedUploadTask(task: string) {
  const normalized = task.toLowerCase();
  return normalized === "triplet" || normalized === "crosspair" || normalized === "listwise";
}

function uploadUnitCount(items: Array<{ id: string; task?: string }>, task: string) {
  if (!isGroupedUploadTask(task)) return items.length;
  return new Set(items.map((item) => uploadGroupKeyFromId(item.id, task.toLowerCase()))).size;
}

function modalSentenceContext(row: BatchResultRow, task: string, relatedRows: BatchResultRow[]) {
  const normalizedTask = task.toLowerCase();
  if (normalizedTask === "triplet") {
    const taskRow = relatedRows.find((item) => item.pair_results?.length) ?? row;
    if (taskRow.pair_results?.length) {
      const s1 = taskRow.pair_results.find((result) => /^s1$/i.test(result.label)) ?? taskRow.pair_results[0];
      const s2 = taskRow.pair_results.find((result) => /^s2$/i.test(result.label)) ?? taskRow.pair_results[1];
      return [
        { label: "Anchor", text: s1?.sentence1 ?? taskRow.sentence1 },
        { label: "S1", text: s1?.sentence2 ?? "" },
        { label: "S2", text: s2?.sentence2 ?? "" }
      ];
    }
    const s1 = relatedRows.find(isTripletS1Row);
    const s2 = relatedRows.find(isTripletS2Row);
    return [
      { label: "Anchor", text: s1?.sentence1 ?? s2?.sentence1 ?? row.sentence1 },
      { label: "S1", text: s1?.sentence2 ?? "" },
      { label: "S2", text: s2?.sentence2 ?? "" }
    ];
  }
  if (normalizedTask === "crosspair") {
    const taskRow = relatedRows.find((item) => item.pair_results?.length) ?? row;
    if (taskRow.pair_results?.length) {
      const pairA = taskRow.pair_results.find((result) => /pair a/i.test(result.label)) ?? taskRow.pair_results[0];
      const pairB = taskRow.pair_results.find((result) => /pair b/i.test(result.label)) ?? taskRow.pair_results[1];
      return [
        { label: "Pair A sentence 1", text: pairA?.sentence1 ?? "" },
        { label: "Pair A sentence 2", text: pairA?.sentence2 ?? "" },
        { label: "Pair B sentence 1", text: pairB?.sentence1 ?? "" },
        { label: "Pair B sentence 2", text: pairB?.sentence2 ?? "" }
      ];
    }
    const pairA = relatedRows.find((item) => /pair_a|_a$|_ab$/i.test(`${item.role ?? ""}_${item.id}`));
    const pairB = relatedRows.find((item) => /pair_b|_b$|_cd$/i.test(`${item.role ?? ""}_${item.id}`));
    return [
      { label: "Pair A sentence 1", text: pairA?.sentence1 ?? "" },
      { label: "Pair A sentence 2", text: pairA?.sentence2 ?? "" },
      { label: "Pair B sentence 1", text: pairB?.sentence1 ?? "" },
      { label: "Pair B sentence 2", text: pairB?.sentence2 ?? "" }
    ];
  }
  if (normalizedTask === "listwise") {
    const taskRow = relatedRows.find((item) => item.pair_results?.length) ?? row;
    if (taskRow.pair_results?.length) {
      return [
        { label: "Anchor", text: taskRow.pair_results[0]?.sentence1 ?? taskRow.sentence1 },
        { label: "Candidates", text: taskRow.pair_results.map((item) => item.sentence2).join(" / ") }
      ];
    }
    return [
      { label: "Anchor", text: row.sentence1 },
      { label: "Candidates", text: candidateRows(relatedRows).map((item) => item.sentence2).join(" / ") }
    ];
  }
  return [
    { label: "Sentence A", text: row.sentence1 },
    { label: "Sentence B", text: row.sentence2 }
  ];
}

function traceMatchesRow(trace: NashTrace, row: BatchResultRow) {
  const metadata = trace.metadata ?? {};
  const baseline = String(metadata.baseline_id ?? metadata.baseline_label ?? "");
  const baselineMatches = !baseline || baseline === row.baseline_id || baseline === row.baseline_label;
  if (row.pair_results?.some((result) => trace.input.sentence1 === result.sentence1 && trace.input.sentence2 === result.sentence2)) {
    return baselineMatches;
  }
  if (trace.input.sentence1 !== row.sentence1 || trace.input.sentence2 !== row.sentence2) return false;
  return baselineMatches;
}

function relatedRowsForVisualization(row: BatchResultRow, rows: BatchResultRow[], task: string) {
  const normalizedTask = task.toLowerCase();
  const selectedKey = uploadVisualizationGroupKey(row, normalizedTask);
  const sameBaselineRows = rows.filter((candidate) => candidate.baseline_id === row.baseline_id);
  const related = sameBaselineRows.filter((candidate) => uploadVisualizationGroupKey(candidate, normalizedTask) === selectedKey);
  return related.length ? related : [row];
}

function uploadVisualizationGroupKey(row: BatchResultRow, task: string) {
  return uploadGroupKeyFromId(row.id, task);
}

function uploadGroupKeyFromId(id: string, task: string) {
  if (task === "triplet") return id.replace(/_(s1|s2|positive|negative)$/i, "");
  if (task === "crosspair") return id.replace(/_(a|b|ab|cd)$/i, "");
  if (task === "listwise") return id.replace(/_candidate_\d+$/i, "");
  return id;
}

function UploadScoreCard({ label, row, tone }: { label: string; row?: BatchResultRow; tone: "teal" | "rose" | "indigo" }) {
  const bar = tone === "rose" ? "bg-rose-500" : tone === "indigo" ? "bg-indigo-600" : "bg-teal-600";
  if (!row) {
    return <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-500">{label}: no row</div>;
  }
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-sm font-semibold">{label}</span>
        <span className="font-mono text-sm">{formatScore(row.nash_score)}</span>
      </div>
      <div className="mb-2 h-2 rounded bg-white">
        <div className={`h-2 rounded ${bar}`} style={{ width: `${clamp01(row.nash_score) * 100}%` }} />
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs">
        <MiniMetric label="baseline" value={formatScore(row.baseline_score)} />
        <MiniMetric label="Text" value={formatScore(row.text_score)} />
        <MiniMetric label="Numeric" value={formatScore(row.numeric_score)} />
      </div>
    </div>
  );
}

function UploadPairResultBlock({ result, tone, showSentences = true, showMasked = true }: { result: UploadPairResult; tone: "teal" | "rose" | "indigo"; showSentences?: boolean; showMasked?: boolean }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
      <div className="mb-2 text-sm font-semibold">{result.label}</div>
      {showSentences && (
        <div className="mb-3 grid gap-2 md:grid-cols-2">
          <ReadOnlySentence label="Sentence 1" value={result.sentence1} compact />
          <ReadOnlySentence label="Sentence 2" value={result.sentence2} compact />
        </div>
      )}
      {showMasked && (result.masked_sentence1 || result.masked_sentence2) && (
        <div className="mb-3 grid gap-2 md:grid-cols-2">
          {result.masked_sentence1 && <CodeLine label="Masked 1" value={result.masked_sentence1} />}
          {result.masked_sentence2 && <CodeLine label="Masked 2" value={result.masked_sentence2} />}
        </div>
      )}
      <div className="mb-3">
        <UploadPairScoreCard result={result} tone={tone} />
      </div>
      <GeneratedAlignmentHeatmap result={result} />
    </div>
  );
}

function CandidateList({ candidates }: { candidates: string[] }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-2">
      <div className="mb-2 text-xs font-semibold uppercase text-slate-500">Candidates</div>
      <div className="grid gap-1">
        {candidates.map((candidate, index) => (
          <div key={`${candidate}-${index}`} className="rounded bg-white px-2 py-1 text-xs leading-5">
            <span className="font-semibold">Candidate {index + 1}:</span> {truncateText(candidate, 160)}
          </div>
        ))}
      </div>
    </div>
  );
}

function CandidateScoreBar({ label, value }: { key?: string; label: string; value: number }) {
  return (
    <div className="grid gap-1">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="font-semibold">{label}</span>
        <span className="font-mono">{formatScore(value)}</span>
      </div>
      <div className="h-2 rounded bg-slate-100">
        <div className="h-2 rounded bg-teal-600" style={{ width: `${clamp01(value) * 100}%` }} />
      </div>
    </div>
  );
}

function UploadPairScoreCard({ result, tone }: { result: UploadPairResult; tone: "teal" | "rose" | "indigo" }) {
  const bar = tone === "rose" ? "bg-rose-500" : tone === "indigo" ? "bg-indigo-600" : "bg-teal-600";
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-sm font-semibold">NASH</span>
        <span className="font-mono text-sm">{formatScore(result.nash_score)}</span>
      </div>
      <div className="mb-2 h-2 rounded bg-white">
        <div className={`h-2 rounded ${bar}`} style={{ width: `${clamp01(result.nash_score) * 100}%` }} />
      </div>
      <div className="grid grid-cols-4 gap-2 text-xs">
        <MiniMetric label="Baseline" value={formatScore(result.baseline_score)} />
        <MiniMetric label="NASH" value={formatScore(result.nash_score)} />
        <MiniMetric label="Text" value={formatScore(result.text_score)} />
        <MiniMetric label="Numeric" value={formatScore(result.numeric_score)} />
      </div>
    </div>
  );
}

function groupRows(rows: BatchResultRow[], keyFor: (row: BatchResultRow) => string) {
  const map = new Map<string, BatchResultRow[]>();
  rows.forEach((row) => {
    const key = keyFor(row);
    map.set(key, [...(map.get(key) ?? []), row]);
  });
  return [...map.entries()].map(([key, groupedRows]) => ({ key, rows: groupedRows }));
}

function tripletTitle(s1: BatchResultRow | undefined, s2: BatchResultRow | undefined, fallback: string) {
  return (s1?.title ?? s2?.title ?? fallback).replace(/: anchor-(s1|s2|positive|negative)$/i, "");
}

function isTripletS1Row(row: BatchResultRow) {
  return /(^|[_\s-])(s1|positive)$/i.test(`${row.role ?? ""}_${row.id}`);
}

function isTripletS2Row(row: BatchResultRow) {
  return /(^|[_\s-])(s2|negative)$/i.test(`${row.role ?? ""}_${row.id}`);
}

function crosspairTitle(pairA: BatchResultRow | undefined, pairB: BatchResultRow | undefined, fallback: string) {
  return (pairA?.title ?? pairB?.title ?? fallback).replace(/: pair [AB]$/i, "");
}

function candidateRows(rows: BatchResultRow[]) {
  return [...rows].sort((left, right) => candidateIndex(left) - candidateIndex(right));
}

function candidateIndex(row: BatchResultRow) {
  const match = /candidate[\s_]*(\d+)/i.exec(`${row.role ?? ""}_${row.id}_${row.title}`);
  return match ? Number(match[1]) : 9999;
}

function TraceNumberList({ title, numbers }: { title: string; numbers: NashTrace["numeric_extraction"]["sentence1_numbers"] }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-2">
      <div className="mb-2 text-xs font-semibold uppercase text-slate-500">{title}</div>
      <div className="grid gap-1">
        {numbers.length ? numbers.map((number, index) => (
          <div key={`${number.text}-${number.start}-${index}`} className="rounded bg-white px-2 py-1 text-xs">
            <span className="font-mono font-semibold">{number.raw ?? number.text}</span>
            <span className="ml-2 text-slate-600">value {formatScore(number.normalized_value ?? number.value)} · [{number.start}, {number.end})</span>
            {number.unit && <span className="ml-2 text-slate-500">{number.unit}</span>}
          </div>
        )) : <span className="text-xs text-slate-500">No numeric mentions</span>}
      </div>
    </div>
  );
}

function CodeLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[72px_1fr] items-center gap-2 rounded-md border border-slate-200 bg-slate-950 px-2 py-2 text-sm">
      <span className="text-xs font-semibold text-slate-400">{label}</span>
      <code className="font-mono text-teal-100">{value}</code>
    </div>
  );
}

function ScoreBar({ label, value, tone }: { label: string; value: number; tone: "slate" | "teal" | "indigo" | "amber" }) {
  const color = {
    slate: "bg-slate-500",
    teal: "bg-teal-600",
    indigo: "bg-indigo-600",
    amber: "bg-amber-500"
  }[tone];
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-semibold text-slate-600">{label}</span>
        <span className="font-mono">{formatScore(value)}</span>
      </div>
      <div className="h-2 rounded bg-slate-100">
        <div className={`h-2 rounded ${color}`} style={{ width: `${clamp01(value) * 100}%` }} />
      </div>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-slate-100 px-2 py-2">
      <div className="text-xs font-semibold text-slate-600">{label}</div>
      <div className="mt-1 truncate font-mono text-lg font-semibold">{value}</div>
    </div>
  );
}

function Notice({ tone, children }: { tone: "info" | "error"; children: ReactNode }) {
  const className = tone === "error" ? "border-rose-300 bg-rose-50 text-rose-900" : "border-teal-200 bg-teal-50 text-teal-950";
  return <div className={`rounded-md border px-3 py-2 text-sm ${className}`}>{children}</div>;
}

async function scoreBatch(
  pairs: BatchPair[],
  baselines: ModelOption[],
  setRows: (rows: BatchResultRow[] | ((current: BatchResultRow[]) => BatchResultRow[])) => void,
  setTraces: (traces: NashTrace[] | ((current: NashTrace[]) => NashTrace[])) => void,
  setLoading: (value: boolean) => void,
  onError: (message: string) => void
) {
  setLoading(true);
  setRows([]);
  setTraces([]);
  try {
    for (const pair of pairs) {
      const response = await fetchBaselineComparison({
        sentence1: pair.sentence1,
        sentence2: pair.sentence2,
        staticMode: false,
        baselines
      });
      const nextRows = comparisonToRows(response, pair);
      setRows((current) => [...current, ...nextRows]);
      setTraces((current) => [...current, ...response.traces]);
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : "Batch scoring failed.");
  } finally {
    setLoading(false);
  }
}

function comparisonToRows(response: BaselineComparisonResponse, pair?: BatchPair): BatchResultRow[] {
  const basePair = pair ?? {
    id: "interactive_example",
    title: "Interactive sentence pair",
    sentence1: response.input.sentence1,
    sentence2: response.input.sentence2
  };
  return response.rows.map((row) => ({
    ...basePair,
    baseline_id: row.baseline_id,
    baseline_label: row.baseline_label,
    backend: row.backend,
    baseline_score: row.baseline_score,
    nash_score: row.nash_score,
    text_score: row.text_score,
    numeric_score: row.numeric_score
  }));
}

function parseUploadedPairs(text: string, filename: string): BatchPair[] {
  if (filename.toLowerCase().endsWith(".csv")) return parseCsvPairs(text);
  const parsed = JSON.parse(text);
  const records = Array.isArray(parsed) ? parsed : parsed.records ?? parsed.data ?? parsed.examples ?? parsed.pairs ?? [];
  if (!Array.isArray(records)) return [];
  return records.map((record, index) => normalizePairRecord(record, index)).filter(Boolean) as BatchPair[];
}

function parseBatchPayload(text: string, filename: string): { pairs: BatchPair[]; rows?: BatchResultRow[]; traces?: NashTrace[]; evaluation?: EvaluationSummary; task?: string } {
  if (filename.toLowerCase().endsWith(".csv")) throw uploadFormatError();
  const parsed = JSON.parse(text);
  if (!parsed || typeof parsed !== "object" || !isRowResultPayload(parsed)) throw uploadFormatError();

  const task = String((parsed as Record<string, unknown>).task ?? "").toLowerCase();
  if (!isSupportedUploadTask(task)) throw uploadFormatError();

  const rows = (parsed as { rows: unknown[] }).rows.map((row, index) => normalizeCliResultRow(row, index, task));
  if (!rows.length || rows.some((row) => !row || !isValidTaskLevelRow(row, task))) throw uploadFormatError();

  return {
    pairs: [],
    rows: rows as BatchResultRow[],
    traces: Array.isArray((parsed as Record<string, unknown>).traces) ? (parsed as Record<string, unknown>).traces as NashTrace[] : [],
    task
  };
}

function isRowResultPayload(value: object) {
  const item = value as Record<string, unknown>;
  return item.kind === "nash_evaluation_results" && typeof item.task === "string" && Array.isArray(item.rows);
}

function isSupportedUploadTask(task: string) {
  return task === "triplet" || task === "crosspair" || task === "listwise";
}

function isValidTaskLevelRow(row: BatchResultRow | null, task: string) {
  const pairCount = row?.pair_results?.length ?? 0;
  if (!row || row.task !== task || pairCount === 0) return false;
  if (task === "triplet" || task === "crosspair") return pairCount === 2;
  if (task === "listwise") return pairCount >= 1;
  return false;
}

function uploadFormatError() {
  return new Error("Format error: upload a completed triplet, crosspair, or listwise NASH evaluation result JSON.");
}

function isEvaluationSummaryPayload(value: object) {
  const item = value as Record<string, unknown>;
  return typeof item.task === "string" && !Array.isArray(item.rows) && item.overall && typeof item.overall === "object";
}

function normalizeEvaluationSummary(item: Record<string, unknown>): EvaluationSummary {
  const overall = item.overall as Record<string, unknown>;
  const model = item.model && typeof item.model === "object" ? item.model as Record<string, unknown> : {};
  const task = String(item.task ?? item.protocol ?? "evaluation");
  const accuracy = numberOrNull(overall.accuracy);
  const correct = numberOrNull(overall.correct);
  const total = numberOrNull(overall.total);
  const tau = numberOrNull(overall.kendall_tau_b_mean ?? overall.mean);
  const exactOrder = numberOrNull(item.exact_order_accuracy);
  const metricValue = accuracy !== null
    ? `${formatScore(accuracy)}${correct !== null && total !== null ? ` (${correct}/${total})` : ""}`
    : tau !== null
      ? formatScore(tau)
      : exactOrder !== null
        ? formatScore(exactOrder)
        : "n/a";
  const metricLabel = accuracy !== null ? "Accuracy" : tau !== null ? "Kendall tau" : exactOrder !== null ? "Exact order" : "Metric";
  return {
    task,
    model: String(model.display_name ?? model.key ?? item.model ?? "n/a"),
    threshold: item.threshold === undefined ? "n/a" : String(item.threshold),
    records: String(item.n_records ?? overall.total ?? overall.count ?? "n/a"),
    metricLabel,
    metricValue,
    runtime: item.runtime_sec === undefined ? "n/a" : `${compactFmt.format(Number(item.runtime_sec))}s`
  };
}

function normalizeCliResultRow(row: unknown, index: number, fallbackTask = ""): BatchResultRow | null {
  if (!row || typeof row !== "object") return null;
  const item = row as Record<string, unknown>;
  const pairResults = normalizeUploadPairResults(item.pair_results ?? item.pairs);
  const sentence1 = stringFromAliases(item, ["sentence1", "sentence_a", "sentenceA", "anchor", "s1"]) || pairResults[0]?.sentence1 || "";
  const sentence2 = stringFromAliases(item, ["sentence2", "sentence_b", "sentenceB", "s2"]) || taskSentence2Preview(item, pairResults);
  if (!sentence1 || !sentence2) return null;
  const model = String(item.model ?? item.baseline_id ?? `model_${index + 1}`);
  const baselineScores = pairResults.map((result) => result.baseline_score);
  const nashScores = pairResults.map((result) => result.nash_score);
  const textScores = pairResults.map((result) => result.text_score);
  const numericScores = pairResults.map((result) => result.numeric_score);
  return {
    id: String(item.id ?? `row_${index + 1}`),
    title: String(item.title ?? `Row ${index + 1}`),
    sentence1,
    sentence2,
    task: typeof item.task === "string" ? item.task.toLowerCase() : fallbackTask,
    role: typeof item.role === "string" ? item.role : undefined,
    baseline_id: model,
    baseline_label: String(item.baseline_label ?? model),
    backend: String(item.backend ?? ""),
    baseline_score: numberFromValue(item.baseline_score, average(baselineScores)),
    nash_score: numberFromValue(item.nash_score ?? item.score, average(nashScores)),
    text_score: numberFromValue(item.text_score, average(textScores)),
    numeric_score: numberFromValue(item.numeric_score, average(numericScores)),
    pair_results: pairResults.length ? pairResults : undefined
  };
}

function normalizeUploadPairResults(value: unknown): UploadPairResult[] {
  if (!Array.isArray(value)) return [];
  return value.map((record, index) => {
    if (!record || typeof record !== "object") return null;
    const item = record as Record<string, unknown>;
    const sentence1 = stringFromAliases(item, ["sentence1", "sentence_a", "sentenceA", "anchor"]);
    const sentence2 = stringFromAliases(item, ["sentence2", "sentence_b", "sentenceB", "s1", "s2"]);
    if (!sentence1 || !sentence2) return null;
    return {
      label: String(item.label ?? item.role ?? `Pair ${index + 1}`),
      sentence1,
      sentence2,
      masked_sentence1: typeof item.masked_sentence1 === "string" ? item.masked_sentence1 : undefined,
      masked_sentence2: typeof item.masked_sentence2 === "string" ? item.masked_sentence2 : undefined,
      baseline_score: numberFromValue(item.baseline_score, 0),
      nash_score: numberFromValue(item.nash_score ?? item.score, 0),
      text_score: numberFromValue(item.text_score, 0),
      numeric_score: numberFromValue(item.numeric_score, 0),
      numbers_sentence1: normalizeGeneratedNumbers(item.numbers_sentence1),
      numbers_sentence2: normalizeGeneratedNumbers(item.numbers_sentence2),
      alignment: normalizeGeneratedAlignment(item.alignment)
    };
  }).filter(Boolean) as UploadPairResult[];
}

function normalizeGeneratedNumbers(value: unknown): GeneratedNumber[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.map((record, index) => {
    if (!record || typeof record !== "object") return null;
    const item = record as Record<string, unknown>;
    const raw = String(item.raw ?? item.text ?? "");
    if (!raw) return null;
    return {
      index: numberFromValue(item.index, index),
      raw,
      value: numberFromValue(item.value, Number(raw.replace(/,/g, ""))),
      start: numberFromValue(item.start, 0),
      end: numberFromValue(item.end, 0)
    };
  }).filter(Boolean) as GeneratedNumber[];
}

function normalizeGeneratedAlignment(value: unknown): GeneratedAlignment | undefined {
  if (!value || typeof value !== "object") return undefined;
  const item = value as Record<string, unknown>;
  const matrix = Array.isArray(item.matrix)
    ? item.matrix.map((row) => Array.isArray(row) ? row.map((cell) => numberFromValue(cell, 0)) : [])
    : undefined;
  const selected = Array.isArray(item.selected_alignment_edges)
    ? item.selected_alignment_edges.map((edge) => {
      if (!edge || typeof edge !== "object") return null;
      const edgeItem = edge as Record<string, unknown>;
      return {
        direction: String(edgeItem.direction ?? "sentence1_to_sentence2"),
        source_index: numberFromValue(edgeItem.source_index, 0),
        target_index: numberFromValue(edgeItem.target_index, 0),
        context_similarity: numberFromValue(edgeItem.context_similarity ?? edgeItem.contextual_similarity, 0),
        aligned: Boolean(edgeItem.aligned ?? edgeItem.valid),
        magnitude_similarity: edgeItem.magnitude_similarity === undefined ? undefined : numberFromValue(edgeItem.magnitude_similarity, 0),
        numeric_contribution: edgeItem.numeric_contribution === undefined ? undefined : numberFromValue(edgeItem.numeric_contribution, 0)
      };
    }).filter(Boolean) as GeneratedAlignmentEdge[]
    : undefined;
  return matrix || selected ? { matrix, selected_alignment_edges: selected } : undefined;
}

function taskSentence2Preview(item: Record<string, unknown>, pairResults: UploadPairResult[]) {
  const task = String(item.task ?? "").toLowerCase();
  if (task === "triplet") {
    const directS1 = stringFromAliases(item, ["candidate1", "s1"]);
    const directS2 = stringFromAliases(item, ["candidate2", "s2"]);
    if (directS1 || directS2) return [directS1, directS2].filter(Boolean).join(" / ");
  }
  if (task === "crosspair" && pairResults.length >= 2) {
    return `${pairResults[0].sentence1} / ${pairResults[0].sentence2} / ${pairResults[1].sentence1} / ${pairResults[1].sentence2}`;
  }
  return pairResults.map((result) => result.sentence2).join(" / ");
}

function parseCsvPairs(text: string): BatchPair[] {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return [];
  const header = splitCsvLine(lines[0]).map((cell) => cell.trim().toLowerCase());
  return lines.slice(1).map(splitCsvLine).map((columns, index) => {
    const record = Object.fromEntries(header.map((key, columnIndex) => [key, columns[columnIndex]?.trim() ?? ""]));
    return normalizePairRecord(record, index);
  }).filter(Boolean) as BatchPair[];
}

function normalizePairRecord(record: unknown, index: number): BatchPair | null {
  if (!record || typeof record !== "object") return null;
  const item = record as Record<string, unknown>;
  const sentence1 = stringFromAliases(item, ["sentence1", "s1", "sentence_a", "text1", "premise"]);
  const sentence2 = stringFromAliases(item, ["sentence2", "s2", "sentence_b", "text2", "hypothesis"]);
  if (!sentence1 || !sentence2) return null;
  return {
    id: String(item.id ?? `pair_${index + 1}`),
    title: String(item.title ?? item.label ?? `Pair ${index + 1}`),
    sentence1,
    sentence2,
    task: typeof item.task === "string" ? item.task : undefined,
    role: typeof item.role === "string" ? item.role : undefined
  };
}

function splitCsvLine(line: string) {
  const cells: string[] = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') quoted = !quoted;
    else if (char === "," && !quoted) {
      cells.push(current);
      current = "";
    } else current += char;
  }
  cells.push(current);
  return cells;
}

function stringFromAliases(item: Record<string, unknown>, aliases: string[]) {
  for (const alias of aliases) {
    const value = item[alias];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function numberOrNull(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function numberFromValue(value: unknown, fallback: number) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function sentencePreview(row: BatchResultRow) {
  if (row.pair_results?.length) {
    return truncateText(row.pair_results.flatMap((result) => [result.sentence1, result.sentence2]).join(" / "), 150);
  }
  return `${truncateText(row.sentence1, 72)} / ${truncateText(row.sentence2, 72)}`;
}

function truncateText(value: string, maxLength: number) {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`;
}

function selectedModels(models: ModelOption[], selectedIds: string[]) {
  const selected = selectedIds.map((id) => models.find((model) => model.id === id)).filter(Boolean) as ModelOption[];
  return selected.length ? selected : fallbackModels;
}

function summarizeBatch(rows: BatchResultRow[]) {
  return {
    count: rows.length,
    avgBaseline: average(rows.map((row) => row.baseline_score)),
    avgNash: average(rows.map((row) => row.nash_score)),
    avgText: average(rows.map((row) => row.text_score)),
    avgNumeric: average(rows.map((row) => row.numeric_score))
  };
}

function average(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function heatColor(value: number) {
  const clamped = clamp01(value);
  return `hsl(174 ${24 + clamped * 48}% ${96 - clamped * 34}%)`;
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function formatScore(value: number | null | undefined) {
  return scoreFmt.format(Number(value ?? 0));
}

function parseInitialTab(value: string | null, pathname: string): TabId {
  if (value === "cases" || value === "batch" || value === "example") return value;
  if (value === "toy") return "example";
  if (pathname.replace(/\/+$/, "").endsWith("/demo")) return "batch";
  return "batch";
}

function defaultBaselines(models: ModelOption[]) {
  return selectedModels(models, defaultModelIds);
}
